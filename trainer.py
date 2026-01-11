#!/usr/bin/env python3
# trainer.py
import csv
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

APP_NAME = "Vokatrainer CLI"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".vokatrainer_config.json")
STATS_PATH = os.path.join(os.path.expanduser("~"), ".vokatrainer_stats.json")


# -----------------------------
# Utils
# -----------------------------
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause(msg: str = "Enter drücken...") -> None:
    input(msg)


def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_answer(
    s: str,
    case_sensitive: bool,
    ignore_punct: bool,
    accent_insensitive: bool,
) -> str:
    s = s.strip()
    if accent_insensitive:
        s = strip_accents(s)
    if not case_sensitive:
        s = s.lower()
    if ignore_punct:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def is_close_enough(user: str, correct: str, tolerance: str) -> bool:
    if tolerance == "off":
        return user == correct
    dist = levenshtein(user, correct)
    L = max(len(user), len(correct))
    if L <= 4:
        max_dist = 0 if tolerance == "low" else 1
    elif L <= 8:
        max_dist = 1 if tolerance == "low" else 2
    else:
        max_dist = 2 if tolerance == "low" else 3
    return dist <= max_dist


def safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


# -----------------------------
# CSV picker (arrow keys)
# -----------------------------
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def list_csv_files_in_script_dir() -> List[str]:
    base = script_dir()
    files = []
    for f in os.listdir(base):
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(base, f)):
            files.append(f)
    return sorted(files)


def pick_csv_with_arrows() -> Optional[str]:
    """
    Arrow-key selector for CSV files in the same directory as this script.
    - Up/Down cycles
    - Enter selects
    - Esc or q cancels
    Falls back to numbered selection if curses isn't available.
    Returns absolute path or None.
    """
    files = list_csv_files_in_script_dir()
    if not files:
        clear_screen()
        print("Keine .csv-Dateien im Programmordner gefunden.")
        pause()
        return None

    try:
        import curses  # type: ignore
    except Exception:
        return pick_csv_fallback(files)

    def _ui(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)

        idx = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            title = "CSV auswählen (↑/↓, Enter = laden, Esc/q = zurück)"
            stdscr.addstr(0, 0, title[: max(0, w - 1)])

            for i, name in enumerate(files):
                line = i + 2
                if line >= h:
                    break
                prefix = "➤ " if i == idx else "  "
                text = (prefix + name)[: max(0, w - 1)]
                if i == idx:
                    try:
                        stdscr.addstr(line, 0, text, curses.A_REVERSE)
                    except Exception:
                        stdscr.addstr(line, 0, text)
                else:
                    stdscr.addstr(line, 0, text)

            hint = "Tipp: Lege deine CSV-Dateien in denselben Ordner wie trainer.py"
            if h - 1 > 0:
                stdscr.addstr(h - 1, 0, hint[: max(0, w - 1)])

            stdscr.refresh()
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(files)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(files)
            elif key in (10, 13, curses.KEY_ENTER):
                return os.path.join(script_dir(), files[idx])
            elif key in (27, ord("q"), ord("Q")):  # ESC or q
                return None

    try:
        return curses.wrapper(_ui)
    except Exception:
        # Terminal doesn't support curses properly -> fallback
        return pick_csv_fallback(files)


def pick_csv_fallback(files: List[str]) -> Optional[str]:
    while True:
        clear_screen()
        print("CSV auswählen (Fallback, keine Pfeiltasten verfügbar)")
        print("-" * 50)
        for i, f in enumerate(files, start=1):
            print(f"{i}) {f}")
        print("0) Zurück")
        print("-" * 50)
        c = input("Auswahl: ").strip()
        if c == "0":
            return None
        idx = safe_int(c, -1) - 1
        if 0 <= idx < len(files):
            return os.path.join(script_dir(), files[idx])


# -----------------------------
# Data formats
# -----------------------------
@dataclass
class LoadedCSV:
    path: str
    mode: str  # "vocab" or "verbs"
    rows: List[Dict[str, str]]
    source_lang_key: str
    target_key: Optional[str]


def sniff_delimiter(sample: str) -> str:
    first = sample.splitlines()[0] if sample.splitlines() else sample
    candidates = [";", ",", "\t"]
    counts = {c: first.count(c) for c in candidates}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ";"


def load_csv(path: str) -> LoadedCSV:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        head = f.read(2048)
    delim = sniff_delimiter(head)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError("CSV hat keine Kopfzeile (Spaltennamen).")

        fields = [(c or "").strip() for c in reader.fieldnames if c]
        rows: List[Dict[str, str]] = []
        for r in reader:
            if not r:
                continue
            rr = {(k or "").strip(): (v.strip() if isinstance(v, str) else "") for k, v in r.items()}
            if all(not (v or "").strip() for v in rr.values()):
                continue
            rows.append(rr)

    fieldset = set(fields)

    if {"de", "base", "past", "pp"}.issubset(fieldset):
        return LoadedCSV(path=path, mode="verbs", rows=rows, source_lang_key="de", target_key=None)

    if "de" in fieldset:
        possible_targets = [k for k in fields if k in ("en", "fr")]
        if len(possible_targets) == 1:
            return LoadedCSV(path=path, mode="vocab", rows=rows, source_lang_key="de", target_key=possible_targets[0])
        other_cols = [k for k in fields if k != "de"]
        if len(other_cols) == 1:
            return LoadedCSV(path=path, mode="vocab", rows=rows, source_lang_key="de", target_key=other_cols[0])

    raise ValueError(
        "Unbekanntes CSV-Format.\n"
        "Erwartet:\n"
        "  Vokabeln: de;en   oder  de;fr\n"
        "  Verben:   de;base;past;pp"
    )


# -----------------------------
# Config / Stats
# -----------------------------
DEFAULT_CONFIG = {
    "case_sensitive": False,
    "ignore_punct": True,
    "accent_insensitive": False,
    "typo_tolerance": "normal",  # off | low | normal
    "repeat_wrong": True,
    "batch_size": 20,
    "order": "sequential",  # sequential | random
    "verb_input": "separate",  # separate | all
    "allow_synonyms_pipe": True,  # "to run|run"
    "wrong_reinsert_gap": 8,  # minimum distance when re-inserting wrong cards
}


def load_json(path: str, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return dict(default)
    except Exception:
        return dict(default)


def save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -----------------------------
# Trainer logic
# -----------------------------
def split_into_batches(n: int, batch_size: int) -> List[Tuple[int, int]]:
    batches = []
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batches.append((start, end))
        start = end
    return batches


def parse_synonyms(correct_raw: str, allow_pipe: bool) -> List[str]:
    if allow_pipe and "|" in correct_raw:
        return [c.strip() for c in correct_raw.split("|") if c.strip()]
    return [correct_raw.strip()]


def check_answer(user_raw: str, correct_raw: str, cfg: dict) -> Tuple[bool, str]:
    user_n = normalize_answer(user_raw, cfg["case_sensitive"], cfg["ignore_punct"], cfg["accent_insensitive"])
    candidates = parse_synonyms(correct_raw, cfg.get("allow_synonyms_pipe", True))
    best_display = candidates[0] if candidates else correct_raw

    for cand in candidates:
        cand_n = normalize_answer(cand, cfg["case_sensitive"], cfg["ignore_punct"], cfg["accent_insensitive"])
        if is_close_enough(user_n, cand_n, cfg["typo_tolerance"]):
            return True, cand
    return False, best_display


def reinsert_wrong(queue: List[int], idx: int, cfg: dict) -> None:
    gap = int(cfg.get("wrong_reinsert_gap", 8))
    pos = min(len(queue), max(0, gap))
    insert_at = random.randint(pos, len(queue)) if len(queue) >= pos else len(queue)
    queue.insert(insert_at, idx)


HELP_TEXT = """Optionen im Abfragemodus:
  ?        Hilfe anzeigen
  :q       Session beenden (zurück ins Menü)
  :exit    Session beenden (zurück ins Menü)
  :menu    Session beenden (zurück ins Menü)
  :skip    Karte überspringen (kommt ans Ende)
  :show    Lösung anzeigen (zählt als falsch)
"""


def handle_session_command(cmd: str) -> str:
    c = cmd.strip().lower()
    if c == "?":
        return "help"
    if c in (":q", ":exit", ":menu"):
        return "quit"
    if c == ":skip":
        return "skip"
    if c == ":show":
        return "show"
    return ""


def choose_batch(total_rows: int, batch_size: int) -> int:
    batches = split_into_batches(total_rows, max(1, batch_size))
    if not batches:
        return 0
    while True:
        clear_screen()
        print(f"Päckchen auswählen (Größe {batch_size})")
        print("-" * 50)
        for idx, (s, e) in enumerate(batches, start=1):
            print(f"{idx}) Zeilen {s+1}-{e} ({e - s} Karten)")
        print("0) Zurück")
        print("-" * 50)
        c = input("Auswahl: ").strip()
        if c == "0":
            return -1
        bi = safe_int(c, -1) - 1
        if 0 <= bi < len(batches):
            return bi


def run_vocab_session(data: LoadedCSV, cfg: dict, batch_index: int, stats: dict) -> None:
    rows = data.rows
    bsize = max(1, int(cfg["batch_size"]))
    batches = split_into_batches(len(rows), bsize)
    if not batches or not (0 <= batch_index < len(batches)):
        print("Ungültiges Päckchen oder leere Datei.")
        pause()
        return

    start, end = batches[batch_index]
    indices = list(range(start, end))
    if cfg["order"] == "random":
        random.shuffle(indices)

    queue = indices[:]
    asked = correct = wrong = skipped = 0

    while queue:
        i = queue.pop(0)
        row = rows[i]
        de = row.get("de", "").strip()
        target_key = data.target_key or "en"
        corr = row.get(target_key, "").strip()

        clear_screen()
        print(f"{APP_NAME} — Vokabeln ({target_key})")
        print(f"Datei: {os.path.basename(data.path)}")
        print(f"Päckchen {batch_index+1}/{len(batches)} (Zeilen {start+1}-{end})")
        print("-" * 60)
        print(f"DE: {de}")
        user = input("=> ").strip()

        action = handle_session_command(user)
        if action == "help":
            clear_screen()
            print(HELP_TEXT)
            pause()
            queue.insert(0, i)  # Karte nochmal dran
            continue
        if action == "quit":
            clear_screen()
            print("Session beendet.")
            print(f"Fragen: {asked} | Richtig: {correct} | Falsch: {wrong} | Skip: {skipped}")
            pause()
            return
        if action == "skip":
            skipped += 1
            # Karte ans Ende (aber ohne als falsch zu zählen)
            queue.append(i)
            continue
        if action == "show":
            asked += 1
            wrong += 1
            print(f"➡ Lösung: {corr}")
            if cfg["repeat_wrong"]:
                reinsert_wrong(queue, i, cfg)
            stats.setdefault("sessions", 0)
            stats.setdefault("total_asked", 0)
            stats.setdefault("total_correct", 0)
            stats.setdefault("total_wrong", 0)
            stats["total_asked"] += 1
            stats["total_wrong"] += 1
            pause()
            continue

        asked += 1
        ok, best = check_answer(user, corr, cfg)
        if ok:
            correct += 1
            print("✅ Richtig")
        else:
            wrong += 1
            print(f"❌ Falsch — richtig: {best}")
            if cfg["repeat_wrong"]:
                reinsert_wrong(queue, i, cfg)

        stats.setdefault("sessions", 0)
        stats.setdefault("total_asked", 0)
        stats.setdefault("total_correct", 0)
        stats.setdefault("total_wrong", 0)
        stats["total_asked"] += 1
        stats["total_correct"] += 1 if ok else 0
        stats["total_wrong"] += 0 if ok else 1
        pause()

    stats["sessions"] = stats.get("sessions", 0) + 1
    clear_screen()
    print("Fertig.")
    print(f"Fragen: {asked}")
    print(f"Richtig: {correct}")
    print(f"Falsch:  {wrong}")
    print(f"Skip:    {skipped}")
    pause()


def run_verbs_session(data: LoadedCSV, cfg: dict, batch_index: int, stats: dict) -> None:
    rows = data.rows
    bsize = max(1, int(cfg["batch_size"]))
    batches = split_into_batches(len(rows), bsize)
    if not batches or not (0 <= batch_index < len(batches)):
        print("Ungültiges Päckchen oder leere Datei.")
        pause()
        return

    start, end = batches[batch_index]
    indices = list(range(start, end))
    if cfg["order"] == "random":
        random.shuffle(indices)

    queue = indices[:]
    asked = correct_all = wrong_any = skipped = 0

    while queue:
        i = queue.pop(0)
        row = rows[i]
        de = row.get("de", "").strip()
        base = row.get("base", "").strip()
        past = row.get("past", "").strip()
        pp = row.get("pp", "").strip()

        clear_screen()
        print(f"{APP_NAME} — Unregelmäßige Verben")
        print(f"Datei: {os.path.basename(data.path)}")
        print(f"Päckchen {batch_index+1}/{len(batches)} (Zeilen {start+1}-{end})")
        print("-" * 60)
        print(f"DE: {de}")
        print("-" * 60)

        def read_field(prompt: str) -> str:
            return input(prompt).strip()

        if cfg["verb_input"] == "all":
            user = read_field("Base / Past / PP (z.B. go/went/gone): ")
            action = handle_session_command(user)
            if action == "help":
                clear_screen()
                print(HELP_TEXT)
                pause()
                queue.insert(0, i)
                continue
            if action == "quit":
                clear_screen()
                print("Session beendet.")
                print(f"Verben: {asked} | Komplett richtig: {correct_all} | Nicht komplett: {wrong_any} | Skip: {skipped}")
                pause()
                return
            if action == "skip":
                skipped += 1
                queue.append(i)
                continue
            if action == "show":
                asked += 1
                wrong_any += 1
                print(f"➡ Lösung: {base} / {past} / {pp}")
                if cfg["repeat_wrong"]:
                    reinsert_wrong(queue, i, cfg)
                stats.setdefault("sessions", 0)
                stats.setdefault("total_asked", 0)
                stats.setdefault("total_correct", 0)
                stats.setdefault("total_wrong", 0)
                stats["total_asked"] += 1
                stats["total_wrong"] += 1
                pause()
                continue

            parts = re.split(r"\s*[\/,;]\s*", user)
            parts = [p.strip() for p in parts if p.strip()]
            while len(parts) < 3:
                parts.append("")
            u_base, u_past, u_pp = parts[0], parts[1], parts[2]
        else:
            # separate input, but commands should still work on any field input
            u_base = read_field("Base: ")
            action = handle_session_command(u_base)
            if action == "help":
                clear_screen()
                print(HELP_TEXT)
                pause()
                queue.insert(0, i)
                continue
            if action == "quit":
                clear_screen()
                print("Session beendet.")
                print(f"Verben: {asked} | Komplett richtig: {correct_all} | Nicht komplett: {wrong_any} | Skip: {skipped}")
                pause()
                return
            if action == "skip":
                skipped += 1
                queue.append(i)
                continue
            if action == "show":
                asked += 1
                wrong_any += 1
                print(f"➡ Lösung: {base} / {past} / {pp}")
                if cfg["repeat_wrong"]:
                    reinsert_wrong(queue, i, cfg)
                stats.setdefault("sessions", 0)
                stats.setdefault("total_asked", 0)
                stats.setdefault("total_correct", 0)
                stats.setdefault("total_wrong", 0)
                stats["total_asked"] += 1
                stats["total_wrong"] += 1
                pause()
                continue

            u_past = read_field("Past: ")
            action = handle_session_command(u_past)
            if action in ("help", "quit", "skip", "show"):
                # handle consistently: push card back and act
                if action == "help":
                    clear_screen()
                    print(HELP_TEXT)
                    pause()
                    queue.insert(0, i)
                    continue
                if action == "quit":
                    clear_screen()
                    print("Session beendet.")
                    print(f"Verben: {asked} | Komplett richtig: {correct_all} | Nicht komplett: {wrong_any} | Skip: {skipped}")
                    pause()
                    return
                if action == "skip":
                    skipped += 1
                    queue.append(i)
                    continue
                if action == "show":
                    asked += 1
                    wrong_any += 1
                    print(f"➡ Lösung: {base} / {past} / {pp}")
                    if cfg["repeat_wrong"]:
                        reinsert_wrong(queue, i, cfg)
                    stats.setdefault("sessions", 0)
                    stats.setdefault("total_asked", 0)
                    stats.setdefault("total_correct", 0)
                    stats.setdefault("total_wrong", 0)
                    stats["total_asked"] += 1
                    stats["total_wrong"] += 1
                    pause()
                    continue

            u_pp = read_field("PP:   ")
            action = handle_session_command(u_pp)
            if action in ("help", "quit", "skip", "show"):
                if action == "help":
                    clear_screen()
                    print(HELP_TEXT)
                    pause()
                    queue.insert(0, i)
                    continue
                if action == "quit":
                    clear_screen()
                    print("Session beendet.")
                    print(f"Verben: {asked} | Komplett richtig: {correct_all} | Nicht komplett: {wrong_any} | Skip: {skipped}")
                    pause()
                    return
                if action == "skip":
                    skipped += 1
                    queue.append(i)
                    continue
                if action == "show":
                    asked += 1
                    wrong_any += 1
                    print(f"➡ Lösung: {base} / {past} / {pp}")
                    if cfg["repeat_wrong"]:
                        reinsert_wrong(queue, i, cfg)
                    stats.setdefault("sessions", 0)
                    stats.setdefault("total_asked", 0)
                    stats.setdefault("total_correct", 0)
                    stats.setdefault("total_wrong", 0)
                    stats["total_asked"] += 1
                    stats["total_wrong"] += 1
                    pause()
                    continue

        asked += 1
        ok1, _ = check_answer(u_base, base, cfg)
        ok2, _ = check_answer(u_past, past, cfg)
        ok3, _ = check_answer(u_pp, pp, cfg)

        ok_all = ok1 and ok2 and ok3

        print("-" * 60)
        print(f"Base: {'✅' if ok1 else '❌'}" + ("" if ok1 else f"  richtig: {base}"))
        print(f"Past: {'✅' if ok2 else '❌'}" + ("" if ok2 else f"  richtig: {past}"))
        print(f"PP:   {'✅' if ok3 else '❌'}" + ("" if ok3 else f"  richtig: {pp}"))

        if ok_all:
            correct_all += 1
            print("\n✅ Komplett richtig")
        else:
            wrong_any += 1
            print("\n❌ Nicht komplett richtig")
            if cfg["repeat_wrong"]:
                reinsert_wrong(queue, i, cfg)

        stats.setdefault("sessions", 0)
        stats.setdefault("total_asked", 0)
        stats.setdefault("total_correct", 0)
        stats.setdefault("total_wrong", 0)
        stats["total_asked"] += 1
        stats["total_correct"] += 1 if ok_all else 0
        stats["total_wrong"] += 0 if ok_all else 1
        pause()

    stats["sessions"] = stats.get("sessions", 0) + 1
    clear_screen()
    print("Fertig.")
    print(f"Verben: {asked}")
    print(f"Komplett richtig: {correct_all}")
    print(f"Nicht komplett:   {wrong_any}")
    print(f"Skip:            {skipped}")
    pause()


# -----------------------------
# Menus
# -----------------------------
def settings_menu(cfg: dict) -> dict:
    while True:
        clear_screen()
        print(f"{APP_NAME} — Einstellungen")
        print("-" * 60)
        print(f"1) Groß/Klein prüfen:       {'AN' if cfg['case_sensitive'] else 'AUS'}")
        print(f"2) Satzzeichen ignorieren:  {'AN' if cfg['ignore_punct'] else 'AUS'}")
        print(f"3) Accents ignorieren:      {'AN' if cfg['accent_insensitive'] else 'AUS'}")
        print(f"4) Tippfehler-Toleranz:     {cfg['typo_tolerance']} (off/low/normal)")
        print(f"5) Falsche wiederholen:     {'AN' if cfg['repeat_wrong'] else 'AUS'}")
        print(f"6) Päckchengröße:           {cfg['batch_size']}")
        print(f"7) Reihenfolge:             {cfg['order']} (sequential/random)")
        print(f"8) Verben-Eingabe:          {cfg['verb_input']} (separate/all)")
        print(f"9) Synonyme mit | erlauben: {'AN' if cfg.get('allow_synonyms_pipe', True) else 'AUS'}")
        print(f"10) Abstand bei falsch:     {cfg.get('wrong_reinsert_gap', 8)}")
        print("0) Zurück")
        print("-" * 60)

        c = input("Auswahl: ").strip()
        if c == "1":
            cfg["case_sensitive"] = not cfg["case_sensitive"]
        elif c == "2":
            cfg["ignore_punct"] = not cfg["ignore_punct"]
        elif c == "3":
            cfg["accent_insensitive"] = not cfg["accent_insensitive"]
        elif c == "4":
            val = input("off / low / normal: ").strip().lower()
            if val in ("off", "low", "normal"):
                cfg["typo_tolerance"] = val
        elif c == "5":
            cfg["repeat_wrong"] = not cfg["repeat_wrong"]
        elif c == "6":
            cfg["batch_size"] = max(1, safe_int(input("Zahl (z.B. 20): ").strip(), cfg["batch_size"]))
        elif c == "7":
            val = input("sequential / random: ").strip().lower()
            if val in ("sequential", "random"):
                cfg["order"] = val
        elif c == "8":
            val = input("separate / all: ").strip().lower()
            if val in ("separate", "all"):
                cfg["verb_input"] = val
        elif c == "9":
            cfg["allow_synonyms_pipe"] = not cfg.get("allow_synonyms_pipe", True)
        elif c == "10":
            cfg["wrong_reinsert_gap"] = max(0, safe_int(input("Zahl (z.B. 8): ").strip(), cfg.get("wrong_reinsert_gap", 8)))
        elif c == "0":
            return cfg


def main():
    cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)
    stats = load_json(STATS_PATH, {})
    loaded: Optional[LoadedCSV] = None

    while True:
        clear_screen()
        print(APP_NAME)
        print("-" * 60)
        print("1) Einstellungen")
        print("2) CSV-Datei laden (Pfeiltasten + Enter)")
        print("3) Lernen starten")
        print("4) Statistik anzeigen")
        print("0) Beenden")
        print("-" * 60)
        if loaded:
            print(f"Geladen: {os.path.basename(loaded.path)} (erkannt: {loaded.mode}, Karten: {len(loaded.rows)})")
        else:
            print("Geladen: (keine)")

        c = input("\nAuswahl: ").strip()

        if c == "1":
            cfg = settings_menu(cfg)
            save_json(CONFIG_PATH, cfg)

        elif c == "2":
            path = pick_csv_with_arrows()
            if not path:
                continue
            try:
                loaded = load_csv(path)
                clear_screen()
                print(f"✅ Geladen: {os.path.basename(loaded.path)} (Modus erkannt: {loaded.mode})")
                pause()
            except Exception as e:
                clear_screen()
                print(f"❌ Laden fehlgeschlagen: {e}")
                loaded = None
                pause()

        elif c == "3":
            if not loaded:
                print("❌ Keine CSV geladen.")
                pause()
                continue

            while True:
                clear_screen()
                print("Modus wählen")
                print("-" * 60)
                print("1) Vokabeln abfragen (DE -> Ziel)")
                print("2) Unregelmäßige Verben (DE -> Base/Past/PP)")
                print("0) Zurück")
                print("-" * 60)
                m = input("Auswahl: ").strip()
                if m == "0":
                    break

                bi = choose_batch(len(loaded.rows), int(cfg["batch_size"]))
                if bi < 0:
                    break

                if m == "1":
                    if loaded.mode != "vocab":
                        print("⚠️ Diese CSV sieht nicht wie Vokabeln aus (erkannt: verbs).")
                        pause()
                    run_vocab_session(loaded, cfg, bi, stats)
                    save_json(STATS_PATH, stats)
                    break

                if m == "2":
                    if loaded.mode != "verbs":
                        print("⚠️ Diese CSV sieht nicht wie Verben aus (erkannt: vocab).")
                        pause()
                    run_verbs_session(loaded, cfg, bi, stats)
                    save_json(STATS_PATH, stats)
                    break

        elif c == "4":
            clear_screen()
            print("Statistik")
            print("-" * 60)
            print(f"Sessions:      {stats.get('sessions', 0)}")
            print(f"Fragen gesamt: {stats.get('total_asked', 0)}")
            print(f"Richtig:       {stats.get('total_correct', 0)}")
            print(f"Falsch:        {stats.get('total_wrong', 0)}")
            pause()

        elif c == "0":
            save_json(CONFIG_PATH, cfg)
            save_json(STATS_PATH, stats)
            break


if __name__ == "__main__":
    main()
