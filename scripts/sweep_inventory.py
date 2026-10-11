"""Track every result file on Sherlock: what exists there, what is here, and when Sherlock would purge it.

Sherlock deletes files on $SCRATCH 90 days after their last change, and only lyxia can reach it. The fetch
(scripts/sweep_fetch.py) copies what the pages and figures read; this keeps the full picture, so that anything else
worth having for comparison or validation is named and can be copied before it is gone.

One command on the Mac (one password and one Duo prompt):

    ssh lyxia@login.sherlock.stanford.edu "$(python3 scripts/sweep_inventory.py remote)" | python3 scripts/sweep_inventory.py receive

  remote    print the command Sherlock runs: one line per file (root, bytes, time of last change, path) under the
            result directories, and the size of every top-level directory of the scratch space
  receive   store that listing as progress/sweep_status/inventory_<UTC time>.tsv.gz and print the report
  report    the newest listing by kind of file: how many and how much is there, how much is here, whether the fetch
            list covers it, and the first date a file of that kind would be purged
  extra KIND [KIND ...] [--max-mb N] [--match REGEX]
            write progress/sweep_status/extra_wanted.txt: every file of those kinds that is not in the fetch list
            (optionally only files up to N MB, or whose path matches REGEX; the kind `any` takes every file, for use
            with --match). scripts/sweep_fetch.py then asks for them too.

Kinds: record (run.toml, input.toml), diagnostics (the CSVs), state, slice, reference (ref_N*), restricted (rstate_N*),
log, and other (the codes' own outputs and checkpoints).
"""
import gzip
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data  # noqa: E402

SCRATCH = "/scratch/users/lyxia"
ROOTS = ("mhd-results-sweep", "mhd-results-ramses2")        # result directories that are listed file by file
PURGE_DAYS = 90
STORE = sweep_data.SITE/"sweep_status"
EXTRA = STORE/"extra_wanted.txt"
KINDS = (("record", re.compile(r"(run|input)\.toml$")), ("diagnostics", re.compile(r"(diagnostics|dense_diagnostics|spectra)\.csv$")),
         ("state", re.compile(r"state_\d+\.bin$")), ("slice", re.compile(r"slice_\d+\.bin$")), ("reference", re.compile(r"ref_N\d+_\d+\.bin$")),
         ("restricted", re.compile(r"rstate_N\d+_\d+\.bin$")), ("log", re.compile(r"\.(log|out|err|txt)$")))


def shown(path):
    try:return str(path.relative_to(sweep_data.ROOT))
    except ValueError:return str(path)


def kind_of(path):
    name = path.rsplit("/", 1)[-1]
    for kind, pattern in KINDS:
        if pattern.search(name):return kind
    return "other"


def cmd_remote():
    finds = "; ".join(f"[ -d {SCRATCH}/{r} ] && (cd {SCRATCH}/{r} && find . -type f -printf '{r}\\t%s\\t%T@\\t%P\\n')" for r in ROOTS)
    print(f"{finds}; du -s --block-size=1 {SCRATCH}/*/ 2>/dev/null | sed 's/^/DU\\t/'")


def parse(lines):
    """({root: [(path, bytes, mtime)]}, {directory: bytes}); anything else (a login banner) is ignored."""
    files, du = {}, {}
    for line in lines:
        cells = line.rstrip("\n").split("\t")
        if cells[0] == "DU" and len(cells) == 3 and cells[1].isdigit():du[cells[2].rstrip("/").rsplit("/", 1)[-1]] = int(cells[1])
        elif len(cells) == 4 and cells[0] in ROOTS and cells[1].isdigit():
            try:files.setdefault(cells[0], []).append((cells[3], int(cells[1]), float(cells[2])))
            except ValueError:pass
    return files, du


def newest():
    found = sorted(STORE.glob("inventory_*.tsv.gz"))
    return found[-1] if found else None


def load(path=None):
    path = path or newest()
    if path is None:sys.exit("No inventory yet. Run the command at the top of scripts/sweep_inventory.py.")
    with gzip.open(path, "rt") as f:return path, *parse(f)


def cmd_receive():
    text = sys.stdin.read();files, du = parse(text.splitlines())
    if not files:sys.exit("No file listing arrived (wrong password, no connection, or the result directories are gone).")
    STORE.mkdir(parents=True, exist_ok=True)
    path = STORE/f"inventory_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tsv.gz"
    with gzip.open(path, "wt") as f:f.write(text)
    print(f"Stored {shown(path)}: {sum(len(v) for v in files.values()):,} files.")
    cmd_report(path)


def cmd_report(path=None):
    path, files, du = load(path);sw = sweep_data.Sweep()
    wanted = {p for p, _, _ in sw.wanted()} | (set(EXTRA.read_text().split()) if EXTRA.is_file() else set())
    taken = time.gmtime(path.stat().st_mtime)
    print(f"Inventory {path.name} (listed {time.strftime('%Y-%m-%d %H:%M UTC', taken)})")
    if du:
        print("Scratch space by top-level directory:")
        for name, size in sorted(du.items(), key=lambda x: -x[1]):print(f"  {size/1e9:10.1f} GB  {name}")
    summary = {}
    for root, rows in files.items():
        local = sw.results if root == ROOTS[0] else None
        print(f"\n{root}: {len(rows):,} files, {sum(r[1] for r in rows)/1e9:,.1f} GB")
        print(f"  {'kind':12s} {'files':>9s} {'GB there':>10s} {'files here':>11s} {'GB here':>9s} {'in fetch list':>14s} {'first purge':>12s}")
        by = {}
        for p, size, mtime in rows:
            k = by.setdefault(kind_of(p), dict(files=0, bytes=0, here=0, here_bytes=0, listed=0, oldest=mtime))
            k["files"] += 1;k["bytes"] += size;k["oldest"] = min(k["oldest"], mtime)
            if p in wanted:k["listed"] += 1
            if local is not None and (local/p).is_file() and (local/p).stat().st_size == size:k["here"] += 1;k["here_bytes"] += size
        for kind in [k for k, _ in KINDS] + ["other"]:
            if kind not in by:continue
            k = by[kind];purge = datetime.fromtimestamp(k["oldest"], timezone.utc) + timedelta(days=PURGE_DAYS)
            print(f"  {kind:12s} {k['files']:9,d} {k['bytes']/1e9:10.2f} {k['here']:11,d} {k['here_bytes']/1e9:9.2f} {k['listed']:14,d} {purge:%Y-%m-%d}")
            k["first_purge"] = f"{purge:%Y-%m-%d}"
        summary[root] = by
    (STORE/"inventory_latest.json").write_text(json.dumps(dict(listing=path.name, roots=summary, scratch=du), indent=1))
    print("\nA kind with fewer files here than there is not fully copied. `extra KIND ...` adds the rest of a kind to the fetch.")


def cmd_extra(argv):
    limit = None
    if "--max-mb" in argv:
        i = argv.index("--max-mb");limit = float(argv[i + 1])*1e6;argv = argv[:i] + argv[i + 2:]
    match = None
    if "--match" in argv:
        i = argv.index("--match");match = re.compile(argv[i + 1]);argv = argv[:i] + argv[i + 2:]
    kinds = set(argv)
    if not kinds or kinds - {k for k, _ in KINDS} - {"other", "any"}:sys.exit(__doc__)
    path, files, _ = load();sw = sweep_data.Sweep();base = {p for p, _, _ in sw.wanted()}
    rows = [(p, s) for p, s, _ in files.get(ROOTS[0], []) if ("any" in kinds or kind_of(p) in kinds) and p not in base
            and (limit is None or s <= limit) and (match is None or match.search(p))]
    keep = set(EXTRA.read_text().split()) if EXTRA.is_file() else set()
    keep |= {p for p, _ in rows};EXTRA.write_text("".join(p + "\n" for p in sorted(keep)))
    need = [(p, s) for p, s in rows if not ((sw.results/p).is_file() and (sw.results/p).stat().st_size == s)]
    print(f"{shown(EXTRA)} now names {len(keep):,} files beyond the figures' list; "
          f"{len(need):,} of the ones just added are not here yet ({sum(s for _, s in need)/1e9:.2f} GB). Run the fetch command to copy them.")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "remote":cmd_remote()
    elif command == "receive":cmd_receive()
    elif command == "report":cmd_report()
    elif command == "extra":cmd_extra(sys.argv[2:])
    else:sys.exit(__doc__)
