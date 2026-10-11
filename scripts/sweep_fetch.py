"""Copy the sweep's result files that the site and the paper figures read from Sherlock to sweep_results/.

One command on the Mac (one password and one Duo prompt):

    python3 scripts/sweep_fetch.py list | ssh lyxia@login.sherlock.stanford.edu 'cd /scratch/users/lyxia/mhd-results-sweep && while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done | tar -cf - -T -' | python3 scripts/sweep_fetch.py receive

  list      the files that are wanted (scripts/sweep_data.py: every part's run.toml and CSVs, and the binary frames
            the figures show; plus any further files named from the inventory, scripts/sweep_inventory.py) and are
            not here yet, one path per line relative to the results directory
  remote    print the command Sherlock runs (the quoted part above): keep the listed files that exist, and write
            them to its output as a tar stream
  receive   unpack that stream into sweep_results/ and report what arrived
  check     what is here and what is missing, without contacting Sherlock

Running it again asks only for what is still missing, so it also picks up runs that finished since the last time.

The same stream brings the sweep's status export (sweep_state/runs.json in the bundle on Sherlock) when it is newer
than progress/sweep_status/export_latest.json. To refresh it first, on Sherlock:

    cd /scratch/users/lyxia/mhd-study/sherlock_sweep && module load python/3.12.1 && python3 tools/sweep.py runs --json sweep_state/runs.json
"""
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data  # noqa: E402

TEXT_BYTES = 12_000           # typical size of a run.toml or CSV, for the estimate only
EXPORT = "../mhd-study/sherlock_sweep/sweep_state/runs.json"      # the status export, relative to the results directory
EXPORT_TAIL = "sweep_state/runs.json"                              # tar stores it without the leading ../


def here(root, path, size):
    p = root/path
    return p.is_file() and (p.stat().st_size == size if size else p.stat().st_size > 0)


EXTRA = sweep_data.SITE/"sweep_status"/"extra_wanted.txt"       # further files to keep, named from the inventory (scripts/sweep_inventory.py)


def extras():
    """Paths beyond the figures' list that scripts/sweep_inventory.py marked for copying (they exist on Sherlock)."""
    return EXTRA.read_text().split() if EXTRA.is_file() else []


def missing(sw):
    """Wanted files that are not here and are not known to be absent on Sherlock."""
    base = [(p, s) for p, s, absent in sw.wanted() if not absent and not here(sw.results, p, s)]
    named = {p for p, _ in base}
    return base + [(p, None) for p in extras() if p not in named and not here(sw.results, p, None)]


def cmd_list():
    sw = sweep_data.Sweep();todo = missing(sw)
    known = sum(s for _, s in todo if s);texts = sum(1 for _, s in todo if not s)
    print(f"Asking Sherlock for {len(todo):,} files (at most {(known + texts*TEXT_BYTES)/1e9:.1f} GB; files that do not "
          f"exist there yet are left out).", file=sys.stderr, flush=True)
    sys.stdout.write(EXPORT + "\n" + "".join(p + "\n" for p, _ in todo))


def cmd_remote():
    # Paths come from this repository's plan (letters, digits, - _ . /), so plain quoting is enough. Files that do
    # not exist are dropped before tar sees them: the stream holds exactly the files that exist.
    print(f'cd {sweep_data.REMOTE} && while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done | tar -cf - -T -')


class Stream:
    """The incoming bytes from the first tar header on. Anything a login script on Sherlock printed before the
    archive would otherwise make the whole stream unreadable; it is kept in `noise` and reported."""

    def __init__(self, raw, probe=1 << 20):
        head = b""
        while len(head) < probe:
            block = raw.read(probe - len(head))
            if not block:break
            head += block
        at = head.find(b"ustar");start = at - 257 if at >= 257 else 0      # the magic sits 257 bytes into a header
        self.noise = head[:start];self.head = head[start:];self.raw = raw

    def read(self, n=-1):
        if self.head:
            out, self.head = (self.head, b"") if n < 0 else (self.head[:n], self.head[n:])
            return out + (self.raw.read() if n < 0 else b"")
        return self.raw.read(n)


def take_export(data, sw):
    """Keep a status export that arrived in the stream if it is newer than the one here; returns a line to print."""
    try:
        new = json.loads(data);stamp = new["time"];new["runs"]
    except (ValueError, KeyError):
        return "  The status export in the stream could not be read; the one here is kept."
    old = (sw.status or {}).get("time", "") if (sw.status or {}).get("source") == "export" else ""
    if stamp <= old:return f"  Status export on Sherlock is of {stamp}: not newer than the one here."
    target = sweep_data.SITE/"sweep_status"/"export_latest.json";target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part");tmp.write_bytes(data);tmp.replace(target)
    return f"  Status export of {stamp} received (the one here was of {old or 'none'})."


def cmd_receive():
    sw = sweep_data.Sweep();root = sw.results;root.mkdir(exist_ok=True)
    wanted = {p: None for p in extras()};wanted.update({p: s for p, s, _ in sw.wanted()})
    count = total = 0;started = shown = time.time();odd = []
    stream = Stream(sys.stdin.buffer)
    if stream.noise:
        print(f"Sherlock printed {len(stream.noise)} bytes before the data (a login script?); skipped: "
              f"{stream.noise[:120].decode(errors='replace')!r}", file=sys.stderr)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if member.name.rsplit("/", 1)[-1].startswith("._"):continue      # macOS tar's attribute entries
                if member.isfile() and member.name.endswith(EXPORT_TAIL):
                    print("\r" + take_export(tar.extractfile(member).read(), sw), file=sys.stderr);continue
                if not member.isfile() or member.name not in wanted:odd.append(member.name);continue
                target = root/member.name;target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part");written = 0
                try:
                    with tar.extractfile(member) as src, open(tmp, "wb") as dst:
                        while True:
                            block = src.read(1 << 20)
                            if not block:break
                            dst.write(block);written += len(block)
                    if written != member.size:raise tarfile.ReadError(f"{member.name} arrived incomplete")
                except BaseException:           # the stream was cut inside this file: leave no partial file
                    tmp.unlink(missing_ok=True);raise
                tmp.replace(target);count += 1;total += member.size
                if time.time() - shown > 2:
                    shown = time.time()
                    print(f"\r  {count:,} files, {total/1e6:,.0f} MB", end="", file=sys.stderr, flush=True)
    except (tarfile.ReadError, EOFError, OSError) as err:
        # Nothing arrived (wrong password, no connection, or nothing to send), or the stream was cut.
        print(f"\nThe stream from Sherlock ended early or was empty ({err}). What arrived is kept; run the same "
              f"command again to get the rest.", file=sys.stderr)
    print(f"\r  {count:,} files, {total/1e6:,.0f} MB received in {time.time() - started:.0f} s", file=sys.stderr)
    if odd:print(f"  {len(odd)} entries in the stream were not asked for and were not written.", file=sys.stderr)
    cmd_check()                    # with the status that may just have arrived
    if count:print("Next: python3 scripts/build_sweep_plan.py && python3 scripts/build_paper_figures.py   (the Sweep page; the figures and the Papers page)")


def cmd_check(sw=None):
    sw = sw or sweep_data.Sweep();w = sw.wanted()
    have = [(p, s) for p, s, _ in w if here(sw.results, p, s)]
    absent = sum(1 for p, s, a in w if a);todo = missing(sw)
    size = sum((sw.results/p).stat().st_size for p, _ in have)
    print(f"sweep_results/: {len(have):,} of {len(w):,} wanted files are here ({size/1e9:.2f} GB).")
    print(f"  {absent:,} cannot exist (the run stopped before that output, as the status export of {sw.status['time']} shows).")
    more = extras()
    if more:print(f"  Beyond that list: {sum(1 for p in more if here(sw.results, p, None)):,} of {len(more):,} further files named from the inventory are here.")
    if not todo:print("  Nothing is missing.");return
    binary = [p for p, s in todo if s]
    runs = sorted({p.split("/")[0] for p, _ in todo})
    open_runs = [r for r in runs if not (sw.results/r).is_dir()]
    print(f"  {len(todo):,} are not here ({len(binary)} binary frames), in {len(runs)} runs; {len(open_runs)} of those runs have "
          f"nothing here at all (not finished on Sherlock, or not fetched yet).")
    for r in runs[:12]:print(f"    {r}")
    if len(runs) > 12:print(f"    ... and {len(runs) - 12} more")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":cmd_list()
    elif command == "remote":cmd_remote()
    elif command == "receive":cmd_receive()
    elif command == "check":cmd_check()
    else:sys.exit(__doc__)
