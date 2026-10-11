"""Assemble the RAMSES patch directory for the comparison suite.

RAMSES patches replace whole source files. Upstream mhd/condinit.f90 also defines helper
routines (velana, prep_collapse, ...) that other files call, so the patched condinit.f90 is
our condinit.f90 followed by every upstream routine except condinit itself.

usage: python3 prepare_patch.py RAMSES_DIR OUT_PATCH_DIR
"""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


def main(ramses, out):
    upstream = (Path(ramses)/"mhd/condinit.f90").read_text()
    match = re.search(r"^subroutine\s+condinit\(.*?^end subroutine condinit\s*$", upstream, re.M | re.S)
    if not match:sys.exit("Could not find condinit in upstream mhd/condinit.f90")
    rest = upstream[:match.start()] + upstream[match.end():]
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    (out/"condinit.f90").write_text((HERE/"condinit.f90").read_text() +
        "\n! ---- Upstream helper routines from mhd/condinit.f90 (unchanged) ----\n" + rest)
    print("RAMSES patch written to", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
