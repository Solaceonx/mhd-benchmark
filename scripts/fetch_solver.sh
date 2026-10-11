#!/bin/bash
# Tom Abel's solver kernels are not kept in this repository (his repository carries no licence). This puts them where
# everything here expects them: GLMMHDTurb/ (a clone at the commit every run used) and <bundle>/vendor/, and checks
# each copy against the checksums in that bundle's vendor/PROVENANCE.json.
#   bash scripts/fetch_solver.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";cd "$ROOT"
REPO="${GLMMHDTURB_REPO:-https://github.com/yipihey/GLMMHDTurb.git}"
COMMIT=15ce37d47782afc9b4304b5cccaf847a4f7a4ae0
[[ -d GLMMHDTurb/.git ]] || git clone -q "$REPO" GLMMHDTurb
[[ "$(cd GLMMHDTurb && git rev-parse HEAD)" == "$COMMIT" ]] || (cd GLMMHDTurb && git checkout -q "$COMMIT")
for record in */vendor/PROVENANCE.json; do
    dir="$(dirname "$record")"
    python3 - "$record" "$dir" GLMMHDTurb <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
record, dest, source = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for name, want in json.loads(record.read_text())["files"].items():
    data = (source/name).read_bytes()
    if hashlib.sha256(data).hexdigest() != want:
        sys.exit(f"{source/name} is not the file {record} records")
    shutil.copyfile(source/name, dest/name)
print(f"{dest}: kernels in place")
PY
done
