"""Prepare a Phantom checkout for the comparison suite (idempotent): copies setup_abel.f90 into src/setup/ and
analysis_abel.f90 into src/utils/. Nothing of Phantom itself changes; build.sh builds it with SETUP=mhdrotor (MHD,
periodic) and SETUPFILE=setup_abel.f90, ANALYSIS=analysis_abel.f90.

    python3 patch_phantom.py PHANTOM_DIR
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    if len(sys.argv) != 2:sys.exit(__doc__)
    root = Path(sys.argv[1])
    shutil.copy(HERE/"setup_abel.f90", root/"src/setup/setup_abel.f90")
    shutil.copy(HERE/"analysis_abel.f90", root/"src/utils/analysis_abel.f90")
    print(f"patched {root}")
