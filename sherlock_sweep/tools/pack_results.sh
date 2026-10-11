#!/bin/bash
# Pack one run directory for download, in two archives:
#   returns/NAME.summary.tar.gz  manifests, CSVs (diagnostics, spectra), logs: small, fetch these first
#   returns/NAME.tar.gz          everything else except files over MHD_PACK_MAX_FILE_MB (default 50)
# Larger files (3D states from 128^3 up, native outputs, checkpoints) stay on scratch and are listed in
# NAME/kept_on_scratch.txt. A resumed or resubmitted run replaces its earlier archives.
# Packing never fails the job that calls it. If it cannot finish (files changing or vanishing under it), the run's
# archives are removed, so an archive is either current or absent: `tools/sweep.py audit` lists unpacked runs and
# `tools/sweep.py repack` packs them. A run in use by another running job (NAME/data.lock) is left to that job.
set -uo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { printf 'Usage: pack_results.sh RESULT_DIRECTORY\n' >&2; exit 2; }
[[ -d "$1" ]] || { printf 'Not packed: %s is not a directory\n' "$1" >&2; exit 0; }
source_dir="$(cd "$1" && pwd)"
name="$(basename "$source_dir")"; parent="$(dirname "$source_dir")"
source "$bundle/jobs/common.sh"
if run_in_use "$source_dir"; then printf 'Not packed: %s is in use by another running job\n' "$name"; exit 0; fi
(
    set -e
    mkdir -p "$bundle/returns"
    cap_mb="${MHD_PACK_MAX_FILE_MB:-50}"
    exclude="$(mktemp)"; summary="$(mktemp)"; trap 'rm -f "$exclude" "$summary"' EXIT
    (cd "$parent" && find "$name" -type f -size +"${cap_mb}M" | sort) > "$source_dir/kept_on_scratch.txt"
    { cat "$source_dir/kept_on_scratch.txt"; printf '%s\n' "$name/*/checkpoint" "$name/*/checkpoint.tmp" "$name/*/*/checkpoint" "$name/*/*/checkpoint.tmp" "$name/data.lock*"; } > "$exclude"
    (cd "$parent" && find "$name" -type f \( -name '*.toml' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' -o -name '*.out' \) \
        -size -"${cap_mb}M" | sort) > "$summary"
    checksum() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
    for kind in summary full; do
        archive="$bundle/returns/$name.tar.gz"; [[ "$kind" == summary ]] && archive="$bundle/returns/$name.summary.tar.gz"
        if [[ "$kind" == summary ]]; then tar -czf "$archive.tmp" -C "$parent" -T "$summary"
        else tar -czf "$archive.tmp" --exclude-from="$exclude" -C "$parent" "$name"; fi
        mv -f "$archive.tmp" "$archive"
        (cd "$bundle/returns" && checksum "$(basename "$archive")" > "$(basename "$archive").sha256")
    done
    printf 'Download: %s (summary: %s.summary.tar.gz); %s large files kept on scratch\n' \
        "$bundle/returns/$name.tar.gz" "$name" "$(wc -l < "$source_dir/kept_on_scratch.txt" | tr -d ' ')"
)
rc=$?
if (( rc != 0 )); then
    for suffix in .tar.gz .tar.gz.sha256 .tar.gz.tmp .summary.tar.gz .summary.tar.gz.sha256 .summary.tar.gz.tmp; do
        rm -f "$bundle/returns/$name$suffix"
    done
    printf 'WARNING: packing %s did not finish (status %s); its archives were removed. Pack it again with: python3 tools/sweep.py repack %s\n' \
        "$name" "$rc" "$name" >&2
fi
exit 0
