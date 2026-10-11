#!/bin/bash
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { printf 'Usage: pack_results.sh RESULT_DIRECTORY\n' >&2; exit 2; }
source_dir="$(cd "$1" && pwd)"
mkdir -p "$bundle/returns"
name="$(basename "$source_dir")"
archive="$bundle/returns/$name.tar.gz"
[[ ! -e "$archive" ]] || { printf 'Archive already exists: %s\n' "$archive" >&2; exit 2; }
tar -czf "$archive.tmp" -C "$(dirname "$source_dir")" "$name"
mv "$archive.tmp" "$archive"
if command -v sha256sum >/dev/null; then
    (cd "$bundle/returns" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
else
    (cd "$bundle/returns" && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")
fi
printf 'Download: %s\n' "$archive"
