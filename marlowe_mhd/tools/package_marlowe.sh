#!/bin/bash
# Create the upload archive beside this directory on the Mac.
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent="$(dirname "$bundle")"
name="$(basename "$bundle")"
archive="$parent/${name}.tar.gz"

find "$bundle" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$bundle" -type f \( -name '.DS_Store' -o -name '._*' \) -delete
# No macOS extended attributes or AppleDouble files: GNU tar on Marlowe warns about them.
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -C "$parent" -czf "$archive" \
    --exclude="$name/logs/*.out" --exclude="$name/logs/*.err" \
    --exclude="$name/returns/*.tar.gz" --exclude="$name/campaigns/*.tsv" \
    "$name"
(cd "$parent" && shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256")
printf '%s\n%s\n' "$archive" "$archive.sha256"

