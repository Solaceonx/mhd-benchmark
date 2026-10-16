# For an assistant picking this project up

Read `HANDOFF.md` first: it holds the rules that must hold, what is where, what is in flight and what comes next.
Then `README.md` and `docs/`.

The short version of the rules:

- Only lyxia can reach Sherlock. Hand over exact commands, one per code block; they paste the output back.
- Never commit Tom Abel's solver source (`scripts/fetch_solver.sh` restores it). Edit it only on the `sweep-fixes`
  branch of `GLMMHDTurb/`, which he asked for (`HANDOFF.md`, section 4).
- The website pages under `mhd/` stay unlisted; do not write their address into this repository.
- Never read lyxia's notes, and never record their password or a token.
- Publish the site only from lyxia's Mac (`HANDOFF.md`, section 3), and check the publisher's exit code.
- Keep replies short. Compute numbers from the records instead of quoting them from memory.
