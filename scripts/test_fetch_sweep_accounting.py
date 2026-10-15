"""Exercise transfer completion/failure locally; SSH is replaced with a fixture."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


class FetchAccountingTests(unittest.TestCase):
    def remote_query(self, root, failure=False):
        """Run the real remote shell body against local Slurm stand-ins."""
        source = Path(__file__).with_name("fetch_sweep_accounting.sh").read_text()
        body = source.split("<<'REMOTE' || fetch_status=$?\n", 1)[1].split('\nREMOTE\n', 1)[0]
        body = body.replace('cd /scratch/users/lyxia/mhd-study/sherlock_sweep', 'cd "$TEST_PROJECT"')
        ledger = root/"sweep_state"; ledger.mkdir()
        (ledger/"ledger.jsonl").write_text('{}\n')
        preamble = '''
module() { :; }
python3() {
    if [ "$1" = tools/sweep.py ]; then
        printf '{"time":"fresh"}' > "$4"
    else
        "$TEST_PYTHON" "$@"
    fi
}
sacct() {
    if [ "$TEST_FAILURE" = yes ]; then
        printf 'fixture accounting refusal on stdout\n'
        exit 23
    fi
    "$TEST_PYTHON" - "$@" <<'PY'
from datetime import datetime, timedelta
import os, sys
print('JobIDRaw|JobID|JobName|Account|Partition|State|AllocCPUS|ElapsedRaw|CPUTimeRAW|Submit|Start|End|AllocTRES')
if os.environ['TEST_FAILURE'] == 'invalid':
    print('sacct: error: fixture server error')
elif os.environ['TEST_FAILURE'] == 'range' and (datetime.fromisoformat(sys.argv[sys.argv.index('-E')+1])-datetime.fromisoformat(sys.argv[sys.argv.index('-S')+1])) > timedelta(days=3):
    print('sacct: error: Too wide of a date range in query')
else:
    print('123|123|experiment|tabel|normal|COMPLETED|64|3600|230400|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T01:00:00|cpu=64')
PY
}
sshare() { printf 'fixture fairshare\n'; }
scontrol() { printf 'PriorityDecayHalfLife = 7-00:00:00\n'; }
'''
        env = dict(os.environ, TEST_PROJECT=str(root), TEST_PYTHON=sys.executable,
                   TEST_FAILURE=failure if isinstance(failure, str) else "yes" if failure else "no")
        result = subprocess.run(["bash"], input=(preamble+body).encode(), env=env, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            files = {member.name.removeprefix('./'): tar.extractfile(member).read().decode()
                     for member in tar.getmembers()
                     if member.isfile() and not Path(member.name).name.startswith('._')}
        return result, files

    def test_remote_shell_wrapper_exit_preserves_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, files = self.remote_query(Path(tmp), failure=True)
            self.assertEqual(result.returncode, 23)
            self.assertEqual(json.loads(files["export_latest.json"])["time"], "fresh")
            self.assertNotIn("query.json", files)
            self.assertIn(b"fixture accounting refusal on stdout", result.stderr)
            self.assertNotIn(b"file is the archive", result.stderr)

    def test_remote_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, files = self.remote_query(Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            for name in ("export_latest.json", "accounting.psv", "ledger.jsonl", "fairshare.txt",
                         "priority_config.txt", "query.json"):
                self.assertIn(name, files)
            self.assertIn('PriorityDecayHalfLife', files['priority_config.txt'])
            self.assertEqual(json.loads(files['query.json'])['since'], '2026-09-01')

    def test_remote_retries_rejected_date_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, files = self.remote_query(Path(tmp), failure='range')
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn(b'splitting this window', result.stderr)
            self.assertNotIn('sacct:', files['accounting.psv'])
            self.assertEqual(files['accounting.psv'].count('JobIDRaw|'), 1)
            self.assertIn('query.json', files)

    def test_remote_error_with_success_exit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, files = self.remote_query(Path(tmp), failure='invalid')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'fixture server error', result.stderr)
            self.assertNotIn('query.json', files)
            self.assertIn('export_latest.json', files)

    def fetch(self, root, files, status=0):
        scripts = root/"scripts"; scripts.mkdir()
        shutil.copy(Path(__file__).with_name("fetch_sweep_accounting.sh"), scripts)
        archive = root/"fixture.tar"
        with tarfile.open(archive, "w") as tar:
            for name, text in files.items():
                data = text.encode(); info = tarfile.TarInfo(name); info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        commands = root/"bin"; commands.mkdir()
        ssh = commands/"ssh"
        ssh.write_text('#!/bin/bash\ncat "$FETCH_ARCHIVE"\nexit "$FETCH_RC"\n')
        ssh.chmod(0o755)
        env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
                   FETCH_ARCHIVE=str(archive), FETCH_RC=str(status))
        return subprocess.run(["bash", str(scripts/"fetch_sweep_accounting.sh")],
                              env=env, text=True, capture_output=True)

    def test_complete_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {name: "fixture\n" for name in
                     ("accounting.psv", "ledger.jsonl", "fairshare.txt", "priority_config.txt")}
            files.update({"export_latest.json": '{"time":"fresh"}', "query.json": '{}'})
            result = self.fetch(root, files)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root/"progress/sweep_status/export_latest.json").read_text())["time"], "fresh")
            for name in files:
                if name != "export_latest.json":
                    self.assertEqual((root/"progress/sweep_status/accounting"/name).read_text(), files[name])
            self.assertIn("Fetched current sweep status", result.stdout)

    def test_failed_query_keeps_export_and_previous_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/"progress/sweep_status/accounting"; old.mkdir(parents=True)
            (old/"accounting.psv").write_text("previous report")
            result = self.fetch(root, {"export_latest.json": '{"time":"fresh"}',
                                       "accounting.psv": "partial report"}, status=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((old/"accounting.psv").read_text(), "previous report")
            self.assertEqual(json.loads((old.parent/"export_latest.json").read_text())["time"], "fresh")
            self.assertIn("accounting is incomplete", result.stderr)

    def test_incomplete_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.fetch(root, {"export_latest.json": '{}', "query.json": '{}'})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/"progress/sweep_status/accounting").exists())
            self.assertIn("missing accounting.psv", result.stderr)


if __name__ == "__main__":
    unittest.main()
