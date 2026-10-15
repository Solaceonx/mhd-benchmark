"""Exercise transfer completion/failure locally; SSH is replaced with a fixture."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


class FetchAccountingTests(unittest.TestCase):
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
