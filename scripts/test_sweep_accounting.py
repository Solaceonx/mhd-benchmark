"""Accounting must not double-count job steps, copied rows or requeue attempts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import sweep_accounting as accounting


class AccountingTest(unittest.TestCase):
    def test_allocation_scopes_and_requeues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'query.json').write_text(json.dumps(dict(since='2026-09-01',fetched_at='2026-10-14T00:00:00Z')))
            ledger = [dict(job='123',key='cpu-unit',options=['--ntasks=128']),
                      dict(job='456',key='gpu-unit',options=['--gpus=1'])]
            (root/'ledger.jsonl').write_text('\n'.join(json.dumps(row) for row in ledger))
            fields = ['JobIDRaw','JobID','JobName','Account','Partition','State','AllocCPUS','ElapsedRaw','CPUTimeRAW','Submit','Start','End','AllocTRES']
            def row(raw,job,cpus,seconds,start='2026-10-01T00:00:00',tres='cpu=128'):
                return [raw,job,'experiment','tabel','normal','COMPLETED',cpus,seconds,cpus*seconds,
                        '2026-09-30T00:00:00',start,'2026-10-01T05:00:00',tres]
            rows = [row('123','123_0',128,5*3600),row('123','123_0',128,5*3600),
                    row('123','123_0.batch',128,5*3600),
                    row('123','123_0',128,3600,start='2026-10-02T00:00:00'),
                    row('456','456',4,2*3600,tres='cpu=4,gres/gpu=1,gres/gpu:h100=1'),
                    row('789','789',2,3600)]
            with (root/'accounting.psv').open('w') as stream:
                writer=csv.writer(stream,delimiter='|');writer.writerow(fields);writer.writerows(rows)
            result=accounting.summarize(root)
            self.assertEqual(len(result['allocations']),4)
            self.assertEqual(result['scopes']['sweep_cpu_lane']['core_hours'],768)
            self.assertEqual(result['scopes']['sweep_gpu_lane']['gpu_hours'],2)
            self.assertEqual(result['scopes']['all_user_jobs']['core_hours'],778)
            self.assertEqual(result['scopes']['outside_sweep_ledger']['core_hours'],2)
            self.assertEqual(result['missing_ledger_jobs'],[])

    def test_typed_gpu_count_does_not_double_count(self):
        self.assertEqual(accounting.gpu_count('cpu=4,gres/gpu=2,gres/gpu:h100=2'),2)
        self.assertEqual(accounting.gpu_count('cpu=4,gres/gpu:h100=2'),2)

    def test_error_on_stdout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'accounting.psv'
            path.write_text('|'.join(sorted(accounting.FIELDS))+'\nsacct: error: Too wide of a date range in query\n')
            with self.assertRaisesRegex(ValueError, 'Too wide'):
                list(accounting.allocation_rows(path))

    def test_header_only_is_a_valid_empty_query(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'accounting.psv'
            path.write_text('|'.join(sorted(accounting.FIELDS))+'\n')
            self.assertEqual(list(accounting.allocation_rows(path)), [])


if __name__=='__main__':unittest.main()
