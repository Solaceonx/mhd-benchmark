"""CPU-only checks of numerical analysis and bundled reference integrity."""
import hashlib
import json
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import analyze_returns as a


class AnalysisChecks(unittest.TestCase):
    def test_vector_norms(self):
        ref=np.ones((4,5,3));value=ref+np.array([3.,4.,0.])
        norms=a.error_norms(value,ref)
        for norm in ('L1','L2','Linf'):
            self.assertAlmostEqual(norms['absolute_'+norm],5)
            self.assertAlmostEqual(norms['relative_'+norm],5/np.sqrt(3))
        self.assertIsNone(a.error_norms(value,np.zeros_like(value))['relative_L1'])

    def test_spectral_modes(self):
        n=16;x=np.arange(n)/n
        wave=np.broadcast_to(np.sin(2*np.pi*3*x),(n,n))
        power=a.spectrum(wave)
        self.assertAlmostEqual(power[3],.5)
        self.assertAlmostEqual(power.sum(),.5)
        vector=np.stack([np.ones((n,n,n)),np.zeros((n,n,n)),np.ones((n,n,n))*2],axis=-1)
        power=a.spectrum(vector,True)
        self.assertAlmostEqual(power[0],5)
        self.assertAlmostEqual(power[1:].sum(),0)

    def test_axis_order_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.bin';shape=(3,4,5,9)
            u=np.arange(np.prod(shape),dtype='<f4').reshape(shape)
            path.write_bytes(u.tobytes(order='F'))
            read=a.load_state(path,{'saved_shape':shape})
            np.testing.assert_array_equal(read,u.transpose(2,1,0,3))
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaises(ValueError):a.load_state(path,{'saved_shape':shape})

    def test_reference_availability(self):
        meta=dict(N=32,test='orszag_tang',gamma=5/3)
        self.assertIsNone(a.reference(meta,.125,None,{},a.BUNDLE/'references')[0])
        self.assertIsNotNone(a.reference(meta,.5,None,{},a.BUNDLE/'references')[0])
        meta['test']='rotor'
        self.assertIsNone(a.reference(meta,.1,None,{},a.BUNDLE/'references')[0])

    def test_driver_reference_file(self):
        with tempfile.TemporaryDirectory() as temp:
            shape=(4,4,9);u=np.arange(np.prod(shape),dtype='<f4').reshape(shape)
            (Path(temp)/'ref_003.bin').write_bytes(u.tobytes(order='F'))
            meta=dict(N=4,test='field_loop_novz',problem='field_loop',saved_shape=shape)
            ref,label=a.reference(meta,1.,Path(temp),{'reference_file':'ref_003.bin','reference':'advected'},a.BUNDLE/'references')
            np.testing.assert_array_equal(ref,u.transpose(1,0,2));self.assertEqual(label,'advected')

    def test_reference_checksums(self):
        root=a.BUNDLE/'references'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(len(manifest['sha256']),12)
        for name,expected in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)

    def test_failed_runs_excluded_from_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'raw';group=root/'fixture';group.mkdir(parents=True)
            (group/'environment.toml').write_text('device="CPU SYNTHETIC TEST"\nsmoke=true\n')
            for repeat,status,seconds in ((1,'complete',2),(2,'failed',999),(3,'running',999)):
                run=group/f'glm_r{repeat}';run.mkdir()
                (run/'run.toml').write_text(f'''test="uniform"
N=8
seed=42
method="glm"
repeat={repeat}
status="{status}"
snapshots_written=false
frames=[]
stepping_seconds={seconds}
solve_seconds={seconds}
''')
            out=Path(temp)/'analysis'
            result=a.analyze(root,out,a.BUNDLE/'references')
            self.assertEqual(len(result['failed_or_incomplete']),2)
            with (out/'timing_summary.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['completed_repeats'],'1')
            self.assertEqual(float(rows[0]['solve_seconds_median']),2)


if __name__=='__main__':
    unittest.main()
