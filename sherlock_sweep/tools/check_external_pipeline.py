"""CPU-only regression check for external result conversion and failure handling."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('external_suite',ROOT/'codes/external.py')
external=importlib.util.module_from_spec(spec)
spec.loader.exec_module(external)


class ExternalPipelineChecks(unittest.TestCase):
    def test_athenapp_block_tracks_mpi_ranks(self):
        self.assertEqual(external.mpi_ranks('srun -n 32'),32)
        self.assertEqual(external.mpi_ranks('srun --ntasks=16'),16)
        self.assertEqual(external.athenapp_block(64,2,32),8)
        self.assertEqual(external.athenapp_block(32,2,16),8)
        with self.assertRaisesRegex(ValueError,'fewer 8-cell blocks'):
            external.athenapp_block(32,2,32)

    def run_case(self,rc):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exe=root/'dummy_executable';exe.write_text('synthetic executable')
            config=root/'config.toml'
            config.write_text('name="uniform"\nt_end=0.1\nframes=3\nrepeats=1\nsnapshot_frames=[0,2]\n')
            out=root/'result';native=out/'athenak_matched_r1'/'bin'
            def fake_spec(_config,n,prefix,seed,state=False):
                Path(str(prefix)+'.spec').write_text('synthetic spec')
            def fake_runner(args,config,spec,settings,rundir,rep,outputs):
                native.mkdir(parents=True);(native/'original.bin').write_bytes(b'original')
                def frames():
                    for i,t in enumerate((0.,.05,.1)):
                        u=np.ones((8,8,1,9));u[...,1:]=0
                        yield t,i,u
                return rc,'synthetic log',{'wall_seconds':1.,'steps':2},frames()
            def fake_postprocess(*args,**kwargs):
                self.assertTrue((native/'original.bin').exists(), 'native output removed before postprocessing')
                return None
            argv=['external.py','athenak','matched',str(config),'8',str(out),
                  '--exe',str(exe),'--geometry','plane']
            with (patch.object(sys,'argv',argv),patch.object(external,'write_spec',fake_spec),
                  patch.dict(external.RUNNERS,{'athenak':fake_runner}),
                  patch.object(external.subprocess,'run',fake_postprocess)):
                if rc:
                    with self.assertRaisesRegex(RuntimeError,'External solver failed'):external.main()
                else:external.main()
            meta=tomllib.loads((out/'athenak_matched_r1'/'run.toml').read_text())
            self.assertEqual(meta['status'],'failed' if rc else 'complete')
            self.assertEqual([f.get('file') for f in meta['frames']],['state_000.bin',None,'state_002.bin'])
            self.assertEqual(native.exists(),bool(rc), 'failed native output must remain for diagnosis')

    def test_success_cleanup_after_conversion(self):self.run_case(0)
    def test_solver_failure_propagates_and_keeps_native_output(self):self.run_case(1)


if __name__=='__main__':unittest.main()
