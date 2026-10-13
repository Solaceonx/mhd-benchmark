"""Numerical and provenance checks for the shared diagnostic builder.

    .venv-movie/bin/python scripts/check_test_diagnostics.py
"""
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
import test_diagnostics as td


def run(meta=None,diag=None,path=None):
    return td.Run("fixture","glm",4,42,path or Path("."),meta or dict(status="complete",t_end_requested=1),
                  diag or dict(time=np.array([0.,1.]),energy=np.array([1.,1.]),mass=np.ones(2)),{})


class Checks(unittest.TestCase):
    def test_vector_direction_is_an_error(self):
        ref=np.zeros((4,4,9));ref[...,0]=1;ref[...,5]=.3;ref[...,6]=.4
        actual=ref.copy();actual[...,5:8]*=-1
        errors=td.compute_errors(actual,ref)
        for norm in td.NORMS:self.assertAlmostEqual(errors['magnetic_relative_'+norm],2)
        self.assertEqual(errors['density_absolute_L1'],0)

    def test_full_volume_norms_and_local_outlier(self):
        ref=np.zeros((2,2,2,9));ref[...,0]=1;actual=ref.copy();actual[0,0,0,0]+=2
        errors=td.compute_errors(actual,ref)
        self.assertAlmostEqual(errors['density_relative_L1'],.25)
        self.assertAlmostEqual(errors['density_relative_L2'],np.sqrt(.5))
        self.assertAlmostEqual(errors['density_relative_Linf'],2)

    def test_zero_field_is_undefined_relative_not_zero_error(self):
        ref=np.zeros((2,2,9));ref[...,0]=1;actual=ref.copy();actual[...,5]=.2
        errors=td.compute_errors(actual,ref)
        self.assertIsNone(errors['magnetic_relative_L1'])
        self.assertAlmostEqual(errors['magnetic_absolute_L1'],.2)
        self.assertEqual(td.metric('blast_2d_b0','magnetic'),('absolute','[code units]',1))

    def test_nonfinite_or_unfinished_is_not_complete(self):
        self.assertTrue(run().completed)
        self.assertFalse(run(diag=dict(time=np.array([0.,.5]),energy=np.ones(2))).completed)
        self.assertFalse(run(diag=dict(time=np.array([0.,1.]),energy=np.array([1.,np.nan]))).completed)

    def test_slice_cannot_be_used_as_full_volume_error(self):
        r=run(meta=dict(status='complete',t_end_requested=1,planar=False,frames=[dict(time=1,file='slice.bin')]))
        data=td.Data.__new__(td.Data);data.runs={('fixture','glm',4,42):r}
        state=np.ones((4,4,9))
        data.reference=lambda *args:(state,'reference')
        data.load_array=lambda *args:state
        data.errors.cache_clear()
        self.assertEqual(data.errors('fixture','glm',4),[])

    def test_timings_exclude_nan_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for rep,sec,nan in [(1,10,False),(2,1,True),(3,20,False)]:
                p=root/f'glm_r{rep}';p.mkdir()
                (p/'run.toml').write_text(f'status = "complete"\nsolve_seconds = {sec}.0\n')
                (p/'diagnostics.csv').write_text('time,mass,energy\n0,1,1\n1,1,'+('nan' if nan else '1')+'\n')
            r=run(path=root/'glm_r1')
            data=types.SimpleNamespace(sw=types.SimpleNamespace(configs={'fixture':dict(repeats=3)}))
            sec,_,count=td.timing(data,r)
            self.assertEqual(sec,15);self.assertEqual(count,2)

    def test_planar_spectral_parseval_on_nonzero_modes(self):
        n=32;x=np.arange(n)/n
        a=np.zeros((n,n,9));a[...,0]=1;a[...,1]=np.sin(2*np.pi*x)[:,None]
        spec=td.planar_spectrum(a)
        self.assertAlmostEqual(spec['v'].sum(),.25)
        self.assertAlmostEqual(spec['v'][0],.25)


if __name__=='__main__':unittest.main()
