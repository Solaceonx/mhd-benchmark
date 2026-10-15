"""Check the compact density reader against the existing full-volume reader."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
import compare_movies
import ramses_movie


def fixture(directory, nvar=5):
    n=4; ranks=2; level=2; suffix='00001'
    (directory/f'info_{suffix}.txt').write_text('ncpu = 2\nndim = 3\ntime = 0.25\nnstep_coarse = 12\n')
    names=['density','momentum_x','momentum_y','momentum_z','total_energy']
    (directory/'hydro_file_descriptor.txt').write_text('\n'.join(f'{i+1}, {name}' for i,name in enumerate(names)))
    centers=np.array([(x,y,z) for z in (.25,.75) for y in (.25,.75) for x in (.25,.75)])
    counts=np.array([[0,0],[4,4]],dtype='<i4')
    def rec(file, values, dtype='<i4'):
        raw=np.asarray(values,dtype=dtype).tobytes();size=struct.pack('<i',len(raw));file.write(size+raw+size)
    for rank in range(1,ranks+1):
        with (directory/f'amr_{suffix}.out{rank:05d}').open('wb') as amr, (directory/f'hydro_{suffix}.out{rank:05d}').open('wb') as hydro:
            for value in (ranks,3,[n,n,n],level,100,0,0):rec(amr,value)
            rec(amr,1.,'<f8')
            for _ in range(11+2):rec(amr,0)
            rec(amr,counts);rec(amr,0)
            for _ in range(6):rec(amr,0)
            for value in (ranks,nvar,3,level,0):rec(hydro,value)
            rec(hydro,1.4,'<f8')
            for ilevel in range(level):
                for owner in range(ranks):
                    grids=counts[ilevel,owner]
                    rec(hydro,ilevel+1);rec(hydro,grids)
                    if not grids:continue
                    for _ in range(3):rec(amr,np.zeros(grids,dtype=int))
                    coordinates=centers[owner*4:owner*4+4]
                    for axis in range(3):rec(amr,coordinates[:,axis],'<f8')
                    for _ in range(1+6+24):rec(amr,np.zeros(grids,dtype=int))
                    for child in range(8):
                        offset=np.array([(child>>axis)&1 for axis in range(3)])
                        ijk=np.floor((coordinates+(offset-.5)/n)*n).astype(int)
                        rho=1+ijk[:,0]+10*ijk[:,1]+100*ijk[:,2]
                        rho=np.where((ijk[:,0]==0)&(ijk[:,1]==0)&(ijk[:,2]==n//2),-.5,rho)
                        rec(hydro,rho,'<f8')
                        for variable in range(1,nvar):rec(hydro,np.full(grids,10. if variable==4 else 0.),'<f8')


class MovieTest(unittest.TestCase):
    def test_midplane_matches_full_reader_including_negative_density(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'output_00001';output.mkdir();fixture(output)
            _,_,full=ramses_movie.R.read_ramses_hydro(output,4)
            actual=ramses_movie.density_slice(output,4)
            np.testing.assert_array_equal(actual,full[:,:,2,0].astype(np.float32))
            self.assertEqual(actual[0,0],-.5)
    def test_wrong_solver_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'output_00001';output.mkdir();fixture(output,nvar=11)
            with self.assertRaisesRegex(ValueError,'five conservative'):ramses_movie.density_slice(output,4)
    def test_plan_never_submits_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'forcing.toml').write_text('[mach_30]\na_rms=1450\n[mach_50]\na_rms=4000\n')
            with patch('subprocess.check_output',side_effect=AssertionError('No Slurm calls in plan mode')):
                with contextlib.redirect_stdout(io.StringIO()) as capture:
                    compare_movies.main(['--pilot',str(root),'--results',str(root/'movies')])
            self.assertIn('4 missing RAMSES runs',capture.getvalue())
            self.assertFalse((root/'movies').exists())
    def test_gamma_groups_cannot_be_mixed(self):
        with self.assertRaises(ValueError):compare_movies.configuration([3,30])
    def test_submission_requests_one_gpu_and_serial_64_core_array(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=root/'bundle';bundle.mkdir()
            for name in ('config_movie.toml','config_movie_highmach.toml'):
                shutil.copyfile(compare_movies.HERE/name,bundle/name)
            pilot=root/'pilot';pilot.mkdir();(pilot/'forcing.toml').write_text('[mach_30]\na_rms=1450\n[mach_50]\na_rms=4000\n')
            executable=root/'codes/ramses/bin-turb-fixed/ramses_turb3d'
            executable.parent.mkdir(parents=True);executable.write_text('#!/bin/sh\n');executable.chmod(0o700)
            calls=[]
            def command(argv,**kwargs):
                calls.append(argv);return '' if argv[0]=='squeue' else '123\n'
            with patch.object(compare_movies,'HERE',bundle),patch.dict(os.environ,{'MHD_CODES':str(root/'codes')}):
                with patch('subprocess.check_output',side_effect=command),contextlib.redirect_stdout(io.StringIO()):
                    compare_movies.main(['--submit','--pilot',str(pilot),'--results',str(root/'results')])
            submissions=[argv for argv in calls if argv[0]=='sbatch']
            self.assertEqual(len(submissions),2)
            self.assertTrue(any('--array=0-3%1' in argv for argv in submissions))
            self.assertEqual(len(json.loads((bundle/'compare_cpu_tasks.json').read_text())),4)
            self.assertEqual(tomllib.loads((bundle/'config_compare_high.toml').read_text())['gpu']['methods'],['glm'])


if __name__=='__main__':unittest.main()
