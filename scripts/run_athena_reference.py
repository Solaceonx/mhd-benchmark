"""Run CPU Athena++ in restart segments to land exactly on comparison times."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=512)
    parser.add_argument('--times',type=float,nargs='+',default=[.1,.25,.5])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    n=args.n;block=min(128,n//2)
    if n<16 or n%block or not all(t>0 for t in args.times) or args.times!=sorted(set(args.times)):
        parser.error('Use N>=16 divisible by min(128,N/2), and increasing positive times')
    out=(args.out or ROOT/'scripts'/'out'/f'athena_{n}').resolve()
    if out.exists() and any(out.iterdir()):parser.error('Output directory must be empty or new')
    out.mkdir(parents=True,exist_ok=True)
    athena=ROOT/'external'/'athena'
    executable=athena/'bin'/'athena'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=athena,text=True).strip()
    config=subprocess.check_output([str(executable),'-c'],text=True)
    (out/'build_configuration.txt').write_text(config)
    env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['OMP_DYNAMIC']='FALSE'
    runs=[];restart=None;started=time.monotonic()
    for endpoint in args.times:
        stage=out/f't{endpoint:.3f}';stage.mkdir()
        if restart is None:
            cmd=[str(executable),'-i',str(ROOT/'scripts'/'athena'/'athinput.reference'),
                 f'mesh/nx1={n}',f'mesh/nx2={n}',f'meshblock/nx1={block}',f'meshblock/nx2={block}']
        else:
            cmd=[str(executable),'-r',str(restart)]
        cmd.append(f'time/tlim={endpoint:.17g}')
        print(f'Athena++ {n}²: advancing to t={endpoint}; log {stage / "run.log"}',flush=True)
        start=time.monotonic()
        with (stage/'run.log').open('w') as log:
            subprocess.run(cmd,cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        restart=stage/'OrszagTang.final.rst'
        if not restart.exists():raise RuntimeError('Missing final restart')
        runs.append({'time':endpoint,'seconds':time.monotonic()-start,'directory':str(stage.relative_to(out)),'command':cmd})
        record={'N':n,'meshblock':block,'threads':4,'commit':commit,'configuration':config,
            'input':str(ROOT/'scripts'/'athena'/'athinput.reference'),'segments':runs,
            'wall_seconds':time.monotonic()-started,'coordinate_shift':[.5,.5],
            'precision':'double evolution; stock VTK outputs float32','completed':len(runs)==len(args.times)}
        (out/'run_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'  finished in {runs[-1]["seconds"]:.1f}s',flush=True)
    print('Reference ready:',out,flush=True)


if __name__=='__main__':main()
