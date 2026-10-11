"""Submit the full comparison behind GPU, build and small-pilot success gates.

Run through tools/submit_campaign.sh so the selected site's modules are loaded.
The command submits jobs; it does not run simulations on the login node.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

BUNDLE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mhd_plan',BUNDLE/'codes/plan.py')
plan=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def matrix():
    """Return (human label, submit.sh args, needs_gpu) for the published matrix."""
    result=[]
    for path in sorted(p for p in (BUNDLE/'configs').glob('*.toml') if not p.name.startswith('.')):
        config=plan.load(path);name=config['name']
        result.append((f'{name}:gpu',[name],True))
        for code in ('athenak','ramses'):
            if plan.tasks(config,code,'run'):
                for variant in ('matched','best'):
                    result.append((f'{name}:{code}:{variant}',[name,'--code',f'{code}:{variant}'],code=='athenak'))
        if plan.tasks(config,'athenak','reference'):
            result.append((f'{name}:athenak:reference',[name,'--reference'],True))
        if plan.tasks(config,'athenapp','reference'):
            result.append((f'{name}:athenapp:reference',[name,'--reference-check'],False))
    return result


def active_dependencies(job_ids):
    """Keep unfinished Slurm dependencies; omit gates already completed.

    Completed jobs eventually age out of slurmctld. Passing those old IDs back
    to sbatch can produce "Job dependency problem", even though sacct retains
    their successful accounting records.
    """
    active=[]
    for job_id in job_ids:
        p=subprocess.run(['sacct','-j',job_id,'-X','-n','-P','--format=State'],
                         text=True,capture_output=True)
        states=[line.split('|',1)[0].strip() for line in p.stdout.splitlines() if line.strip()]
        if p.returncode==0 and states and all(state.startswith('COMPLETED') for state in states):
            continue
        active.append(job_id)
    return active


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',choices=('sherlock','marlowe'),default=os.environ.get('MHD_SITE','marlowe'))
    parser.add_argument('--gpu-feature',default=None,
                        help='Sherlock GPU feature; omitted on Marlowe, whose compute nodes are H100 DGX systems')
    parser.add_argument('--dry-run',action='store_true',help='print stages and matrix; submit nothing')
    parser.add_argument('--resume',type=Path,help='continue a partial campaign ledger after a submission-limit error')
    parser.add_argument('--gpu-only',action='store_true',
                        help='submit only GLM/CT and AthenaK work; skip RAMSES and Athena++ CPU jobs')
    parser.add_argument('--gates-only',action='store_true',
                        help='submit smoke/build/pilot gates but no production matrix')
    args=parser.parse_args()
    if args.site == 'sherlock' and args.gpu_feature is None:
        args.gpu_feature='GPU_SKU:H100_SXM5'
    if args.gpu_feature and not re.fullmatch(r'GPU_(?:SKU|CC|GEN|MEM):[A-Za-z0-9_.]+',args.gpu_feature):
        parser.error('Expected a GPU feature such as GPU_SKU:H100_SXM5')
    if args.site == 'marlowe' and args.gpu_feature:
        parser.error('Do not pass --gpu-feature on Marlowe')
    jobs=matrix()
    if args.gpu_only:
        jobs=[job for job in jobs if job[2]]
    gpu_description=args.gpu_feature or 'Marlowe H100 allocation (no feature constraint)'
    print(f'Campaign: {len(jobs)} matrix commands, {gpu_description}',flush=True)
    if args.dry_run:
        print('Gates: smoke; GPU and CPU builds; Orszag–Tang 64 GPU/each external code; tile-6 throughput 96')
        for label,parts,gpu in jobs:print(('GPU ' if gpu else 'CPU ')+label+': bash tools/submit.sh '+' '.join(parts))
        return
    if not os.environ.get('MHD_RESULTS'):raise SystemExit('MHD_RESULTS must be set by the site profile')
    if not shutil_which('sbatch'):raise SystemExit(f'sbatch unavailable; run on {args.site}')
    # Fetch before submitting anything, so missing source/credentials fail early.
    # Resume uses the exact source trees fetched on the first attempt.
    if not args.resume:subprocess.run(['bash','tools/fetch_codes.sh'],cwd=BUNDLE,check=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    ledger=args.resume.resolve() if args.resume else BUNDLE/'campaigns'/f'{stamp}.tsv'
    ledger.parent.mkdir(exist_ok=True)
    previous={}
    if args.resume:
        for line in ledger.read_text().splitlines()[1:]:
            label,job_id,command=line.split('\t',2)
            if args.gpu_feature and '--constraint=GPU_' in command and f'--constraint={args.gpu_feature}' not in command:
                raise SystemExit(f'GPU feature differs from the existing campaign in {ledger}')
            previous[label]=job_id
    elif ledger.exists():raise SystemExit(f'Campaign ledger already exists: {ledger}')
    print(f'Job ledger: {ledger}',flush=True)
    with ledger.open('a' if args.resume else 'x') as log:
        if not args.resume:log.write('stage\tjob_id\tcommand\n');log.flush()
        def submit(label,parts,*,feature=False,deps=(),overrides=()):
            if label in previous:
                print(f'{label}: {previous[label]} (already submitted)',flush=True)
                return previous[label]
            command=['bash','tools/submit.sh',*parts,'--parsable',*overrides]
            if feature and args.gpu_feature:command.append(f'--constraint={args.gpu_feature}')
            if deps:command.append('--dependency=afterok:'+':'.join(deps))
            p=subprocess.run(command,cwd=BUNDLE,text=True,capture_output=True)
            if p.returncode:
                raise RuntimeError(f'Submission stopped at {label}. Earlier jobs remain queued; '
                                   f'ledger: {ledger}\n{p.stdout}{p.stderr}')
            job_id=p.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job_id):raise RuntimeError(f'Unexpected sbatch output for {label}: {p.stdout!r}')
            log.write(label+'\t'+job_id+'\t'+' '.join(command)+'\n');log.flush()
            print(f'{label}: {job_id}',flush=True)
            return job_id

        smoke=submit('gate:smoke',['smoke'],feature=True,overrides=['--time=01:30:00','--mem=16G'])
        gpu_build=submit('gate:build-gpu',['build-gpu-codes'],feature=True,deps=[smoke],overrides=['--time=02:00:00'])
        pilot_gpu=submit('gate:pilot-glm-ct',['orszag_tang'],feature=True,deps=[smoke],overrides=['--array=1'])
        pilot_tile=submit('gate:pilot-tile6',['throughput_ot3d'],feature=True,deps=[smoke],
                          overrides=['--array=0','--mem=16G'])
        pilot_ak=submit('gate:pilot-athenak',['orszag_tang','--code','athenak:matched'],
                        feature=True,deps=[gpu_build,pilot_gpu],overrides=['--array=1'])
        if args.gpu_only:
            all_gates=[smoke,gpu_build,pilot_gpu,pilot_tile,pilot_ak]
        else:
            cpu_build=submit('gate:build-cpu',['build-cpu-codes'],deps=[smoke],overrides=['--time=02:00:00'])
            pilot_ramses=submit('gate:pilot-ramses',['orszag_tang','--code','ramses:matched'],
                                deps=[cpu_build,pilot_gpu],overrides=['--array=1'])
            pilot_athenapp=submit('gate:pilot-athenapp',['orszag_tang','--code','athenapp:matched'],
                                  deps=[cpu_build,pilot_gpu],overrides=['--array=1'])
            all_gates=[smoke,gpu_build,cpu_build,pilot_gpu,pilot_tile,pilot_ak,pilot_ramses,pilot_athenapp]
        if args.gates_only:
            print('Submitted validation gates only. Submit production after every gate completes.',flush=True)
            return
        matrix_deps=active_dependencies(all_gates) if args.resume else all_gates
        for label,parts,gpu in jobs:submit('matrix:'+label,parts,feature=gpu,deps=matrix_deps)
    print(f'Submitted {len(jobs)} matrix commands behind the gates. Watch {ledger}.',flush=True)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__=='__main__':main()
