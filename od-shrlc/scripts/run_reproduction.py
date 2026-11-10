"""Launch a retained experimental version in its own process."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',choices=['main','extended','delay','immediate','apartment'])
    p.add_argument('--stage',default='smoke')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=1)
    p.add_argument('--variant',choices=['teacher_h_only','teacher_l_only','full_teacher'],default='full_teacher')
    p.add_argument('--scenario',choices=['peak','typical'],default='typical')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/apartment')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();v=ROOT/'reproduction'/('revision_v2' if a.suite=='main' else 'revision_v3')
    if a.suite in ['main','extended']:
        allowed=['smoke','main','model','unseen','noise','weather','all'] if a.suite=='main' else ['smoke','all']
        if a.stage not in allowed:p.error('Stage must be one of '+str(allowed))
        cmd=[sys.executable,str(v/'run_matrix.py'),'--stage',a.stage,'--workers',str(a.workers)]
    elif a.suite=='delay':cmd=[sys.executable,str(v/'delay_experiment/run_delay.py'),'--workers',str(a.workers)]
    elif a.suite=='immediate':
        # Single-case mode avoids the historical batch script's unrelated baseline wait.
        cmd=[sys.executable,str(v/'immediate_teachers.py'),'--variant',a.variant,'--scenario',a.scenario]
    else:cmd=[sys.executable,str(ROOT/'scripts/run_apartment.py'),'--output',str(a.output)]
    if a.dry_run:print(subprocess.list2cmdline(cmd));return
    raise SystemExit(subprocess.call(cmd,cwd=v))
if __name__=='__main__':main()
