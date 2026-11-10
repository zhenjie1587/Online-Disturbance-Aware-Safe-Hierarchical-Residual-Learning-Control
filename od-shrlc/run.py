"""Single-run public entry point; --dry-run does not contact BOPTEST."""
import argparse,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['peak','typical','unseen_day45','unseen_day75'],default='typical')
    p.add_argument('--variant',choices=['full','direct_h','direct_l','full_teacher','wo_do','wo_compensation','wo_tcn','wo_gru','wo_gate','rule_pid_do','rule_pid','teacher_h_only','teacher_h_only_do','teacher_l_only'],default='full')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--days',type=float,default=14)
    p.add_argument('--base-url',default=os.environ.get('BOPTEST_BASE_URL','http://127.0.0.1:8080'))
    p.add_argument('--output',type=Path,default=Path('outputs/single_run'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.days<=0 or abs(a.days*96-round(a.days*96))>1e-8:p.error('--days must be a positive multiple of 1/96')
    if not a.base_url.startswith(('http://','https://')):p.error('Invalid server URL')
    plan=dict(scenario=a.scenario,variant=a.variant,seed=a.seed,days=a.days,decisions=round(a.days*96),base_url=a.base_url,forgetting=.998,output=str(a.output))
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty output folder')
    import torch,controller
    from run_matrix import VARIANTS,WINDOWS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    controller.BASE_URL=a.base_url.rstrip('/');controller.SIM_DAYS=a.days;controller.SCENARIO_START_DAY.update(WINDOWS)
    a.output.mkdir(parents=True,exist_ok=True)
    result=controller.run_one(a.scenario,str(a.output),variant_name=a.variant,seed=a.seed,environment_seed=a.seed,forgetting=.998,**VARIANTS[a.variant])
    result['public_run_plan']=plan
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
