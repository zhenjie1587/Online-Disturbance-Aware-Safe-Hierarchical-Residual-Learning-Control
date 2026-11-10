"""Exploratory no-network controls without neural readiness/participation delay."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/immediate_teachers'
VARIANTS={
 'teacher_h_only':dict(high_mode='direct',low_mode='off',use_do=False),
 'teacher_l_only':dict(high_mode='off',low_mode='direct'),
 'full_teacher':dict(high_mode='direct',low_mode='direct')}

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def case(variant,scenario):
    import numpy as np,pandas as pd,torch,controller
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Only direct-teacher variants are admitted; no learned network can receive this bypass.
    assert 'learned' not in VARIANTS[variant].values()
    from controller_config import CONFIG
    def direct_participation(step,warmup,ramp,ready):
        mode=(VARIANTS[variant]['high_mode'] if warmup==CONFIG.target_warmup
              else VARIANTS[variant]['low_mode'])
        return 1.0 if mode=='direct' else 0.0
    controller.participation=direct_participation
    controller.SIM_DAYS=14
    folder=OUT/variant/scenario;folder.mkdir(parents=True,exist_ok=True)
    result=controller.run_one(scenario,str(folder),variant_name=variant,seed=2026,
        forgetting=.998,**VARIANTS[variant])
    f=pd.read_csv(result['csv_path']);assert len(f)==1344
    assert f.target_train_loss.isna().all() and f.action_train_loss.isna().all()
    assert not f.use_tcn.any() and not f.use_gru.any()
    assert f.u_hp.between(0,1).all() and np.isfinite(f[['Tz_C','u_hp']]).all().all()
    result.update(participation='Direct residuals eligible at the first decision; original gates/bounds remain. DO identification warm-up/ramp unchanged.',
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      command_tv=float(np.abs(np.diff(f.u_hp)).sum()))
    dump(folder/'result.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--scenario',choices=['peak','typical'])
    a=p.parse_args()
    if a.variant:
        case(a.variant,a.scenario);return
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'results/evaluation/source_freeze.json').read_text(encoding='utf-8'))
    source['immediate_teachers.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    freeze=OUT/'source_freeze.json'
    if freeze.exists():assert json.loads(freeze.read_text(encoding='utf-8'))==source
    else:dump(freeze,source)
    plan=[dict(variant=v,scenario=sc) for v in VARIANTS for sc in ['peak','typical']]
    dump(OUT/'plan.json',dict(jobs=plan,interpretation='Exploratory deterministic direct-teacher deployment comparison; no teacher retuning or neural training. Specified before observing these outcomes. Complements, does not replace, readiness-matched v2 controls.'))
    while True:
        status=ROOT/'results/baseline_fairness/status.json'
        if status.exists():
            s=json.loads(status.read_text(encoding='utf-8'))
            if s['finished']==s['total']:
                assert all(r['status'] in ['completed','existing'] for r in s['records']);break
        time.sleep(10)
    done=[]
    for job in plan:
        folder=OUT/job['variant']/job['scenario'];folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'result.json').exists():
            with (folder/'console.log').open('w',encoding='utf-8') as f:
                code=subprocess.run([sys.executable,str(Path(__file__)), '--variant',job['variant'],'--scenario',job['scenario']],stdout=f,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Immediate-teacher control failed: '+str(job))
        done.append(job);dump(OUT/'status.json',dict(finished=len(done),total=len(plan),completed=done))
        print('Immediate teachers',len(done),'/',len(plan),job,flush=True)

if __name__=='__main__':main()
