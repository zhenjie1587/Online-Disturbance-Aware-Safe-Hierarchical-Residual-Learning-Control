"""Frozen reviewer-driven BOPTEST matrix with independent, resumable run records."""
import argparse
import concurrent.futures
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
SEEDS=[2026,2027,2028,2029,2030]
WINDOWS={'peak':16,'typical':108,'unseen_day45':45,'unseen_day75':75}
VARIANTS={
 'full':dict(high_mode='learned',low_mode='learned'),
 'direct_h':dict(high_mode='direct',low_mode='learned'),
 'direct_l':dict(high_mode='learned',low_mode='direct'),
 'full_teacher':dict(high_mode='direct',low_mode='direct'),
 'wo_do':dict(high_mode='learned',low_mode='learned',use_do=False),
 'wo_compensation':dict(high_mode='learned',low_mode='learned',use_compensation=False),
 'wo_tcn':dict(high_mode='off',low_mode='learned'),
 'wo_gru':dict(high_mode='learned',low_mode='off'),
 'wo_gate':dict(high_mode='learned',low_mode='learned',use_gate=False),
 'rule_pid_do':dict(high_mode='off',low_mode='off'),
 'rule_pid':dict(high_mode='off',low_mode='off',use_do=False),
 'teacher_h_only':dict(high_mode='direct',low_mode='off',use_do=False),
 'teacher_h_only_do':dict(high_mode='direct',low_mode='off'),
 'teacher_l_only':dict(high_mode='off',low_mode='direct'),
}
SOURCES=['controller.py','controller_config.py','unified_rules.py',
         'nn_compensator_tcn_gru_v2.py','experiment_support.py','run_matrix.py']


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def build_matrix(stage):
    jobs=[]
    def add(group,variant,config=None,options=None,days=14,seeds=SEEDS):
        for scenario in ['peak','typical']:
            for seed in seeds:
                jobs.append(dict(group=group,variant=variant,scenario=scenario,seed=seed,
                    environment_seed=seed,sim_days=days,options=options or {},config_changes=config or {}))
    if stage=='smoke':
        add('smoke_reference','full',days=1.125,seeds=SEEDS[:1])
        add('smoke_bounds','full',dict(target_bound=1.0,action_bound=.1),days=1.125,seeds=SEEDS[:1])
        add('smoke_filter','direct_l',options=dict(direct_filter_alpha=.25),days=1.125,seeds=SEEDS[:1])
        add('smoke_auxiliary','rule_pid',options=dict(auxiliary_mode='full_on'),days=1.125,seeds=SEEDS[:1])
        return jobs
    add('reference_check','full',seeds=SEEDS[:1])
    factors={
       'gate':(.75,1.25), 'comfort_penalty':(.5,2.), 'smooth_weight':(.5,2.),
       'residual_bounds':(.75,1.25), 'participation_time':(.5,2.)}
    from controller_config import CONFIG
    for factor,levels in factors.items():
        for scale in levels:
            fields={'gate':['sample_margin','directional_margin','hard_margin'],
              'comfort_penalty':['teacher_low_weight','teacher_high_weight'],
              'smooth_weight':['teacher_smooth_weight'],
              'residual_bounds':['target_bound','action_bound'],
              'participation_time':['target_warmup','target_ramp','action_warmup','action_ramp']}[factor]
            changes={k:getattr(CONFIG,k)*scale for k in fields}
            if factor=='participation_time':changes={k:int(v) for k,v in changes.items()}
            add(f'sensitivity_{factor}_{scale:g}','full',changes)
    add('simple_filter','direct_l',options=dict(direct_filter_alpha=.25))
    add('auxiliary_full_on','full',options=dict(auxiliary_mode='full_on'))
    add('auxiliary_full_on','rule_pid',options=dict(auxiliary_mode='full_on'),seeds=SEEDS[:1])
    return jobs


def worker_init():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def execute(job,out,hashes):
    import numpy as np
    import pandas as pd
    import torch
    import controller
    import controller_config, unified_rules
    from dataclasses import replace
    CONFIG=replace(controller_config.ControllerConfig(),**job['config_changes'])
    controller.CONFIG=CONFIG
    unified_rules.C=CONFIG
    out=Path(out)
    run=out/job['group']/job['variant']/job['scenario']/f"seed_{job['seed']}"
    run.mkdir(parents=True,exist_ok=True)
    if (out/'STOP_AFTER_CURRENT').exists():
        return dict(state='deferred',job=job)
    fingerprint=hashlib.sha256(json.dumps(dict(job=job,hashes=hashes),sort_keys=True).encode()).hexdigest()
    result_path=run/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['fingerprint']!=fingerprint:raise RuntimeError('Existing run fingerprint differs')
        return dict(state='skipped',job=job,result=result)
    # Incomplete groups are rerun from empty controller state; do not reuse a partial trajectory.
    atomic(run/'case.json',dict(job=job,hashes=hashes,fingerprint=fingerprint,config=CONFIG.as_dict()))
    start=time.perf_counter()
    try:
        controller.SCENARIO_START_DAY.update(WINDOWS)
        controller.SIM_DAYS=job['sim_days']
        with (run/'console.log').open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            result=controller.run_one(job['scenario'],str(run),variant_name=job['variant'],seed=job['seed'],
                                      forgetting=.998,environment_seed=job['environment_seed'],
                                      **VARIANTS[job['variant']],**job['options'])
        frame=pd.read_csv(result['csv_path'])
        expected=round(job['sim_days']*96)
        if len(frame)!=expected or not frame.time_s.diff().dropna().eq(900).all():
            raise RuntimeError('Incomplete or misaligned trajectory')
        if not np.isfinite(frame[['Tz_C','Tz_true_C','u_hp','Tsp_C','d_hat']].to_numpy()).all():
            raise RuntimeError('Non-finite measurements/actions')
        if not frame.u_hp.between(0,1).all():raise RuntimeError('Action outside bounds')
        modes=VARIANTS[job['variant']]
        for mode,alpha,loss in [('high_mode','beta_t','target_train_loss'),('low_mode','alpha_u','action_train_loss')]:
            if modes[mode]=='learned' and not frame.loc[frame[alpha]>0,loss].notna().all():
                raise RuntimeError('Network participation before training')
            if modes[mode]!='learned' and frame[loss].notna().any():
                raise RuntimeError('Disabled/direct branch trained a network')
        if job['options'].get('noise_sigma',0)==0 and frame.measurement_noise_C.abs().max()>1e-10:
            raise RuntimeError('Unexpected measurement corruption')
        if job['options'].get('measurement_delay_steps') == 1:
            expected_temp = frame.Tz_true_C.shift(1).fillna(frame.Tz_true_C.iloc[0])
            if (frame.Tz_C - expected_temp).abs().max() > 1e-12:
                raise RuntimeError('Recorded zone-temperature input is not delayed one step')
            if not frame.measurement_delay_steps.eq(1).all():
                raise RuntimeError('Incorrect delay metadata')
            if job['variant'] == 'wo_compensation' and not frame.alpha_d.eq(0).all():
                raise RuntimeError('Compensation not disabled')
        result.update(job=job,fingerprint=fingerprint,hashes=hashes,
                      elapsed_seconds=time.perf_counter()-start,
                      environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                                       pandas=pd.__version__,platform=platform.platform()),
                      control_ms_mean=float(frame.controller_seconds.mean()*1000))
        atomic(result_path,result)
        gc.collect()
        return dict(state='completed',job=job,result=result)
    except BaseException:
        error=traceback.format_exc()
        atomic(run/'failure.json',dict(job=job,error=error))
        # Ownership was recorded before queue waiting; cleanup only this run's session.
        if (run/'session.json').exists():
            try:controller.stop(json.loads((run/'session.json').read_text())['testid'])
            except Exception:pass
        return dict(state='failed',job=job,error=error)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','all'],default='smoke')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.workers<1 or args.workers>4:p.error('workers must be 1 to 4')
    out=ROOT/'results'/('smoke' if args.stage=='smoke' else 'evaluation')
    out.mkdir(parents=True,exist_ok=True)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
    freeze=out/'source_freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=hashes:raise RuntimeError('Code changed after freeze; use a new version directory')
    else:
        atomic(freeze,hashes)
        snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
        for name in SOURCES:(snap/name).write_bytes((ROOT/name).read_bytes())
    jobs=build_matrix(args.stage)
    atomic(out/f'plan_{args.stage}.json',dict(jobs=jobs,created_utc=datetime.now(timezone.utc).isoformat(),
                  rationale='Prespecified one-factor local sensitivity; v2 nominal configuration retained, no tuning on outcomes.',
                  measurement_noise='Synthetic stress levels, not a claim about field sensor accuracy.',
                  unseen_scope='Same building, disjoint calendar windows; no new-building/climate claim.'))
    finished=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        futures=[pool.submit(execute,job,str(out),hashes) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            record=future.result();finished.append(record)
            counts={s:sum(r['state']==s for r in finished) for s in ['completed','skipped','failed','deferred']}
            atomic(out/'batch_status.json',dict(stage=args.stage,total=len(jobs),finished=len(finished),counts=counts,
                                               latest=record['job'],updated_utc=datetime.now(timezone.utc).isoformat()))
            j=record['job'];print(f"{len(finished)}/{len(jobs)} {record['state']} {j['group']}/{j['variant']}/{j['scenario']}/{j['seed']}",flush=True)
            if record['state']=='failed':print(record['error'],flush=True)
    if any(r['state'] in ['failed','deferred'] for r in finished):
        raise RuntimeError('Some planned cases incomplete; inspect batch_status and failure records')
    print('Batch complete',flush=True)


if __name__=='__main__':main()
