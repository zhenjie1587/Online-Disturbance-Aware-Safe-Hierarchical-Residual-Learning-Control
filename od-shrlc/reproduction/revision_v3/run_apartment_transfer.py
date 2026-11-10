"""Run one preselected cold-start transfer case with a building-specific I/O adapter."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import controller as c


ROOT = Path(__file__).resolve().parent
CASE = "twozone_apartment_hydronic"
START_DAY = 16
SEED = 2026
RLS_FORGETTING = 0.998
# Modelica HydronicSystem uses 2480 / 3600 / 2 kg/s total rated flow,
# with half allocated to each active zone at its native design stage.
ZONE_FLOW = 2480 / 3600 / 4
DAY_VALVE = "hydronicSystem_oveMDayZ"
NIGHT_VALVE = "hydronicSystem_oveMNigZ"
PUMP = "hydronicSystem_oveMpumCon"
MEASUREMENTS = {
    "reaTZon_y": "dayZon_reaTRooAir_y",
    "weaSta_reaWeaTDryBul_y": "weatherStation_reaWeaTDryBul_y",
    "weaSta_reaWeaHGloHor_y": "weatherStation_reaWeaHGloHor_y",
    "reaPHeaPum_y": "hydronicSystem_reaPeleHeaPum_y",
    "reaPPumEmi_y": "hydronicSystem_reaPPum_y",
    "reaCOP_y": "hydronicSystem_reaCOPhp_y",
}
BOUNDS = {"LowerSetp[1]": "LowerSetp[Day]", "UpperSetp[1]": "UpperSetp[Day]"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/"result.json").exists(), "Use a new run directory"
    frozen=json.loads((ROOT/"results/evaluation/source_freeze.json").read_text(encoding="utf-8"))
    hashes={name:digest(ROOT/name) for name in (
        "controller.py", "controller_config.py", "unified_rules.py",
        "experiment_support.py", "nn_compensator_tcn_gru_v2.py")}
    assert all(frozen[name]==value for name,value in hashes.items())
    save(output/"case.json",dict(case=CASE,window_days=[START_DAY,START_DAY+c.SIM_DAYS],
        benchmark_typical_day=23,warmup_days=c.WARMUP_DAYS,step_s=c.STEP,seed=SEED,
        tariff=c.PRICE_SCENARIO,variant="full",core_config=c.CONFIG.as_dict(),rls_forgetting=RLS_FORGETTING,
        hashes=hashes,adapter_sha256=digest(Path(__file__)),controlled_zone="Day",
        day_valve=DAY_VALVE,pump=PUMP,branch_design_flow_kg_s=ZONE_FLOW,
        pump_rule="branch design flow times day command plus one native night demand stage",
        retained_native_loops=["night thermostat and valve", "heat pump supply temperature weather curve"]))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c.TESTCASE_NAME=CASE
    c.SCENARIO_START_DAY={**c.SCENARIO_START_DAY, "apartment":START_DAY}
    native_init,native_forecast=c.initialize,c.put_forecast
    state={}
    rows=[]

    def alias(measurements):
        return {**measurements, **{name:measurements[source] for name,source in MEASUREMENTS.items()},
                "reaPFan_y":0.0}

    def initialize(testid, start_time, warmup_period):
        metadata={}
        for kind in ("name","inputs","measurements","forecast_points"):
            metadata[kind]=c.api_get(f"/{kind}/{testid}")["payload"]
        assert metadata["name"]["name"]==CASE,metadata["name"]
        for key in (DAY_VALVE,PUMP):
            assert key+"_u" in metadata["inputs"] and key+"_activate" in metadata["inputs"]
        for key in [*BOUNDS.values(),"LowerSetp[Night]","UpperSetp[Night]"]:
            assert key in metadata["forecast_points"],key
        save(output/"interface.json",metadata)
        measurements=native_init(testid,start_time,warmup_period)
        state["measurement"]=measurements
        return alias(measurements)

    def forecast(testid, point_names, horizon, interval):
        translated=[BOUNDS.get(key,key) for key in point_names]
        translated.extend(["LowerSetp[Night]","UpperSetp[Night]"])
        payload=native_forecast(testid,translated,horizon,interval)
        state["night_bounds"]=[payload[key][0]-273.15 for key in ("LowerSetp[Night]","UpperSetp[Night]")]
        return {**payload,**{key:payload[name] for key,name in BOUNDS.items()}}

    def advance(testid, inputs):
        current=state["measurement"]
        day=float(inputs["oveHeaPumY_u"])
        night=float(current[NIGHT_VALVE+"_u"])
        flow=ZONE_FLOW*(day+float(night>.9))
        measurement=c.api_post(f"/advance/{testid}",json={
            DAY_VALVE+"_activate":1,DAY_VALVE+"_u":day,
            PUMP+"_activate":1,PUMP+"_u":flow,
        },timeout=240)["payload"]
        applied_day=float(measurement[DAY_VALVE+"_u"])
        applied_flow=float(measurement[PUMP+"_u"])
        if abs(applied_day-day)>1e-6 or abs(applied_flow-flow)>1e-6:
            raise RuntimeError("Emulator did not apply the requested valve or pump command")
        rows.append(dict(time_s=current["time"],next_time_s=measurement["time"],
            day_C=current["dayZon_reaTRooAir_y"]-273.15,
            night_C=current["nigZon_reaTRooAir_y"]-273.15,
            night_low_C=state["night_bounds"][0],night_high_C=state["night_bounds"][1],
            day_valve=applied_day,night_valve=night,pump_kg_s=applied_flow,
            day_floor_heat_W=measurement["dayZon_reaPowFlooHea_y"],
            night_floor_heat_W=measurement["nigZon_reaPowFlooHea_y"]))
        state["measurement"]=measurement
        return alias(measurement)

    c.initialize,c.put_forecast,c.advance=initialize,forecast,advance
    try:
        result=c.run_one("apartment",str(output),variant_name="full",seed=SEED,
                         environment_seed=SEED,forgetting=RLS_FORGETTING)
    finally:
        pd.DataFrame(rows).to_csv(output/"apartment_interface_trace.csv",index=False)
    trace=pd.DataFrame(rows)
    series=pd.read_csv(result["csv_path"])
    assert len(trace)==len(series)==1344
    np.testing.assert_array_equal(trace.time_s,series.time_s)
    np.testing.assert_allclose(trace.next_time_s-trace.time_s,c.STEP)
    np.testing.assert_allclose(trace.day_valve,series.u_hp,atol=1e-6)
    assert trace.day_valve.between(0,1).all() and trace.pump_kg_s.between(0,2*ZONE_FLOW+1e-6).all()
    assert np.isfinite(trace.to_numpy()).all()
    assert series.beta_t.max()>0 and series.alpha_u_raw.max()>0
    assert (trace.day_valve>0).any() and (trace.day_floor_heat_W>0).any()
    for label,temp,lo,hi in [
        ("day",trace.day_C,series.T_low,series.T_high),
        ("night",trace.night_C,trace.night_low_C,trace.night_high_C),
    ]:
        below=(lo-temp).clip(lower=0)
        above=(temp-hi).clip(lower=0)
        result[label]=dict(min_C=float(temp.min()),max_C=float(temp.max()),
            sampled_below_hours=float((below>0).sum()*c.STEP/3600),
            sampled_above_hours=float((above>0).sum()*c.STEP/3600),
            sampled_degree_hours=float((below+above).sum()*c.STEP/3600))
    result["case"]=CASE
    result["valve_active_steps"]=int((trace.day_valve>0).sum())
    result["target_max_participation"]=float(series.beta_t.max())
    result["action_max_participation"]=float(series.alpha_u.max())
    result["action_nonzero_residual_steps"]=int((series.alpha_u*series.delta_u_nn!=0).sum())
    result["validation_passed"]=True
    save(output/"result.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]).resolve())
