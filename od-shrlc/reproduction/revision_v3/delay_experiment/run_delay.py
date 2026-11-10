"""Run only the ten preselected Typical one-step temperature-delay jobs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
OUT = HERE / "results"
SOURCE = HERE / "source"
FILES = ("controller.py", "controller_config.py", "unified_rules.py",
         "nn_compensator_tcn_gru_v2.py", "experiment_support.py", "run_matrix.py")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one source anchor: {old!r}")
    return text.replace(old, new)


def prepare():
    frozen = json.loads((V3 / "results/evaluation/source_freeze.json").read_text())
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        data = (V3 / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == frozen[name], name
        if name in ("controller.py", "run_matrix.py"):
            code = data.decode("utf-8").replace("\r\n", "\n")
            if name == "controller.py":
                code = replace_once(code, "    noise_bias: float = 0.0,\n",
                    "    noise_bias: float = 0.0,\n    measurement_delay_steps: int = 0,\n")
                code = replace_once(code, "    sensor = MeasurementStream(steps, environment_seed, noise_sigma, noise_bias)",
                    "    if measurement_delay_steps not in (0, 1):\n"
                    "        raise ValueError('Only zero or one step of zone-temperature delay is supported')\n"
                    "    if measurement_delay_steps and (noise_sigma or noise_bias):\n"
                    "        raise ValueError('This delay experiment does not mix delay and additive noise')\n"
                    "    previous_zone_K = None\n"
                    "    sensor = MeasurementStream(steps, environment_seed, noise_sigma, noise_bias)")
                code = replace_once(code, "            observed=sensor.observe(meas,i)\n",
                    "            observed=sensor.observe(meas,i)\n"
                    "            current_zone_K = float(observed['reaTZon_y'])\n"
                    "            if measurement_delay_steps and previous_zone_K is not None:\n"
                    "                observed['reaTZon_y'] = previous_zone_K\n"
                    "            previous_zone_K = current_zone_K\n")
                code = replace_once(code, '                "measurement_noise_C": tz_c-true_tz_c,',
                    '                "measurement_noise_C": 0.0 if measurement_delay_steps else tz_c-true_tz_c,\n'
                    '                "measurement_delay_steps": measurement_delay_steps,\n'
                    '                "measurement_delay_error_C": tz_c-true_tz_c if measurement_delay_steps else 0.0,')
            else:
                anchor = "        result.update(job=job,fingerprint=fingerprint,hashes=hashes,"
                code = replace_once(code, anchor,
                    "        if job['options'].get('measurement_delay_steps') == 1:\n"
                    "            expected_temp = frame.Tz_true_C.shift(1).fillna(frame.Tz_true_C.iloc[0])\n"
                    "            if (frame.Tz_C - expected_temp).abs().max() > 1e-12:\n"
                    "                raise RuntimeError('Recorded zone-temperature input is not delayed one step')\n"
                    "            if not frame.measurement_delay_steps.eq(1).all():\n"
                    "                raise RuntimeError('Incorrect delay metadata')\n"
                    "            if job['variant'] == 'wo_compensation' and not frame.alpha_d.eq(0).all():\n"
                    "                raise RuntimeError('Compensation not disabled')\n" + anchor)
            data = code.encode("utf-8")
        target = SOURCE / name
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f"Prepared source changed: {target}")
        if not target.exists():
            target.write_bytes(data)
    return {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    hashes = prepare()
    sys.path.insert(0, str(SOURCE))
    from run_matrix import atomic, execute, worker_init

    jobs = [dict(group="delay_1step", variant=variant, scenario="typical", seed=seed,
                 environment_seed=seed, sim_days=14, config_changes={},
                 options={"measurement_delay_steps": 1})
            for seed in range(2026, 2031) for variant in ("full", "wo_compensation")]
    plan_path = OUT / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        assert previous["jobs"] == jobs and previous["hashes"] == hashes
    else:
        atomic(plan_path, dict(jobs=jobs, hashes=hashes,
            created_utc=datetime.now(timezone.utc).isoformat(),
            intervention="Lag only controller zone temperature by 900 s. Initialize with current zone temperature.",
            primary_metrics=["cost", "discomfort"],
            reference="Existing revision_v2 clean Typical full and wo_compensation, same seed labels."))
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        tasks = {pool.submit(execute, job, str(OUT), hashes): job for job in jobs}
        for future in concurrent.futures.as_completed(tasks):
            record = future.result()
            results.append(record)
            atomic(OUT / "status.json", dict(finished=len(results), total=len(jobs),
                records=[dict(job=r["job"], state=r["state"]) for r in results]))
            print(len(results), "/", len(jobs), record["state"], record["job"], flush=True)
            if record["state"] == "failed":
                print(record["error"], flush=True)
    if any(r["state"] not in ("completed", "skipped") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
