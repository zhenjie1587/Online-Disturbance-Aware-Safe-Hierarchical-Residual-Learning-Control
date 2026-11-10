# Reproduction commands

Run from the repository root after installing requirements and starting BOPTEST v0.8.0 at localhost:8080. Archived code uses that endpoint, irrespective of `BOPTEST_BASE_URL`. Source-freeze JSON files are metadata, not result data, and must remain in the tree.

```sh
# Original short validation matrix (8 cases, each 1.125 days):
python scripts/run_reproduction.py main --stage smoke --workers 1
# Five-seed internal comparisons; deterministic teachers use one run:
python scripts/run_reproduction.py main --stage main --workers 1
# Other main-matrix stages: model, unseen, noise, weather, all
python scripts/run_reproduction.py main --stage unseen --workers 1
# Sensitivity, low-pass teacher and auxiliary-policy extension:
python scripts/run_reproduction.py extended --stage all --workers 1
# Ten Typical one-step measurement-delay cases:
python scripts/run_reproduction.py delay --workers 1
# An immediately available deterministic teacher, without the old batch wait:
python scripts/run_reproduction.py immediate --variant full_teacher --scenario typical
# Two-zone apartment, controlled daytime loop, RLS forgetting 0.995:
python scripts/run_reproduction.py apartment --output outputs/apartment_2026
```

Each launcher accepts `--dry-run`. No simulation was started while preparing this repository. The extended matrix has its own `smoke` stage. The main matrix uses calendar start days 16/108 and additional days 45/75, with seed labels 2026–2030. These are different windows of one building, not independent climates. Sensitivity and matrix commands can run many 14-day deployments; inspect a dry-run first.

Start with one worker. More workers (up to four) require enough independently available BOPTEST worker capacity; the client does not create workers. Keep source-freeze files intact. Existing completed matrix records are skipped only when fingerprints agree; partial runs restart from empty controller state. Changing a frozen source requires a separate copy/output tree. A `STOP_AFTER_CURRENT` file in the matrix result folder prevents not-yet-started cases from proceeding.

Do not run the archived `immediate_teachers.py` without case arguments: its historical batch entry waits for an unrelated baseline status file. The supplied public launcher invokes single-case mode and avoids that dependency.

The retained apartment adapter file has a 0.998 default from later attempts. The public apartment launcher explicitly uses 0.995, matching the reported completed exploratory trial. Its I/O mapping controls the daytime valve and shared-pump request while retaining the native nighttime thermostat and heat-pump supply-temperature control. This case uses adapted settings, not an unchanged cross-building parameter transfer. One completed case does not estimate seed variability or comparative savings.

For KPIs, use `python scripts/summarize_results.py <result-folder>`. Summarize homogeneous experiment folders separately; do not pool main clean runs with equivalence checks, stress cases or repeat exports. The summarizer reports means/sample SD, not paired significance tests. Bitwise equality across different hardware, dependency builds or BOPTEST images is not guaranteed. Timing includes forecast-service effects as described in the manuscript.
