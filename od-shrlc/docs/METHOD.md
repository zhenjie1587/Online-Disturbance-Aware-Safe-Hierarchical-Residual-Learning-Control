# Method and configuration

TCN adjusts a rule-derived temperature target. GRU adjusts the disturbance-compensated baseline heating command. Both imitate teachers that remain available online; their training/inference does not eliminate teacher computation. Direct teacher replacement differs from removing a residual channel.

The RLS predictor has generic initial coefficients and updates on observed transitions. The equivalent disturbance mixes model error, omitted gains and observation error; it does not identify individual occupancy gains. The binary input means the current lower comfort bound is at least 21°C; it is a comfort-schedule proxy, not measured occupancy. `comfort_schedule_offset = 0.5` is added to that lower bound when the proxy is one, before weather/price adjustments and clipping. Archived source uses the equivalent field name `occupied_offset`.

Target TCN: 24-step history, warm-up 96 decisions, ramp 192. Action GRU: 12-step history, warm-up 288, ramp 192. Training can begin before participation, once 64 eligible samples are available. A finite training update establishes readiness; schedule/gates determine applied authority. Short sequences repeat the earliest available feature vector. Seven days of emulator conditioning is distinct from neural participation warm-up and is not supplied as learner history.

Main runs explicitly use RLS forgetting 0.998, despite the low-level function's historical 0.995 default. The public single-run wrapper supplies 0.998. Bounds and gates constrain corrections; comfort remains a measured closed-loop outcome. The GRU provides condition-dependent command regularization and is not asserted to be necessary or uniformly better than direct/filtered teachers.
