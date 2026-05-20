# Ant Behavior Benchmarks

Read this before changing ant movement, pheromone emission/decay, food
handling, or benchmark scoring. The common failure mode is improving a number
while missing what the sim visibly does.

## Behavior Rules

- No perfect hidden GPS: returning ants may use a flawed 30x30 map planner,
  but must not use Home-smell fallback or any mechanism that routes through
  walls.
- Direct home movement is acceptable when the coarse map planner has a clear
  route. The hard failure is aiming home through a blocking wall, not a
  planned straight return through open space.
- Ants should repeatedly find food, bring it home, leave home, and forage
  again.
- Depleted food must not leave a dead-source ball. FoodSmell should clear, and
  stale routes should become less attractive.
- Stale or useless trails should emit Repellent/no-entry pheromone, and ants
  must actually avoid that signal.
- Curved or duplicate paths should converge toward shorter useful routes when
  reinforcement supports that.
- Closed loops with no food should decay. They should not magically turn into
  useful straight paths.
- In open space with no food, no walls, and no pheromone signal, workers should meander instead of walking in long straight lines.
- Wall scenarios must recover without GPS and must reject broad off-route
  trail clutter, side loops, and wall-basin milling.
- Manual food placement must work repeatedly, including click-drag placement.
- Dead ants are removed immediately. Carrying ants may drop the food they were
  already carrying before death, but the dead body itself must not create a
  corpse or extra Food object.
- The UI default pheromone intensity is `0.30`.
- Keep the bench suite small and targeted. Add or keep a bench only when it
  protects a specific behavior rule.
- Run simulations and inspect visual output. Do not trust score alone.
- Normal GUI runs include mild seeded I/O imperfections. The scored benches
  set perception, motor, and deposit noise to zero so scores remain a stable
  brain/pheromone behavior contract.

## "Show Benchmark"

When the user asks to "show benchmark", "show the benchmark", "whole bench", or
asks for a brain-vs-bench chart, answer with the latest validated comparison
data below unless they explicitly ask to rerun. If they ask for current scores,
run:

```bash
./target/release/ant-backend bench_default --brain neural
./target/release/ant-backend bench_default --brain classic
./target/release/ant-backend bench_default --brain weighted
```

The answer must include one table with these columns, in this order:

| Brain | Bench | Score | Thresh For PASS | Range | Desc |
|---|---|---:|---:|---|---|

Rules:

- After any long refinement/tuning loop, show the benchmark numbers even if
  the user did not explicitly ask again. Include the current scores and say
  which commands were run.
- Include every scored bench for each selectable worker brain.
- Keep `Desc` as the last column.
- Include a separate composite table with `Total score` and `Gate`.
- State that scores are positive and higher is better.
- Use the global score threshold: `40.0` per bench, `360.0` composite.
- State that hard behavior gates still override score.
- Do not answer with only the total score.
- `classic` remains the rule-based baseline. `neural` is the learned-turn
  comparison point. `weighted` remains the parameterized comparison brain.

## Current Benchmark Chart

Latest validated whole-bench comparison:

| Brain | Bench | Score | Thresh For PASS | Range | Desc |
|---|---|---:|---:|---|---|
| `neural` | `wall_test` | `80.3` | `40.0` | `0.0` to `100.0` | Wall routing must find home quickly around obstacles and sustain visible food-to-home carrier traffic |
| `neural` | `arc_to_line` | `100.0` | `40.0` | `0.0` to `100.0` | Curved pheromone route should shorten toward a chord corridor without straight-home return dashes |
| `neural` | `multi_path` | `99.5` | `40.0` | `0.0` to `100.0` | Multiple paths to same food should favor the shorter/faster path corridor |
| `neural` | `loop_decay` | `91.9` | `40.0` | `0.0` to `100.0` | Closed false loop should decay, not become a magic path |
| `neural` | `food_cycle` | `84.3` | `40.0` | `0.0` to `100.0` | Depleted pile recovery, second food placement, no dead-source ball |
| `neural` | `post_pickup` | `85.7` | `40.0` | `0.0` to `100.0` | Carrier pickup produces enough return traces; planned direct return is allowed |
| `neural` | `lost_carrier` | `76.9` | `40.0` | `0.0` to `100.0` | Carriers with no Home trail keep searching instead of turning back to food |
| `neural` | `meander` | `91.4` | `40.0` | `0.0` to `100.0` | No-pheromone open-field workers should curve and wander, not march in straight lines |
| `neural` | `cluster_escape` | `100.0` | `40.0` | `0.0` to `100.0` | Dense wall-side, wall-throat, and active-food wall-pocket clumps disperse instead of pile up |
| `classic` | `wall_test` | `80.3` | `40.0` | `0.0` to `100.0` | Wall routing must find home quickly around obstacles and sustain visible food-to-home carrier traffic |
| `classic` | `arc_to_line` | `100.0` | `40.0` | `0.0` to `100.0` | Curved pheromone route should shorten toward a chord corridor without straight-home return dashes |
| `classic` | `multi_path` | `99.5` | `40.0` | `0.0` to `100.0` | Multiple paths to same food should favor the shorter/faster path corridor |
| `classic` | `loop_decay` | `91.9` | `40.0` | `0.0` to `100.0` | Closed false loop should decay, not become a magic path |
| `classic` | `food_cycle` | `84.3` | `40.0` | `0.0` to `100.0` | Depleted pile recovery, second food placement, no dead-source ball |
| `classic` | `post_pickup` | `85.7` | `40.0` | `0.0` to `100.0` | Carrier pickup produces enough return traces; planned direct return is allowed |
| `classic` | `lost_carrier` | `72.9` | `40.0` | `0.0` to `100.0` | Carriers with no Home trail keep searching instead of turning back to food |
| `classic` | `meander` | `91.4` | `40.0` | `0.0` to `100.0` | No-pheromone open-field workers should curve and wander, not march in straight lines |
| `classic` | `cluster_escape` | `100.0` | `40.0` | `0.0` to `100.0` | Dense wall-side, wall-throat, and active-food wall-pocket clumps disperse instead of pile up |
| `weighted` | `wall_test` | `89.0` | `40.0` | `0.0` to `100.0` | Wall routing must find home quickly around obstacles and sustain visible food-to-home carrier traffic |
| `weighted` | `arc_to_line` | `61.1` | `40.0` | `0.0` to `100.0` | Curved pheromone route should shorten toward a chord corridor without straight-home return dashes |
| `weighted` | `multi_path` | `80.3` | `40.0` | `0.0` to `100.0` | Multiple paths to same food should favor the shorter/faster path corridor |
| `weighted` | `loop_decay` | `99.6` | `40.0` | `0.0` to `100.0` | Closed false loop should decay, not become a magic path |
| `weighted` | `food_cycle` | `94.3` | `40.0` | `0.0` to `100.0` | Depleted pile recovery, second food placement, no dead-source ball |
| `weighted` | `post_pickup` | `85.4` | `40.0` | `0.0` to `100.0` | Carrier pickup produces enough return traces; planned direct return is allowed |
| `weighted` | `lost_carrier` | `77.4` | `40.0` | `0.0` to `100.0` | Carriers with no Home trail keep searching instead of turning back to food |
| `weighted` | `meander` | `74.3` | `40.0` | `0.0` to `100.0` | No-pheromone open-field workers should curve and wander, not march in straight lines |
| `weighted` | `cluster_escape` | `55.0` | `40.0` | `0.0` to `100.0` | Dense wall-side, wall-throat, and active-food wall-pocket clumps disperse instead of pile up |

| Brain | Total Score | Thresh For PASS | Range | Gate |
|---|---:|---:|---|---|
| `neural` | `809.9` | `360.0` | `0.0` to `900.0` | `PASS` |
| `classic` | `805.9` | `360.0` | `0.0` to `900.0` | `PASS` |
| `weighted` | `716.5` | `360.0` | `0.0` to `900.0` | `PASS` |

Scores are normalized positive values. Each bench is reported on `0.0..100.0`; the global numeric threshold is `40.0` per bench and `360.0` composite. Hard behavior gates still override the total score. All three selectable worker brains pass the current scored suite.

`cluster_escape` has a visual-pile cap: if a wall-bottleneck or active-food
wall clump is visible, the score is capped to a low value even when other ants
are still moving. This prevents mean displacement from hiding the exact
right-of-wall pileup seen in the GUI.

## Wall `100.0`

A `wall_test` score of `100.0` means the wall scenario is not merely passing;
it has saturated the score after hard-gate checks. Visually, it should look
like this:

- A clear top or bottom corridor is established around the wall early.
- Food stored is already increasing by the 8k-tick checkpoint.
- By 12k ticks, the route is stable and carrying ants are repeatedly returning
  home instead of exploring around the food.
- Carriers do not aim through the wall; `wAim` and `wallAim` stay near zero.
- Most traffic is on a narrow around-wall route, not a broad yellow haze.
- Side loops, wall-basin milling, and off-route trail clutter stay below
  their gates.

In metric terms, `100.0` requires the weighted wall score to reach the cap: roughly `bRet/bFast/bPrompt` near `1.0`, mean behind-wall return close to or below `1200` ticks, at least `200` deliveries by 8k, at least `600` by 12k, `1100+` total deliveries, `route >= 0.25`, and little or no penalty from `wallPr`, `offRoute`, `offTrail`, `offClump`, blocked-wall aiming, or branch count.

## Current Defaults

```text
home_diffusion           = 0.030
worker_brain            = classic
food_lay_strength       = 1.5
outbound_lay_threshold  = 0.5
sensor_dist             = 24.0
deposit_decay_horizon   = 1200
bilinear_deposit        = false
pheromone_intensity_ui  = 0.30
wall_home_diffusion_cap = 0.020
wall_trail_lay_scale    = 0.52   # effective wall trail lay strength = 0.78
wall_trail_decay_cap    = 0.9975
body_repel_radius       = 8.5
body_repel_strength     = 0.42
active_food_wall_repel_radius = 42.0
active_food_wall_repel_strength = 5.0
active_food_wall_context = near wall body, near active food pile, not at food/nest
wall_bottleneck_escape = classic/neural dense wall pile only; weighted uses its own wall-crowd policy
blocked_wall_food_deposit = suppress Food reinforcement for blocked wall-bound carrier approaches
multi_food_wall_context = wall map has ever had more than one active food pile
weighted_wall_food_smell_scale = 0.90 normally, 0.10 in multi_food_wall_context
trail_wall_follow_ticks = 24
weighted_wall_follow_ticks = 18
weighted_wall_follow_scale = 0.30
weighted_wall_crowd_follow_ticks = 24
weighted_wall_crowd_follow_scale = 0.35
weighted_wall_crowd_tangent_weight = 1.5
classic_stale_trail_limit = 35       # wall+food maps use 60 to avoid false wall-route rejection
classic_open_long_lay_scale = 0.05   # far open return routes keep topology without branch fog
no_food_declump_context = food_piles == 0
open_no_signal_meander = smooth low-frequency turn bias when no food/walls/pheromone signal exists
food_smell_search_weight = 1.5
map_planner              = 30x30 coarse per-ant map
map_plan_noise           = 0.10
map_success_dilate       = 64.0   # successful route memory spreads by physical radius
map_home_weave_min_dist  = 180.0  # short open-field food cycles can use noisy home weave
weighted_long_return_bend = on    # weighted-only guard after breadcrumb replay
weighted_open_return_dot = 0.75
weighted_open_return_weave = 1.25
weighted_open_return_blend = 0.85
weighted_open_return_no_route_weave = 1.05
weighted_open_return_no_route_blend = 0.75
weighted_open_long_lay_scale = 0.40
weighted_open_route_deviation_keep = 0.16
weighted_open_route_jitter = 22.0
perception_position_noise = 0.03  # GUI/default sim; scored benches set to 0
perception_heading_noise  = 0.001
perception_signal_noise   = 0.002
motor_speed_noise         = 0.01
motor_turn_noise          = 0.001
deposit_strength_noise    = 0.01
deposit_position_noise    = 0.0
weighted_wall_no_signal_jitter_mult = 2.2
weighted_wall_route_memory_scale = 1.57
weighted_wall_no_signal_neighbor_avoid_weight = 2.0
weighted_blocked_home_deposit_band = 62.0
map_cost_norm            = false  # tested; worsened 40x40 wall routing
map_noise_scale          = false  # tested; worsened 40x40 wall routing
map_wall_dilate          = 0.0    # tested; broadened off-route clutter
map_home_plan_weight     = 2.2
neural_obs_dim           = 40     # 24 local sensors + 16 coarse-map features
neural_default_blend     = 0.55
neural_default_min_ticks = 0
```

Latest validated diagnostic metrics snapshot. The normalized score table above is the source of truth for current brain-vs-bench results; refresh this block with the individual regression commands below when detailed diagnostics are needed.

```text
weighted wall: deliveries=1386 checkpoints=11->221->708->1386 route=0.09 wallPr=185 bRet=0.81 bFast=0.59 bPrompt=0.67 mean=1633 stream=145.7/188 scatter=0.16 offRoute=0.51 offTrail=0.23 wallDz=0.20 offClump=0.02 branch=2291
weighted arc: deliveries=1086 checkpoints=99->425->855->1086 straight=0.99 arc=0.00 off=0.01 meanLineDist=14.2 direct=0/285418 chord=149621/285418 returnDist=40.3 perfect=4/1264 scatter=0.77 branch=1166
weighted multi_path: short=0.98 long=0.00 off=0.02 scatter=0.62 deliveries=629
loop_decay: final=0.000 swarm=0.04 ticks=8000
food_cycle: first=45 second=45 old=0.00/0.01 smell=0.00
post_pickup: direct=0/43474 line=0/120 streak=0
lost_carrier classic: back=370/25761 ret=0/120
lost_carrier neural: back=321/25579 ret=0/120 maxDrop=42.0
classic cluster_escape: score=100.0 from bench_default; detailed rate snapshot not refreshed after blocked-wall deposit filtering
neural cluster_escape: score=100.0 from bench_default; detailed rate snapshot not refreshed after blocked-wall deposit filtering
weighted cluster_escape: score=55.0 workers=180 disp=933.3 sample=0.14 final=0.09 bottle=0.04/0.01/0.06 active=0.00/0.00/0.00 rev=0.00 trap=0.00
classic meander: score=91.4 workers=24 path=899.8 disp=617.1 straight=0.686 line=0.000 turn=0.167
neural meander: score=91.4 workers=24 path=899.8 disp=617.1 straight=0.686 line=0.000 turn=0.167
weighted meander: score=74.3 workers=24 path=894.8 disp=625.4 straight=0.699 line=0.000 turn=0.040
```

## Commands

```bash
cargo test --locked -p ant-backend
cargo build --locked --release -p ant-backend
./target/release/ant-backend bench_default --brain neural
./target/release/ant-backend bench_default --brain classic
./target/release/ant-backend bench_default --brain weighted
./target/release/ant-backend wall_regression --brain classic
./target/release/ant-backend arc_regression --brain classic
./target/release/ant-backend food_cycle_regression --brain classic
./target/release/ant-backend meander_regression --brain classic
./target/release/ant-backend cluster_regression --brain classic
./target/release/ant-backend dump_wall_test --brain classic
./target/release/ant-backend dump_arc_to_line --brain classic
./target/release/ant-backend dump_arc_progress --brain classic
./target/release/ant-backend dump_food_cycle --brain classic
```

Dump images are written as PPM files in `/tmp/claude/`; convert them to PNG
for visual inspection when needed.

Use `--brain weighted` or `--brain neural` on any of those commands to compare
alternate worker brains against the classic baseline. The neural brain reads
`assets/neural_worker_weights.json` by default, or the path in
`REALANTSIM_NEURAL_WORKER_WEIGHTS`; if no weights are present, it falls back
to classic behavior.

`bench_default` runs the nine scored benches concurrently for one brain and is the canonical normalized score source; the latest full rows take about 3.5 to 4.5 minutes on this host.

## Metric Glossary

- `wAim`, `wSt`: wall-test carriers aiming home through a blocking wall.
- `wClr`, `clearLine`: carriers after clearing the wall. Direct clear-line
  returns are now context metrics, not failure gates.
- `bRet`: behind-wall pickups that return home within the trace window.
- `bFast`, `bPrompt`, `mean`: behind-wall pickup-to-home latency. These and a
  bounded throughput component drive `wall_test`; final delivery count alone
  is not enough to pass.
- `stream`: wall-test loaded carriers away from nest/food endpoints. It catches
  the visual failure where food is found but no food-to-home traffic is visible.
- `scatter`: workers away from nest/food that are not on Home or Food trails.
- `homeDash`: carrying ants aimed at the nest while not on Home trail. It is
  context only when the 30x30 planner has a clear route.
- `branch`, `dead`, `cover`: route clutter, abandoned fragments, and map-wide
  pheromone fog/webbing.
- `offRoute`: workers outside the ideal around-wall corridor; context only.
- `offTrail`: off-route workers standing on Home or Food pheromone. This is
  the hard metric for broad side-loop clutter; the current gate is `<=0.28`.
- `wallDz`: wall-side or loop-basin milling. Scored and gated, but broader
  than `offTrail`.
- `offClump`: locally dense off-route worker groups; complements
  `cluster_escape`.
- `cluster bottle`: `worker/clump/peak` rates for the dense-colony wall-throat
  probe. `worker` counts traffic in the central blocked wall zone; `clump` and
  `peak` are the hard pileup signals.
- `cluster active`: `worker/clump/peak` rates for an active food pile near a
  wall. This is the regression for visible food-side wall balls in the GUI.
- `straight`, `arc`, `off`, `meanLineDist`: arc-to-line shortening metrics.
- `arc direct`, `arc chord`, `returnDist`, `arc perfect`: post-pickup carrier
  realism in the arc bench. `direct` counts immediate heading-to-home samples.
  `chord` counts returns riding the narrow straight nest-food chord.
  `returnDist` is the average return distance from that chord; too low means
  a visible straight-home rail. `perfect` counts near-straight traces. For
  `direct`, `chord`, and `perfect`, lower is better.
- `multi short`, `multi long`, `multi off`: duplicate-route selection metrics.
- `loop final`, `loop swarm`: false-loop chemical decay and temporary milling.
  The bench runs long enough for ants to disperse after the false trail
  chemically decays.
- `meander path`, `disp`, `straight`, `line`, `turn`: open-field no-signal wandering. `straight` is displacement divided by path length, `line` is the fraction of long traces that stayed too straight, and `turn` is mean absolute heading change per moving sample.
- `food_cycle old`, `food_cycle smell`: depleted-source swarm and phantom
  FoodSmell.
- `post_pickup pDash`, `pLine`, `stk`: context metrics for pickup returns.
  Planned direct returns are allowed; blocked-wall aiming is still forbidden.
- `lost_carrier back`, `ret`, `maxDrop`: no-Home-signal carrier recovery.

## Review Checklist

Before changing behavior:

1. Identify which rule the change protects.
2. Run the relevant bench or dump before and after.
3. Inspect visual output; the table is not enough.
4. Reject changes that improve one scenario by breaking another.
5. Update this file when a new repeated failure pattern appears.

## Lessons

Repeated mistakes:

- Reintroducing nest-vector or Home-smell fallbacks that behave like GPS.
- Treating FoodSmell as route evidence. FoodSmell means food is nearby; it
  does not prove a usable path exists.
- Letting FoodSmell linger after depletion, creating a phantom target.
- Using throughput alone as proof. High deliveries can come from unrealistic
  webby paths.
- Treating Repellent as sufficient without verifying ants avoid it.
- Fixing a screenshot symptom without adding a bench that catches it.
- Treating isolated `cluster_escape` as proof that wall-route bottleneck
  congestion is fixed. The bench now includes a dense wall-throat subtest, but
  broad off-route trail following still belongs to `wall_test`.
- Treating no-food cluster escape as proof that active food-side wall pockets
  are fixed. Those need their own `cluster active` worker/clump/peak metrics.
- Dropping near-wall breadcrumbs from return memory; wall-corner breadcrumbs
  can be the only local evidence for how to get around a barrier.
- Removing the world-level no-GPS hook without replacing it in the brain
  immediately reintroduces bad home dashes. With the 30x30 planner, clear
  direct returns are acceptable, but through-wall home aiming must still be
  validated by `wall_test`.

Useful patterns:

- Keep the blocked-wall gate hard: `wAim` and behind-wall `wallAim`.
  `wClr` and `clearLine` are now context metrics.
- Use wall-only tuning for obstacle-only issues; do not disturb open-field
  arc/multi/food-cycle behavior unless the bench proves it is necessary.
- `bRet` made behind-wall recovery measurable.
- `offRoute`, `offTrail`, `wallDz`, and `offClump` made side-loop clutter
  measurable. `offTrail` is the most useful hard signal.
- For the coarse map, successful-route dilation helped because it spreads
  proven path memory over a physical radius. Cost normalization, noise scaling,
  and wall dilation were tested and worsened wall routing or clutter.
- For `loop_decay`, distinguish chemical persistence from physical dispersal.
  The false trail should have zero remaining Food pheromone, then ants need
  enough ticks to walk out of the old ring before scoring swarm.
- Mild wall-only trail evaporation helped topology. Stronger evaporation,
  stronger wall laying, broad crowd repellent, and Home-sensor fallback looked
  plausible but hurt route quality or wall-basin behavior.

Removed from the main score:

- Density/overlap: too indirect for route quality.
- Pesticide: real feature, but unrelated to trail-route tuning.
- Generic dissipation: replaced by concrete depleted-food and loop scenarios.

## Realism Snapshot

```text
visual plausibility: 9/10 in the GUI; diagnostic PPM dumps look harsher because ants are overlaid as bright white dots
foraging/trail dynamics: 7.5/10; covered by wall, arc, multi-path, loop-decay, food-cycle, post-pickup, and lost-carrier gates
biological accuracy: 7/10 as a stylized foraging model; not a species-level biology simulator
```
