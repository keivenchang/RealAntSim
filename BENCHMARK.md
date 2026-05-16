# Ant Behavior Benchmarks

Read this before changing ant movement, pheromone emission/decay, food
handling, or benchmark scoring. The common failure mode is improving a number
while missing what the sim visibly does.

## Behavior Rules

- No hidden GPS: returning ants must not use a direct nest vector, Home-smell
  fallback, or any mechanism that creates a perfect home beeline.
- After pickup, a carrier must not immediately dash straight home. It must
  search, follow real Home pheromone, or use local route memory.
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
- Wall scenarios must recover without GPS and must reject broad off-route
  trail clutter, side loops, and wall-basin milling.
- Manual food placement must work repeatedly, including click-drag placement.
- The UI default pheromone intensity is `0.30`.
- Keep the bench suite small and targeted. Add or keep a bench only when it
  protects a specific behavior rule.
- Run simulations and inspect visual output. Do not trust score alone.

## "Show Benchmark"

When the user asks to "show benchmark", "show the benchmark", or asks for the
benchmark chart, answer with the latest validated default-row data below unless
they explicitly ask to rerun. If they ask for current scores, run:

```bash
./target/release/ant-backend path_regression
```

The answer must include one table with these columns, in this order:

| Benchmark | Score | Range / Read | PASS Criteria | Benchmark Description |
|---|---:|---|---|---|

Rules:

- Include every scored bench in the current chart.
- Keep `Benchmark Description` as the last column.
- Include a separate composite table with `Total score` and `Gate`.
- State that higher scores are better, but hard gates override score.
- Do not answer with only the total score.

## Current Benchmark Chart

Latest validated default row:

| Benchmark | Score | Range / Read | PASS Criteria | Benchmark Description |
|---|---:|---|---|---|
| `wall_test` | `-193729` | `-250000` to `-150000`; passing, biggest drag | `deliveries >= 250`, `route >= 0.03`, `wallPr <= 250`, `wAim <= 0.05`, `wSt <= 8`, `wClr <= 0.01`, `clearLine = 0`, `bRet >= 0.20`, `mean return <= 3000`, `wallAim <= 0.03`, `scatter <= 0.85`, `offTrail <= 0.20`, `wallDz <= 0.35`, `offClump <= 0.08` | Wall routing, behind-wall returns, off-route clutter, no direct-home wall dashing |
| `arc_to_line` | `38952` | `25000` to `50000`; strong | `straight >= 0.75`, `arc <= 0.10`, `meanLineDist <= 25`, `scatter <= 0.80` | Curved pheromone route should shorten toward a straighter chord |
| `multi_path` | `44090` | `30000` to `50000`; strong | `deliveries >= 250`, `short >= 0.55`, `long <= 0.35`, `scatter <= 0.75` | Multiple paths to same food should favor the shorter/faster path |
| `loop_decay` | `-11320` | `-15000` to `0`; normal | `final mass ratio <= 0.08`, `loop swarm <= 0.35` | Closed false loop should decay, not become a magic path |
| `food_cycle` | `4162` | `0` to `8000`; passing | both piles consumed, `second deliveries >= 20`, `old swarm <= 0.03`, `phantom smell <= 0.5` | Depleted pile recovery, second food placement, no dead-source ball |
| `post_pickup` | `561` | `0` to `700`; good | enough samples/traces, `direct-home samples = 0`, `straight traces = 0`, `direct streak = 0` | Hard guard against direct-home dash immediately after pickup |
| `lost_carrier` | `-10899` | `-20000` to `5000`; passing but weak | `samples >= 1000`, `traces >= 50`, `backtrack <= 0.08`, `returned-to-food traces = 0` | Carriers with no Home trail keep searching instead of turning back to food |
| `cluster_escape` | `9694` | `0` to `12000`; good | `cluster sample <= 0.35`, `final cluster <= 0.15`, `reversal <= 0.16`, `trapped <= 0.05` | Dense wall-side clump disperses instead of ping-ponging |

| Composite | Value | PASS Criteria |
|---|---:|---|
| Total score | `-118489` | Informational only; higher is better |
| Gate | `PASS` | All benchmark hard gates pass |

## Current Defaults

```text
home_diffusion           = 0.030
food_lay_strength       = 1.5
outbound_lay_threshold  = 0.5
sensor_dist             = 24.0
bilinear_deposit        = false
pheromone_intensity_ui  = 0.30
wall_home_diffusion_cap = 0.020
wall_trail_lay_scale    = 0.8    # effective wall trail lay strength = 1.2
wall_trail_decay_cap    = 0.9975
```

Latest validated raw metrics:

```text
wall: deliveries=518 route=0.30 wallPr=6 wAim=0.00 wClr=0.00 bRet=0.31 scatter=0.67 offRoute=0.78 offTrail=0.19 wallDz=0.30 offClump=0.02 branch=828
arc: straight=0.85 arc=0.01 off=0.14 meanLineDist=9.1 scatter=0.58
multi_path: short=0.99 long=0.00 off=0.01 scatter=0.55 deliveries=730
loop_decay: final=0.000 swarm=0.28
food_cycle: first=17 second=31 old=0.00 smell=0.00
post_pickup: direct=0/33822 line=0/61 streak=0
lost_carrier: back=4597/71673 ret=0/120
cluster_escape: workers=180 disp=501.8 sample=0.08 final=0.00 rev=0.00 trap=0.00
```

## Commands

```bash
cargo test --locked -p ant-backend
cargo build --locked --release -p ant-backend
./target/release/ant-backend path_regression
./target/release/ant-backend wall_regression
./target/release/ant-backend cluster_regression
./target/release/ant-backend dump_wall_test
./target/release/ant-backend dump_arc_to_line
./target/release/ant-backend dump_arc_progress
./target/release/ant-backend dump_food_cycle
```

Dump images are written as PPM files in `/tmp/claude/`; convert them to PNG
for visual inspection when needed.

## Metric Glossary

- `wAim`, `wSt`: wall-test carriers aiming home through a blocking wall.
- `wClr`, `clearLine`: carriers after clearing the wall; catches visible
  direct-home trajectories.
- `bRet`: behind-wall pickups that return home within the trace window.
- `scatter`: workers away from nest/food that are not on Home or Food trails.
- `homeDash`: carrying ants aimed at the nest while not on Home trail. Use
  `post_pickup` as the hard no-GPS gate.
- `branch`, `dead`, `cover`: route clutter, abandoned fragments, and map-wide
  pheromone fog/webbing.
- `offRoute`: workers outside the ideal around-wall corridor; context only.
- `offTrail`: off-route workers standing on Home or Food pheromone. This is
  the hard metric for broad side-loop clutter.
- `wallDz`: wall-side or loop-basin milling. Scored and gated, but broader
  than `offTrail`.
- `offClump`: locally dense off-route worker groups; complements
  `cluster_escape`.
- `straight`, `arc`, `off`, `meanLineDist`: arc-to-line shortening metrics.
- `multi short`, `multi long`, `multi off`: duplicate-route selection metrics.
- `loop final`, `loop swarm`: false-loop chemical decay and temporary milling.
- `food_cycle old`, `food_cycle smell`: depleted-source swarm and phantom
  FoodSmell.
- `post_pickup pDash`, `pLine`, `stk`: immediate post-pickup no-beeline gates.
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
- Treating `cluster_escape` as proof that broad top-left wall clutter is fixed.
  It catches dense synthetic clumps, not broad off-route trail following.
- Dropping near-wall breadcrumbs from return memory; wall-corner breadcrumbs
  can be the only local evidence for how to get around a barrier.

Useful patterns:

- Keep no-GPS gates hard: `post_pickup`, `wAim`, `wClr`, and `clearLine`.
- Use wall-only tuning for obstacle-only issues; do not disturb open-field
  arc/multi/food-cycle behavior unless the bench proves it is necessary.
- `bRet` made behind-wall recovery measurable.
- `offRoute`, `offTrail`, `wallDz`, and `offClump` made side-loop clutter
  measurable. `offTrail` is the most useful hard signal.
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
