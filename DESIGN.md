# Ant Sim — Architecture

This is what's built in v1. The earlier version of this file laid out
alternatives; that's now replaced with the actual decisions.

```
┌───────────────────────────────────────────────────────────────────────┐
│  ant-backend  (one Rust binary, port 8080)                            │
│                                                                       │
│  ┌─────────────────────────┐    broadcast(String)                     │
│  │  sim task               │ ──────────┐                              │
│  │   - owns World          │           │                              │
│  │   - 30 Hz server loop   │           ▼                              │
│  │   - N sim steps/tick    │      ┌────────────┐                      │
│  │   - drains commands     │      │  WS conn 1 │ ────▶ browser        │
│  │     each iteration      │      │  WS conn 2 │ ────▶ browser        │
│  └──────▲──────────────────┘      │  ...       │                      │
│         │                          └────────────┘                     │
│         │ mpsc(Command)                                               │
│         └──────────────────────────────────────────                   │
│                                                                       │
│  axum: /         → ServeDir(frontend/)                                │
│        /ws       → snapshot stream out, command stream in             │
└───────────────────────────────────────────────────────────────────────┘
```

## Stack

- **Language:** Rust 2021.
- **HTTP/WS server:** `axum 0.7` on `tokio`.
- **Static frontend:** `tower-http::ServeDir`, served from `frontend/`.
- **Math:** `glam` for `Vec2`.
- **RNG:** `rand::rngs::SmallRng`, seeded for reproducibility.
- **Wire format:** JSON via `serde_json`.
- **Frontend:** plain HTML + Canvas 2D + vanilla JS, no build step.

No ECS, no protobuf, no React. v1 stays small on purpose; we'll layer those
in if we hit a real wall.

## Process model

**One process, three concurrent jobs running on tokio:**

1. **Sim task (single owner of `World`).** A `tokio::time::interval` fires at
   `SERVER_HZ` (30 Hz). Each tick:
   - drain pending commands from the mpsc channel,
   - run `config.speed_mult` sim steps in a tight loop (1×/10×/100×/1000×),
   - serialize a snapshot, push to the broadcast channel.
2. **WS reader (per connection).** Parses incoming JSON commands, forwards
   to the sim task via `mpsc::UnboundedSender<Command>`.
3. **WS writer (per connection).** Subscribes to the snapshot broadcast,
   pushes each new snapshot to its socket.

Both per-connection tasks live in the same `handle_socket` async function,
multiplexed with `tokio::select!`. No locks on `World` — single owner.

## Tick model & speed multiplier

- **Server loop = 30 Hz**, independent of sim speed.
- **Sim steps per loop = `speed_mult`** (clamped 1..=1000).
- Snapshots ship at 30/sec regardless. At 1000× speed, 1000 sim steps happen
  between consecutive snapshots — the world jumps further each frame, but
  bandwidth and render rate are unchanged.

This decouples "how fast we're simulating" from "how fast we render," which
is what you need for both real-time observation and fast-forwarded
emergent-behavior runs.

## Sim loop, per step

Four phases. Each phase finishes before the next starts.

1. **Perceive.** For every ant, build a `Perception` struct (self stats,
   nearby food, nearby ants, pheromone gradients, nest position, colony
   food). Immutable borrow on the world; trivially parallelizable later.
2. **Decide.** For every ant, call `brain.decide(&perception, &mut rng) -> Vec<Action>`.
   Each ant has its own `Box<dyn Brain>`. The brain is the only piece that
   defines the ant's behavior — sim code never inspects it.
3. **Apply.** Walk the action list and mutate the world: move, pickup, drop,
   lay pheromone, attack, spawn (queen only).
4. **Bookkeeping.** Pheromone decay, food respawn (if enabled), energy
   tick-down, reap dead ants.

This phase split is also what RL/NN training expects (observation → action →
environment transition), so swapping a `RuleBased` brain for an `NnBrain`
later requires zero changes to the sim.

## The `Brain` trait

```rust
pub trait Brain: Send + Sync {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action>;
}
```

- One `Box<dyn Brain>` per ant, keyed by `EntityId` in a `HashMap`.
- Current worker implementations: `WorkerBrain` (`classic`),
  `WeightedWorkerBrain` (`weighted`), and `NeuralWorkerBrain` (`neural`).
  Queen and soldier brains are separate.
- `SimConfig.worker_brain_kind` selects the worker brain. `classic` remains
  the rule-based baseline; `weighted` and `neural` are selectable comparison
  brains covered by the same benchmark suite.
- `NeuralWorkerBrain` loads `assets/neural_worker_weights.json` by default,
  or `REALANTSIM_NEURAL_WORKER_WEIGHTS` when set. If no weights are
  available it falls back to `classic`. The loaded worker net sees a
  40-float observation: 24 local sensor/context values plus 16 summaries from
  the ant's coarse map. Neural steering is gated off when those map summaries
  do not contain a meaningful cue.
- `Action` is an enum: `SetHeading`, `SetHeadingImmediate`, `Forward`,
  `PickupFood`, `DropFood`, `LayPheromone`, `Attack`, `Spawn`, `Idle`.
- `Perception` is a flat struct with all the info an ant can "see" — ready
  to be flattened into a fixed-size float vector for a neural net.
- The world owns the I/O boundary. Brains output clean intent, while `World`
  applies configurable perception and actuator imperfections before/after the
  decision. This keeps brain logic comparable without making live ants move
  like exact mathematical particles.
- Classic workers maintain a leaky 30x30 coarse world map. The map records
  visited cells, wall/repellent evidence, food/home sightings, outbound
  back-hints, and cells from successful trips.
- The classic brain runs a small Dijkstra-style search over that 30x30 map
  and blends the planned direction with pheromone sensors. Planned direct
  home returns through open space are allowed; through-wall home aiming is
  still rejected by wall benches.
- The neural worker reuses the classic brain for pickup/drop, pheromone
  safety, and base motion, then blends a learned local turn into the carrier
  return path only when the map has evidence. This keeps the neural branch
  from worsening empty-map lost-carrier cases.

## Neural Worker Today

The current neural worker is a tiny turn-prediction MLP. It is not the whole
ant brain. The explicit rule-based brain still owns pickup/drop, pheromone
laying, stale-trail repellent, wall safety, and fallback movement. The network
only predicts a bounded heading delta, and only in a narrow carrier-return
case where the map has useful evidence.

PyTorch-equivalent model:

```python
class WorkerTurnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs)).squeeze(-1) * 0.7
```

Parameter count: `13,345`.

Runtime inputs:

- `24` local sensor/context floats: carrying flag, wall-ahead flag, nest flag,
  pickup distance, left/center/right Food/Home/Repellent sensors, local
  Food/FoodSmell/Repellent values, and local pheromone gradients.
- `16` coarse-map summary floats from the ant's map.

The map is a leaky `30x30` matrix per worker ant (`900` cells). Each cell
tracks visited strength, wall evidence, repellent evidence, food/home
sightings, successful-route reinforcement, and a back-hint vector. The full
matrix is not fed into the MLP. It is compressed into these 16 features:

| Features | Meaning |
|---|---|
| `0..2` | planned direction to known food, in ant-local coordinates |
| `2..4` | planned direction home, in ant-local coordinates |
| `4..6` | local return/back-hint vector |
| `6..8` | local map avoidance vector |
| `8..10` | rough home direction |
| `10..12` | learned path-home hint |
| `12` | current cell wall evidence |
| `13` | current cell repellent evidence |
| `14` | current/success-route confidence |
| `15` | wall-scenario flag |

Neural steering is gated by:

- carrying food
- no explicit return-route breadcrumb path
- weak Home trail sensor signal
- pickup distance in the configured neural range, default `300..500`
- map cue strength above `0.05`

If any gate fails, `NeuralWorkerBrain` returns the classic action unchanged.
If the gate passes, the predicted turn is blended into the classic heading
with `REALANTSIM_NEURAL_BLEND` (default `0.55`).

Training is in `tools/train_neural_worker.py`. It uses PyTorch, trains the
same MLP shape above, uses CUDA when available, and wraps the model in
`nn.DataParallel(..., device_ids=[0, 1])` when two GPUs are visible. The Rust
runtime does not embed PyTorch; it reads exported JSON weights from
`assets/neural_worker_weights.json`.

Normalized score impact today:

| Brain | Total | Wall | Meander | Cluster | Lost Carrier | Notes |
|---|---:|---:|---:|---:|---:|---|
| `classic` | `805.9` | `80.3` | `91.4` | `100.0` | `72.9` | map-aware rule baseline; passes current 9-bench suite |
| `neural` | `809.9` | `80.3` | `91.4` | `100.0` | `76.9` | same map plus learned local turn; passes current 9-bench suite |
| `weighted` | `716.5` | `89.0` | `74.3` | `55.0` | `77.4` | weighted vector blend; lower total mostly from arc and cluster caps, still passes all gates |

The current benchmark does not isolate "map vs no map", because both `classic` and `neural` use the same `30x30` map. The isolated neural gain today is mostly `lost_carrier` (`+4.0`) and total score (`+4.0`). The map is more foundational: it enables acceptable direct open-space returns, wall-aware planning around obstacles, and the neural input features. To measure the map's standalone value, add a `classic_no_map` brain or runtime flag and include it in the normalized bench chart.

## Sim Split

Runtime flow:

```text
true world
  -> local perception with small seeded noise
  -> per-ant leaky coarse map / brain state
  -> clean brain intent
  -> actuator and deposit noise
  -> true world mutation
```

The brain does not mutate the world directly. It returns intent such as
`SetHeading`, `Forward`, `PickupFood`, `DropFood`, or `LayPheromone`. The
world validates event actions against true state, so a noisy perception can
make an ant try to pick up too early, but only actual food in range succeeds.

Normal GUI defaults keep mild imperfections active:

```text
REALANTSIM_PERCEPTION_POSITION_NOISE = 0.03
REALANTSIM_PERCEPTION_HEADING_NOISE  = 0.001
REALANTSIM_PERCEPTION_SIGNAL_NOISE   = 0.002
REALANTSIM_MOTOR_SPEED_NOISE         = 0.01
REALANTSIM_MOTOR_TURN_NOISE          = 0.001
REALANTSIM_DEPOSIT_STRENGTH_NOISE    = 0.01
REALANTSIM_DEPOSIT_POSITION_NOISE    = 0.0
```

These values are deterministic per ant/tick, so reruns are reproducible. The
scored bench harness sets all I/O noise to zero inside `apply_bench_params`;
those scores remain a stable behavior contract for the brain/pheromone
algorithm. Use the env vars above for separate noisy visual experiments.

## Related Work And Ideas

This project is not the first "neural ant" idea. Existing work clusters into
several categories:

For the current product direction, split them this way:

- Game / interactive demo: NetLogo Ants, Ant Foraging Simulation, PyNAnts,
  Firas Jaber Ant-Sim, Ant Colony Foraging & Predator Simulation, NEAT Ant
  Foraging & Predator Simulation, and Neuralant. These are most useful for UI,
  scenario selection, visual legibility, player tools, and fast feedback loops.
- Research / modeling: the SNN pheromone papers, neuroevolution environment,
  Active Inferants, stochastic two-pheromone model, attractive/repellent
  pheromone model, PDE trail model, DeepACO, ANTS, and insect route-memory
  work. These are most useful for brain alternatives, scoring metrics,
  ablations, and plausibility checks.

Because the goal is to gamify, research ideas should enter through visible
mechanics and scenarios: selectable bench starts in the GUI, readable trail
colors, player-placed food/walls/pesticide, and score/achievement hooks. Avoid
turning the main experience into a research dashboard; keep heavy metrics in
the bench and debug views.

| Project / paper | What it is about | Common with this project | Different from this project | Ideas to borrow |
|---|---|---|---|---|
| [Foraging Ants Controlled by Spiking Neural Networks and Double Pheromones](https://arxiv.org/abs/1507.08467) | Individual ants use a trained spiking neural circuit; the swarm uses attractive and negative pheromones. | Neural per-ant control, food/obstacle stimuli, positive and negative pheromone, foraging. | Their SNN is the main ant controller and learns via associative/classical-conditioning style rules. Our MLP only adjusts a gated local turn; rules still own safety and pheromone deposit. | Keep negative/no-entry pheromone as a first-class signal; consider a separate SNN brain for comparison; benchmark positive-only vs positive+negative pheromone. |
| [Emergent Communication Enhances Foraging Behaviour in Evolved Swarms Controlled by SNNs](https://arxiv.org/abs/2212.08484) | Evolves SNN-controlled ant-like agents; pheromone communication emerges rather than being hand-coded. | Colony-level foraging, neural controllers, pheromone communication, rule-based baseline comparison. | Their training evolves communication and deposit behavior. Our deposit rules are explicit and the net is behavior-cloned/teacher-guided. | Add ablations: no pheromone, no repellent, no map; eventually train a brain that decides when/where to deposit pheromone. |
| [A Simulation Environment for the Neuroevolution of Ant Colony Dynamics](https://arxiv.org/abs/2406.13147) | ALIFE environment for evolving models that reproduce ant trail dynamics from sensory data. | Same goal of emergent colony behavior from local agent observations. | Their focus is neuroevolution and real-data imitation of ant trails; ours is an interactive simulator with benchmarked behaviors and tunable brains. | Add trajectory-record/replay tools; create a "match target trail" bench from saved GUI or real-video traces. |
| [NetLogo Ants](https://ccl.northwestern.edu/netlogo/models/Ants) | Classic educational model where simple ant rules, evaporation, and diffusion produce colony-level foraging. | Same basic local-sniffing loop, pheromone reinforcement, evaporation, and sequential food exploitation. | Uses a nest-scent shortcut for returning home and a simpler single-chemical world. Our sim needs wall-aware returns, negative pheromone, maps, and visual plausibility benches. | Keep diffusion/evaporation controls visible; add food-order and critical-mass benches; use it as a minimal sanity baseline for "simple rules can work". |
| [Ant Foraging Simulation](https://sveinnthorarins.github.io/project/ant-foraging-simulation) | Browser/PixiJS visualization with food particles and two pheromone particle types for outbound and return guidance. | Similar interactive browser surface and two-direction trail idea. | Particle-based visualization and hand-coded physics, without our benchmark suite, neural brain switch, or wall-route scoring. | Borrow clear color semantics and lightweight live-demo polish; keep two trail channels understandable in the GUI. |
| [PyNAnts](https://github.com/Nikorasu/PyNAnts) | Python/Pygame pheromone trail simulator with adjustable ant counts, food placement, and pheromone surface resolution. | Same practical GUI concerns: click-to-place food, pheromone field resolution, and visible trail-following. | Its README calls out obstacle/wall avoidance while returning home as a TODO; that is exactly one of our hard requirements. | Treat wall-return as a differentiator; keep user interaction tests around food placement and drag/paint behavior. |
| [Firas Jaber Ant-Sim](https://firrj.com/projects/ant-sim/) | Go/Raylib ant colony simulator with separate scenarios for home navigation, wandering, food, pheromone pathfinding, and multi-colony runs. | Scenario-driven simulator with simple rules producing colony behavior. | More of a scenario showcase than a scored regression harness; no neural worker path documented. | Keep our benches scenario-shaped and add multi-colony scenarios later only after single-colony foraging is solid. |
| [Ant Colony Foraging & Predator Simulation](https://muuuh.com/simulations/ant-colony/) | Browser simulation with two colonies, food pheromone trails, and a predator that leaves a danger pheromone. | Same idea of multiple pheromone semantics, including a danger/avoidance field. | Includes predator/competition dynamics that are outside current scope. | Danger pheromone supports keeping `Repellent` as a general avoidance channel, not only stale-trail cleanup. |
| [NEAT Ant Foraging & Predator Simulation](https://neat-javascript.org/examples/ant-simulator.html) | Browser demo where prey/predator neural networks evolve with NEAT; prey use food, obstacle, and predator sensors. | Browser-visible ant-like agents, sensors, neural controllers, adjustable simulation. | More predator-prey/evolution demo than pheromone-route realism. No explicit wall-route/food-cycle/no-GPS benchmark suite. | Add optional sensor overlays, generation/fitness plots for neural experiments, and "play as ant" debugging mode. |
| [Active Inferants](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2021.647732/full) | Active-inference model of ant-colony foraging with local pheromone sensing in a T-maze. | Stigmergy, local pheromone access, colony-level round-trip and coherence metrics. | Active inference rather than neural nets; simpler T-maze setting and single idealized pheromone. | Add swarm-coherence and round-trip-over-time metrics; treat pheromone as a preferred local observation when designing future RL rewards. |
| [A Stochastic Model of Ant Trail Following With Two Pheromones](https://arxiv.org/abs/1508.06816) | Off-lattice stochastic model with random exploration plus pheromone-biased motion; studies diffusion regimes and environmental changes. | Stochastic motion, two pheromone signals, trail formation, adaptation to changed environments. | Mathematical/biophysical model, not a neural-controller project. | Keep stochasticity in sensors; add diffusion/evaporation parameter sweeps; benchmark dynamic wall/food changes, not only static scenarios. |
| [Attractive and Repellent Pheromones in Ant Decision Making](https://eprints.whiterose.ac.uk/id/eprint/46211/) | Agent-based Pharaoh's-ant model with attractive and repellent trail pheromones at bifurcations. | Same positive/negative trail split we use for reinforcement and discouraging failed paths. | Focused on trail-choice experiments rather than an open 2D world with obstacles and GUI. | Make repellent benches sharper: a failed branch should lose traffic, but not permanently poison a route after the world changes. |
| [A Continuous Model of Ant Foraging With Pheromones and Trail Formation](https://arxiv.org/abs/1402.5611) | PDE/chemotaxis model of ant density and pheromone fields that reproduces trail formation and food removal. | Treats trail formation as field dynamics rather than individual cleverness, which matches our pheromone-grid emphasis. | Population-density math model, not individual ants with per-ant memory and neural switches. | Add aggregate field metrics: trail width, branch entropy, dead trail mass, and food-removal rate over time. |
| [DeepACO](https://arxiv.org/abs/2309.14032) | Neural-enhanced Ant Colony Optimization for combinatorial optimization; a neural model strengthens heuristic measures for ACO. | Neural model augments an ant/pheromone-style system rather than replacing it. | It solves graph/combinatorial problems, not embodied ant simulation. | Train neural heatmaps/heuristic fields that bias exploration while pheromones remain the online memory. Useful for wall-route priors. |
| [Ant-based Neural Topology Search (ANTS)](https://repository.rit.edu/other/997/) | Uses ant-colony optimization to evolve recurrent neural-network topologies. | Shares ant roles, pheromone-style search, and neural systems vocabulary. | It is neural architecture search, not an ant colony simulator. | Naming caution: "ANTS"/"Neural Ant" language is already overloaded. Borrow role-specialized search ideas only if we evolve network topology. |
| [Neuralant StackOverFlow](https://interpret.itch.io/neuralant/) | Existing HTML5 action game/project using the "Neuralant" name. | Name/theme overlap only. | Action game, not a foraging/pheromone research simulator. | Avoid naming this project exactly `Neuralant`; it is already taken enough to cause confusion. |
| [Insect visual route-memory models](https://www.nature.com/articles/s41467-025-67545-3) | Real ant navigation work: long-term visual memories and central-complex heading control produce stable route steering from noisy cues. | Supports the idea that ants use partial route memories and rough heading representations, not only pheromones. | Biology/navigation paper, not a software ant project. | Keep the map leaky, partial, and noisy; prefer local left/right route cues over a perfect global map. |

Practical backlog from this survey:

1. Add `classic_no_map` or `map_off` to isolate how much the `30x30` matrix
   contributes to wall, lost-carrier, and food-cycle scores.
2. Add communication ablations: no pheromone, no repellent, no map, and
   neural-without-map.
3. Add a first-route convergence bench aligned with the GUI complaint:
   food stored must rise before the 8k tick mark in the wall scenario.
4. Add a target-trail replay bench: record a good GUI run or real trail trace,
   then score how closely a brain reproduces its route shape over time.
5. Add sensor/map debug overlays so a single selected ant can show its
   sensors, map cell, planned vector, and neural turn.
6. Track aggregate field metrics inspired by PDE and classic pheromone models:
   trail width, branch entropy, dead trail mass, and food-removal rate over
   time.
7. Treat `Neuralant` and plain `Neural Ant` as weak names because both collide
   with existing usage. Prefer a more specific name around trails, maps, and
   pheromones.

To add a new worker behavior: write a struct, implement `Brain`, add a
`WorkerBrainKind` variant, and register it in `brains::make_worker_brain`.
The GUI, reset path, scenario loader, and CLI benches already route through
that switch.

## Pheromones

- **6-channel scalar grid**, cell size 8u (240×135 cells for a 1920×1080
  world).
- Channels:
  - `Home` — laid by outbound workers, followed by returning workers.
  - `Food` — laid by returning (carrying) workers, followed by outbound
    workers (the trail-establishment channel, rendered yellow).
  - `Alarm` — laid by soldiers in combat, recruits more soldiers.
  - `FoodSmell` — emitted by food piles themselves (not by ants); its
    gradient points unambiguously TO a pile.
  - `Repellent` — "don't bother this way"; laid by stuck-loop escapes and
    crowd-cluster events. Outbound ants bias AWAY from it.
  - `Outbound` — visual-only direction-of-travel channel (cyan), laid by
    outbound ants on established Food trails so the human eye can see
    bidirectional flow. Doesn't influence navigation.
- **Deposit:** ants add `strength` to the cell they're standing in,
  saturated at `max_value` (50 by default).
- **Decay:** per-channel multiplier (e.g., 0.9997 Home / 0.998 Food /
  0.988 Alarm).
- **Diffusion:** per-channel 4-neighbor stencil with wall-blocking;
  ping-pong scratch buffer. Diffusion rates also per-channel
  (e.g., 0.15 Home / 0.10 Food / 0.02 Repellent).
- **Gradient sampling:** 8-neighbor difference, returns a unit vector toward
  the steepest increase (or zero if there's no nearby signal).
- **Stochastic forward sampling:** every 5th tick a brain draws random
  rays into its 180° forward cone (`ForwardSample`) and picks the
  highest-scoring direction — replaces deterministic gradient descent
  with a noisy proposal that breaks symmetric ties.

## Wire format

JSON over WebSocket. Snapshot every server-loop tick (30 Hz):

```jsonc
{
  "tick": 1234,
  "width": 800, "height": 600,
  "ants":  [{"id":1, "x":..., "y":..., "h":..., "r":"W", "c":false, "hp":1.0, "colony":0}, ...],
  "food":  [{"x":..., "y":..., "amount":...}, ...],
  "nest":  {"x":..., "y":..., "radius":..., "food_stored":..., "queen_alive":true},
  "pher_cols": 160, "pher_rows": 120,
  "pher_food":     "base64-encoded u8 × cols·rows",  // yellow trail
  "pher_home":     "base64-encoded u8 × cols·rows",  // blue trail
  "pher_smell":    "base64-encoded u8 × cols·rows",  // green food-smell
  "pher_repel":    "base64-encoded u8 × cols·rows",  // magenta repellent
  "pher_outbound": "base64-encoded u8 × cols·rows",  // cyan outbound
  "stats":  {"n_workers":..., "n_soldiers":..., "n_queens":..., "food_stored":..., "food_in_world":...},
  "config": {"speed_mult":..., "food_respawn":..., "food_respawn_interval_ticks":...}
}
```

Pheromone bytes are produced by `snapshot_downsampled`: **max** over the
source window (so thin trails survive downsample) with a **sqrt gamma** ramp
(so faint trails stay visible). At 160×120 × 2 channels × 30 Hz the
heatmap dominates bandwidth but it's still under 2 MB/s on localhost. We'll
switch to packed binary in v2 when this starts hurting.

## Command protocol

Same WebSocket, opposite direction. JSON only — they're low-rate.

```jsonc
{"op": "set_speed",            "value": 100}    // 1, 10, 100, 1000
{"op": "set_respawn",          "value": true}
{"op": "set_respawn_interval", "value": 300}    // ticks between respawns
{"op": "set_worker_brain",     "value": "weighted"} // classic|weighted|neural
{"op": "spawn_food"}                            // immediate one-shot
{"op": "reset"}                                 // wipe and re-seed
```

A `Command` enum (with `#[serde(tag = "op")]`) parses these in
`command::apply`, which mutates `world.config` (or invokes one-shot
behaviors). No restart required for any of these.

## Frontend

Two stacked canvases inside a `position: relative` div:

1. **World canvas** (800×600) — drawn each animation frame. Ground, food
   piles, nest, ants, heading ticks. Ant color is blended toward the ground
   color by `(1 - hp)` so wounded ants visibly fade.
2. **Pheromone canvas** (CSS-stretched to world size) — composites all
   five rendered channels per-pixel: yellow (Food trail), blue (Home),
   green (FoodSmell), magenta (Repellent), cyan (Outbound). A slider
   in the side panel scales global pheromone alpha 0→2× so the user
   can dim/boost the heatmap independent of sim state. Composited with
   `mix-blend-mode: screen` over the world canvas.

A right-side panel shows live stats and the control surface (speed buttons,
respawn toggle, spawn-now, reset). Controls send JSON commands; the panel
also reflects current `config` from incoming snapshots so it stays in sync
across reloads.

Total frontend: one HTML file, no framework, no build step.

## File layout

```
RealAntSim/
├── Cargo.toml                 # workspace root + release profile
├── README.md
├── DESIGN.md                  # (this file)
├── backend/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs            # axum + sim driver + ws handler
│       ├── world.rs           # World, SimConfig, tick loop, action apply
│       ├── entities.rs        # Ant, Food, Nest, Role
│       ├── brain.rs           # Brain trait, Perception, Action  ← NN swap point
│       ├── brains.rs          # WorkerBrain, SoldierBrain, QueenBrain
│       ├── pheromone.rs       # PheromoneField + downsample for snapshot
│       ├── command.rs         # WS command parsing + apply
│       └── protocol.rs        # Snapshot DTOs + serialization
└── frontend/
    └── index.html             # canvas 2d + control panel + WS client
```

## What's intentionally missing in v1

These would be straightforward to add but aren't needed yet:

- **Multiple colonies.** Single colony, single queen. The colony id is
  already a `u8` field and the soldier brain already discriminates by
  colony, so v2 is mostly "spawn a second nest."
- **ECS.** A few hundred ants fit happily in a `Vec<Ant>` + `HashMap`. ECS
  becomes worth it past ~10k ants.
- **Binary wire format.** JSON debugs nicely in DevTools. Switch when the
  heatmap bandwidth becomes annoying.
- **WebGL rendering.** Canvas 2D handles a few hundred ants at 60fps.
- **Obstacles.** Listed as a v1 element in the README but not implemented
  yet — the world is bounded with reflective walls and nothing else.

## When to upgrade what

| Symptom you're seeing | Change to make |
|---|---|
| Sim drops below 30 Hz at 1× speed | Add a uniform-grid spatial index in `world.rs` |
| Pheromone heatmap dominates bandwidth | Switch snapshot to packed binary, or downsample harder |
| 50k+ ants on screen, render lags | Replace Canvas 2D with WebGL2 instanced sprites |
| Want learned behaviors | Implement `NnBrain` that consumes `Perception` → outputs `Action`s |
| Want multiple colonies | Make `nest` a `Vec<Nest>`, give each its own pheromone channel set |
