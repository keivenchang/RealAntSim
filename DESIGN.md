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
pub trait Brain: Send {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action>;
}
```

- One `Box<dyn Brain>` per ant, keyed by `EntityId` in a `HashMap`.
- Three implementations in v1: `WorkerBrain`, `SoldierBrain`, `QueenBrain`.
  Each is ~30 lines.
- `Action` is an enum: `SetHeading`, `Forward`, `PickupFood`, `DropFood`,
  `LayPheromone`, `Attack`, `Spawn`, `Idle`.
- `Perception` is a flat struct with all the info an ant can "see" — ready
  to be flattened into a fixed-size float vector for a neural net.

To add a new behavior: write a struct, implement `Brain`, register it in
`World::spawn_ant`. To swap rule-based for a neural net: write
`NnBrain { net: SmallNet }` that does the same.

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
pherotrail-lab/
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
