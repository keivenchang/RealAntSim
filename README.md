# Pherotrail Lab

A real-time ant colony simulation focused on pheromone trail behavior, with a
fast native backend running the simulation and a browser-based frontend that
visualizes it live.

## What it simulates (v1: single colony)

For the first cut we run **one colony** with a fixed cast of roles, so we can
nail the sim loop, rendering, and protocol before adding the inter-colony
combat layer.

- **Queen.** One per colony. Stationary (lives in the nest). Consumes stored
  food and produces new ants over time. If the queen dies, the colony stops
  spawning — the run is effectively over.
- **Worker ants.** Wander the world looking for food, carry it back to the
  nest, and lay pheromone trails so nest-mates can follow. They don't fight.
- **Soldier ants.** Patrol around the nest and along active trails. They
  attack hostile ants on contact. In v1 there are no other colonies yet, so
  soldiers mostly idle or respond to the (currently absent) intruder
  pheromone — the role and combat code are wired up so v2 just turns it on.
- **Pheromones.** Decay and diffuse over time, so unused trails fade.
  Six channels: `Food` (yellow trail back to a pile), `Home` (blue trail
  back to the nest), `FoodSmell` (green, emitted by piles themselves),
  `Alarm` (recruits soldiers), `Repellent` (magenta, "don't bother this
  way" — laid by stuck-loop escapes and crowd clusters), and `Outbound`
  (cyan, visual-only direction-of-travel marker so bidirectional flow is
  visible).
- **Food sources.** Scattered across the world, deplete as workers harvest.
- **Obstacles.** Block movement so the colony has to find paths around them.
- **The nest.** Stockpiles food, houses the queen, and is where new ants
  spawn. We track colony stats (population by role, food in store, total
  food collected, queen health, sim tick rate) and surface them in the UI.

The point is not biological accuracy — it's an emergent-behavior toy: a few
simple per-ant rules + pheromone fields produce trail-finding, shortest-path
discovery, and recruitment dynamics that look a lot like the real thing.

## Roadmap

- **v1 (now):** one colony, queen + workers + soldiers, food, obstacles,
  pheromones. No combat yet beyond soldiers being able to attack.
- **v2:** multiple colonies, each with its own queen and pheromone channels.
  Soldiers attack non-colony ants on sight; alarm pheromone recruits more
  soldiers. Colonies fight to the death — last queen standing wins.
- **v3+:** territory, raids, food stealing, colony stats / leaderboard,
  scriptable scenarios.

## Goals

1. **Fast.** Simulate tens of thousands of ants at 60 ticks/sec on a laptop.
2. **Live.** Watch the colony in any modern browser, no install.
3. **Hackable.** Tune ant behavior, pheromone decay, food layout, world size
   from the UI without restarting the backend.
4. **Scriptable.** Headless mode for batch experiments (e.g., "how does food
   placement affect time-to-discovery?").

## Architecture, in one paragraph

A native backend runs the simulation in a tight fixed-timestep loop, decoupled
from the network. A websocket bridge streams compact binary world snapshots
(or deltas) to one or more browser clients at a configurable display rate
(e.g., 30 fps), independent of the sim tick rate. The frontend renders ants
and pheromone fields with WebGL2 instanced draws, so 50k+ entities stay
smooth. See `DESIGN.md` for the choices behind this and the alternatives we
considered.

## Repo layout (planned)

```
pherotrail-lab/
├── README.md           # this file
├── DESIGN.md           # design options + recommended choices
├── backend/            # simulation + websocket server
├── frontend/           # browser client (static assets, no build step ideal)
├── protocol/           # shared wire-format definitions
└── scripts/            # dev helpers, headless runs, benchmarks
```

## Running it

```bash
cd pherotrail-lab
cargo run --release -p ant-backend
# open http://localhost:8080  (or http://<host>:8080 from another machine)
```

Single static binary, single port. The frontend is served from `frontend/`
by the same process.

## Controls (right panel in the browser)

- **Speed: 0× / 1× / 10× / 100× / 1000×.** `0×` pauses the simulation;
  the other settings multiply sim steps per server tick. Snapshot rate stays
  at 30/sec, so the world fast-forwards without flooding the browser. Useful
  for watching trails emerge in seconds.
- **Food respawn (toggle).** When on, a new food pile drops every ~10s of
  sim time (up to 12 piles total). When off, the world drains until nothing
  is left — useful for watching the colony struggle.
- **+ spawn pile now.** One-shot food drop at a random spot away from the
  nest.
- **Reset world.** Wipes ants, food, and pheromones; keeps your current
  speed and respawn settings.

## What to look for

- **Trails forming.** When a worker first finds a food pile and starts
  returning, it lays the yellow **food trail**. Within a few seconds of sim
  time other workers latch onto it and a clear yellow highway forms between
  the pile and the nest.
- **Home trail.** Outbound workers lay the blue **home trail** so they
  (and nest-mates) can navigate back. Visible as a diffuse blue cloud
  around the nest at first, then sharper paths to recently-found piles.
- **Outbound flow.** Once a trail is established, outbound ants lay a
  cyan reinforcement on the same highway — the bidirectional flow
  (cyan outbound + yellow inbound) is what real Argentine and pharaoh
  ants do, and you can see it clearly when both directions are saturated.
- **Stuck-loop escape.** An ant that gets trapped circling a small area
  for ~90 ticks lays magenta **repellent** at the centroid, then
  commits to a heading that points away from it for ~50 ticks. The
  effect: dead-trail clearance — wasted exploration becomes a "no entry"
  sign for nestmates.
- **Health fade.** Each ant's brightness reflects its HP — a wounded ant
  visibly blends into the ground color, almost vanishing right before it
  dies.
- **Queen pressure.** The queen consumes 1 stored food per spawn. If
  workers can't keep food coming in faster than spawning consumes it, the
  colony stalls.

## Status

v1 POC. One colony, queen + workers + soldiers, pheromone trails,
obstacles, adjustable speed, optional food respawn, and headless behavior
benches. Next on the list: the v2 multi-colony combat layer (see
`DESIGN.md`).
