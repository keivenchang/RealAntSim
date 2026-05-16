# GPU Acceleration Plan — Toward 10,000× Speedup

Target: 10,000× faster sim AND 10,000× faster bench. This is a real
4-order-of-magnitude target and requires a full GPU port — partial
offloading gets at most 100×. Below: profile, options, phased plan,
realistic numbers.

## Current baseline (CPU)

Order-of-magnitude profile of one sim step (500 ants, 1920×1080 world,
PHEROMONE_CELL=8u → 240×135 = 32,400 cells × 6 channels = 194K cells):

| Phase | Per-tick cost | Why |
|---|---|---|
| Pheromone diffusion + decay | ~0.9 ms | 6 channels × 32K cells × 4-neighbor stencil + bilinear deposit |
| Forward-cone sampling | ~0.5 ms | 500 ants × 16 rays × ~30 step ray-march each = 240K wall_at + 240K cell reads |
| Perception build (spatial grid + neighbors) | ~0.3 ms | 500 ants × ~14 nearby on average |
| Brain decide | ~0.2 ms | dot products + max over 16 samples per ant |
| Apply actions (movement + repulsion + deposits) | ~0.2 ms | 500 × spatial-grid loop + bilinear splat |
| Bookkeeping (HP, deaths, corpses) | ~0.1 ms | small loops |
| **Total** | **~2.1 ms/step** | → 470 sim-steps/sec single-threaded |

Bench: 486 combos × 18,000 ticks × 2.1 ms = ~18,400 CPU-sec, divided by
32 cores via rayon par_iter ≈ 575 sec wall (matches observed ~600s).

**To get 10,000× faster** we need to convert per-step cost from ~2 ms
to ~0.2 microseconds OR run thousands of sims simultaneously. Both
strategies (Option E and Option F below) are required.

## Options surveyed

### A — wgpu (Vulkan/Metal/DX12, cross-platform)

| Pros | Cons |
|---|---|
| Works on Linux/Mac/Windows + browser | Setup verbose, type marshalling |
| Same Rust binary | WGSL is a separate language |
| Modern compute shader API | Less mature debugger than CUDA |

### B — CUDA via `cudarc`

| Pros | Cons |
|---|---|
| Best raw perf on Nvidia | Locked to Nvidia |
| Mature ecosystem | Larger binary; CUDA toolkit dep |
| Best profilers | No browser story |

### C — Pre-built CA / agent frameworks (NetLogo-GPU, Voro)

| Pros | Cons |
|---|---|
| Faster ramp-up | Abstractions don't match: multi-channel pheromone + bilinear deposit + per-ant state machine + scenario events is not a stock CA |

### D — Hybrid CPU/GPU: only pheromone field on GPU

| Pros | Cons |
|---|---|
| Smallest refactor (~3 days) | Ceiling: ~10× because forward-cone sampling reads the field every tick — round-trip CPU↔GPU per ant per tick kills it |

### E — Full GPU sim (everything on GPU)

| Pros | Cons |
|---|---|
| 500-5000× per sim | Major rewrite: Box<dyn Brain> doesn't translate (must be a single kernel with branch-on-nav_algo); pop-allocated Vecs become fixed-size GPU buffers |
| All data stays on GPU between ticks | Per-ant brain state must be flat POD |

### F — Batch many sims on one GPU (for bench)

| Pros | Cons |
|---|---|
| 100-1000× bench speedup on top of E (run hundreds of param-combos concurrently on one GPU) | Memory: 32K cells × 6 channels × 4 bytes × N combos. 100 combos = 77 MB — fits easily |

## Phased plan

Each phase ships standalone speedup; you can stop after any phase if
ROI is enough.

### Phase 1 — Profile & instrument (1 day)
- Add `puffin` or `tracing` spans around each phase
- Generate a chrome trace JSON for one bench combo
- Confirm where the time actually goes (numbers above are estimates)

### Phase 2 — Move pheromone field to GPU (3-5 days)
- WGSL compute shader for `decay_step`: one workgroup-thread per cell,
  reads neighbors from `scratch`, writes `grid`
- Walls passed as a separate texture / buffer
- Deposit kernel: ant positions in a buffer; one thread per deposit,
  bilinear splat with atomic adds
- Field lives on GPU permanently; only summary stats (or render
  snapshot) downloaded
- **Expected: 5-15× on the diffusion step, 0% on rest**

### Phase 3 — Forward-cone sampling on GPU (3-5 days)
- Ant array on GPU (Structure-of-Arrays: positions, headings, hp, etc.)
- One thread per ray (500 ants × 16 rays = 8000 threads = perfectly
  saturates a modern GPU)
- Ray-march reads from the GPU pheromone field directly (no transfer!)
- Output: 16 ForwardSample structs per ant in a GPU buffer
- **Expected: full sim phases 1+2 on GPU = 50-200× overall**

### Phase 4 — Full ant kernel (1-2 weeks)
- Brain logic in one compute kernel; the polymorphic `Box<dyn Brain>`
  becomes a switch on `nav_algo` inside the kernel
- Per-ant state (wall_follow_dir, ticks_no_progress, first_poison_tick,
  age) all moves into GPU buffers as flat POD
- Spatial hash on GPU (sort-based — Morton codes or radix sort)
- Movement + collision + deposits all run on GPU
- **Expected: 500-5000× per sim. CPU only orchestrates ticks and
  reads snapshots for the WS server.**

### Phase 5 — Concurrent bench combos on one GPU (1 week)
- Each parameter-combo is its own independent world. Pack N worlds
  into one big GPU buffer.
- Workgroup per combo; threads handle ants within a combo.
- Each tick advances ALL combos simultaneously.
- Memory budget for 1 combo ≈ 32K cells × 5 chans × 4 B + 500 ants
  × 64 B = ~700 KB. A 24 GB GPU fits 30,000+ combos in memory.
- Realistic: 100-500 combos concurrent (limited by SM occupancy)
- **Expected: 100-500× on top of Phase 4 = 50,000-2,500,000× combined
  bench speedup. Bench that took 15 min collapses to milliseconds.**

## Realistic outcomes per phase

| Stop after | Sim speedup | Bench speedup | Effort |
|---|---:|---:|---:|
| Phase 1 | 1× | 1× (informational only) | 1 day |
| Phase 2 | 1.2× | 1.5× | 3-5 days |
| Phase 3 | 50× | 50× | 6-10 days |
| Phase 4 | 500-5000× | 500-5000× | 3-5 weeks |
| Phase 5 | 500-5000× | **10,000-100,000×** | 4-6 weeks |

For the literal **10,000×** target on the bench: need Phases 4+5
combined. ~5-6 weeks of work.

For most practical purposes Phase 3 (50×) is the sweet spot — most of
the gain, ~1 week of work, sim stays understandable.

## Risks & blockers

| Risk | Mitigation |
|---|---|
| Per-ant divergent branches (nav_algo switch, brain state machines) → warp divergence kills GPU perf | Sort ants by nav_algo before kernel launch so warps are uniform |
| Bilinear pheromone deposits need atomic add — slow if many ants overlap a cell | Use shared-memory accumulation per workgroup, flush to global once |
| Spatial hash on GPU is non-trivial | Use a sort-based approach (Morton codes + radix sort); well-documented |
| Walls / wall_at queries inside ray-march | Walls as a 1-bit texture; texture sampler is free on GPU |
| WS server still wants snapshots on CPU | Use `wgpu::Buffer::map_async` once per server tick (33 ms) — negligible cost |
| Brain code complexity (many branches) doesn't fit a single kernel | Refactor brains into pure functions, accept some warp divergence |
| Rust ↔ GPU type marshalling boilerplate | `bytemuck` for POD types; `encase` for WGSL layout |

## Order-of-magnitude reality check

Modern desktop GPU (RTX 4090 / M3 Ultra):
- ~80 TFLOPS FP32
- ~10,000 concurrent threads
- ~1 TB/s memory bandwidth

Our sim per-tick work (estimated):
- 32K cells × 6 channels × 5 FLOPs/cell stencil = 960K FLOPs (diffusion)
- 500 ants × 16 rays × 30 steps × 10 FLOPs = 2.4M FLOPs (sampling)
- 500 ants × 100 FLOPs (brain + movement) = 50K FLOPs
- **Total ~3.3 MFLOPs/tick**

A 4090 with 80 TFLOPs can do 24 million such ticks/sec **in theory**.
Practically, memory access patterns, kernel launch overhead, and the
WS server constrain us. Realistic sustainable: ~1M ticks/sec.

That's **~33,000× the current 30 Hz** live rate. The 10,000× target
is achievable with margin.

## Recommendation

Do **Phase 1 (profile)** unconditionally — even if no GPU work is
ever done, the profile is useful for CPU tuning. After that:

- If you want **a real speedup but limited time**: stop at Phase 3 (~1 week, 50× sim, 50× bench).
- If you want **the 10,000× headline**: commit to Phase 4 + 5 (~5-6 weeks).
- If you want **just the bench fast** (sim stays CPU): Phase 5 alone, but it needs Phase 4 first because the sim must be a GPU kernel for Phase 5 to batch it.

## File-level notes (for whoever implements)

- `backend/src/pheromone.rs::decay_step` — already structured for GPU
  port (channels independent, scratch buffers exist, walls passed in)
- `backend/src/world.rs::sample_forward_cone` — pure function modulo
  &self; trivial to convert to a compute kernel
- `backend/src/world.rs::perceive` — gathers nearby ants from spatial
  grid; on GPU this becomes sort-then-scan
- `backend/src/brains.rs` — five algorithms, each ~50 lines of
  scoring code. Refactor to a single function selected by `nav_algo`
- The bench loop in `backend/src/main.rs::run_path_regression` is
  already rayon-parallel across combos — Phase 5 replaces the
  par_iter with a single GPU kernel launch over packed combos.

## Parallel CPU/GPU development

Feasible. Recommended workspace layout:

```
pherotrail-lab/
├── Cargo.toml          (workspace)
├── ant-core/           (shared lib: World, Brain, scenarios, entities)
├── ant-cpu/            (current binary, depends on ant-core)
└── ant-gpu/            (new binary, depends on ant-core + wgpu)
```

`ant-core` holds pure data + pure brain-logic functions. No I/O,
no `dyn Brain`, no GPU. Both backends consume it.

Workflow:
1. Day 1: extract `ant-core` from current `backend` (4-8 hr refactor,
   zero behavior change — `ant-cpu` keeps working).
2. CPU dev continues in `ant-cpu` (features, scenarios, GUI, bench
   tuning).
3. GPU dev builds `ant-gpu` Phase 2-5 in parallel. Sample test:
   run identical 10-combo bench on both backends, compare metric
   ranks; assert they match within tolerance.
4. CPU stays canonical; GPU is additive. If GPU diverges, fix GPU.

Cross-validation harness: ~1 day setup. Fixed seed, 10 combos,
runs on both, compares cost ranks. Pass = ranks identical (up to
ties) OR delta within ±5% magnitude AND sign agrees.

Risks of parallel dev:
- Drift between implementations → mitigated by cross-validation
  test that runs in CI.
- `ant-core` API churn → keep it minimal at first; expand only as
  both backends need new accessors.
- Two test suites to maintain → acceptable cost for the speedup.

Realistic timeline (one developer per side, in parallel):
- Week 0 (joint): extract `ant-core` (~1 day)
- Weeks 1-2 GPU: Phase 2+3 (diffusion + sampling kernels)
- Week 2 GPU: cross-validation harness
- Weeks 3-4 GPU: Phase 4 (full GPU sim, brain kernel, spatial hash)
- Weeks 5-6 GPU: Phase 5 (concurrent bench combos)
- CPU dev runs the whole 6 weeks unblocked.
