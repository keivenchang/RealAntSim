mod brain;
mod brains;
mod command;
mod entities;
mod pheromone;
mod protocol;
mod world;

use crate::brain::PheromoneChannel;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tower_http::services::ServeDir;
use world::World;

const FRONTEND_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../frontend");
const SERVER_HZ: u64 = 30;
const WORLD_W: f32 = 1920.0;
const WORLD_H: f32 = 1080.0;

#[derive(Clone)]
struct AppState {
    snap_tx: broadcast::Sender<String>,
    cmd_tx: mpsc::UnboundedSender<command::Command>,
}

#[tokio::main]
async fn main() {
    // CLI: `ant-backend bench` runs the headless parameter-sweep harness
    // instead of the WS server. Useful for tuning runs.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "bench" || a == "path_regression") {
        run_path_regression();
        return;
    }
    if args.iter().any(|a| a == "wall_regression") {
        run_wall_regression();
        return;
    }
    if args.iter().any(|a| a == "cluster_regression") {
        run_cluster_regression();
        return;
    }
    if args.iter().any(|a| a == "dump_wall_test") {
        dump_wall_test();
        return;
    }
    if args.iter().any(|a| a == "dump_arc_to_line") {
        dump_arc_to_line();
        return;
    }
    if args.iter().any(|a| a == "dump_arc_progress") {
        dump_arc_progress();
        return;
    }
    if args.iter().any(|a| a == "dump_food_cycle") {
        dump_food_cycle();
        return;
    }

    // Small broadcast buffer — the WS handler always drains to the latest
    // queued snapshot before sending, so we never queue old state behind a
    // slow consumer. 32 slots is more than enough for momentary stalls.
    let (snap_tx, _rx) = broadcast::channel::<String>(32);
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<command::Command>();

    // Sim task owns the world. It pulls commands and ticks at SERVER_HZ.
    // On each server tick it runs `config.speed_mult` sim steps, then ships
    // one snapshot. So bandwidth and frame rate stay constant regardless of
    // sim speed.
    let snap_tx_sim = snap_tx.clone();
    tokio::spawn(async move {
        let mut world = World::new(WORLD_W, WORLD_H);
        let period = Duration::from_millis(1000 / SERVER_HZ);
        let mut interval = tokio::time::interval(period);
        // Critical for high speed_mult: when a batch of sim steps overruns
        // 33ms, the default Burst behavior would fire the interval many
        // times in a row to "catch up", each one running another batch —
        // causing the event loop to freeze and the WS UI to hang. Skip
        // dropped ticks instead.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut pause_heartbeat = 0u32;
        loop {
            interval.tick().await;
            // Drain any incoming commands. We always process these even
            // when paused (e.g., user clicks "resume" or paints a wall).
            let mut got_cmd = false;
            while let Ok(cmd) = cmd_rx.try_recv() {
                command::apply(&mut world, cmd);
                got_cmd = true;
            }
            let ran = if !world.config.paused {
                let steps = world.config.speed_mult.max(1);
                for _ in 0..steps {
                    if !world.is_running() {
                        break;
                    }
                    world.step();
                }
                true
            } else {
                false
            };
            // Only push snapshots when state actually changed (sim ticked
            // OR a command landed), with a 1 Hz heartbeat while paused so
            // newly-connected clients still get a fresh frame.
            let should_send = if ran || got_cmd {
                pause_heartbeat = 0;
                true
            } else {
                pause_heartbeat += 1;
                if pause_heartbeat >= SERVER_HZ as u32 {
                    pause_heartbeat = 0;
                    true
                } else {
                    false
                }
            };
            if should_send {
                let json = serde_json::to_string(&protocol::snapshot(&world)).unwrap();
                let _ = snap_tx_sim.send(json);
            }
        }
    });

    let state = AppState { snap_tx, cmd_tx };
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .nest_service("/", ServeDir::new(FRONTEND_DIR))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("ant sim listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Squared distance from point `p` to segment `a→b`.
fn dist2_to_seg(p: glam::Vec2, a: glam::Vec2, b: glam::Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    (p - (a + ab * t)).length_squared()
}

#[derive(Clone, Copy, Default)]
struct BehaviorMetrics {
    trail_components: u32,
    largest_component_frac: f32,
    branch_cells: u32,
    dead_end_cells: u32,
    trail_coverage: f32,
    scatter_rate: f32,
    food_swarm_rate: f32,
    home_dash_rate: f32,
}

#[derive(Clone, Copy)]
struct BenchParams {
    name: &'static str,
    home_diffusion: f32,
    food_lay_strength: f32,
    outbound_lay_threshold: f32,
    bilinear_deposit: bool,
}

#[derive(Clone, Copy, Default)]
struct ArcLineMetrics {
    straight_ratio: f32,
    arc_ratio: f32,
    off_ratio: f32,
    mean_line_dist: f32,
}

struct WallBench {
    deliveries: u32,
    route_ratio: f32,
    wall_press: f32,
    offroute_clutter_rate: f32,
    offroute_trail_rate: f32,
    wall_dead_zone_rate: f32,
    offroute_clump_rate: f32,
    blocked_home_aim_rate: f32,
    max_blocked_home_aim_streak: u32,
    blocked_home_aim_samples: u32,
    wall_return_samples: u32,
    clear_home_direct_rate: f32,
    clear_home_direct_samples: u32,
    clear_home_samples: u32,
    max_clear_home_streak: u32,
    clear_home_trajectory_samples: u32,
    clear_home_straight_traces: u32,
    behind_wall_pickups: u32,
    behind_wall_returns: u32,
    behind_wall_return_rate: f32,
    behind_wall_timeouts: u32,
    behind_wall_wall_aim_rate: f32,
    mean_behind_wall_return_ticks: f32,
    max_behind_wall_return_ticks: u32,
    behavior: BehaviorMetrics,
    checkpoints: Vec<u32>,
    score: f32,
}

#[derive(Default)]
struct WallClutterSamples {
    workers: u32,
    offroute_workers: u32,
    offroute_trail_workers: u32,
    wall_dead_zone_workers: u32,
    offroute_clumped_workers: u32,
}

struct ArcBench {
    deliveries: u32,
    metrics: ArcLineMetrics,
    behavior: BehaviorMetrics,
    checkpoints: Vec<u32>,
    score: f32,
}

struct MultiPathBench {
    deliveries: u32,
    short_ratio: f32,
    long_ratio: f32,
    off_ratio: f32,
    behavior: BehaviorMetrics,
    checkpoints: Vec<u32>,
    score: f32,
}

struct LoopDecayBench {
    initial_food_mass: f32,
    final_food_mass: f32,
    final_mass_ratio: f32,
    loop_swarm_rate: f32,
    behavior: BehaviorMetrics,
    score: f32,
}

struct FoodCycleBench {
    first_deliveries: u32,
    second_deliveries: u32,
    first_left: f32,
    second_left: f32,
    old_swarm_after_depletion: f32,
    old_swarm_final: f32,
    phantom_smell: f32,
    repellent_at_old_source: f32,
    checkpoints: Vec<u32>,
    score: f32,
}

struct PostPickupBench {
    samples: u32,
    direct_home_samples: u32,
    direct_home_rate: f32,
    max_direct_streak: u32,
    trajectory_samples: u32,
    straight_home_traces: u32,
    straight_home_rate: f32,
    max_trace_straightness: f32,
    max_trace_home_progress: f32,
    score: f32,
}

struct LostCarrierBench {
    samples: u32,
    backtrack_samples: u32,
    backtrack_rate: f32,
    traces: u32,
    returned_to_food_traces: u32,
    max_return_drop: f32,
    score: f32,
}

struct ClusterBench {
    workers: u32,
    mean_displacement: f32,
    cluster_sample_rate: f32,
    final_cluster_rate: f32,
    reversal_rate: f32,
    trapped_oscillation_rate: f32,
    score: f32,
}

struct PostPickupTrace {
    start: glam::Vec2,
    last: glam::Vec2,
    path_len: f32,
    start_home_dist: f32,
    age: u32,
}

struct BehindWallReturnTrace {
    last: glam::Vec2,
    age: u32,
    wall_aim_ticks: u32,
}

struct LostCarrierTrace {
    last: glam::Vec2,
    path_len: f32,
    max_food_dist: f32,
    final_food_dist: f32,
    active: bool,
    completed_home: bool,
}

struct ClusterTrace {
    start: glam::Vec2,
    last: glam::Vec2,
    prev_move: glam::Vec2,
    moves: u32,
    reversals: u32,
}

struct BenchRow {
    params: BenchParams,
    wall: WallBench,
    arc: ArcBench,
    multi_path: MultiPathBench,
    loop_decay: LoopDecayBench,
    cycle: FoodCycleBench,
    post_pickup: PostPickupBench,
    lost_carrier: LostCarrierBench,
    cluster: ClusterBench,
    total: f32,
    dur: f32,
}

fn compute_behavior_metrics(
    world: &World,
    grid: u32,
    trail_threshold: f32,
    scatter_threshold: f32,
) -> BehaviorMetrics {
    let n = grid as usize;
    let cell_w = world.width / grid as f32;
    let cell_h = world.height / grid as f32;
    let mut active = vec![false; n * n];
    let mut clear_cells = 0u32;
    let mut trail_cells = 0u32;

    for y in 0..n {
        for x in 0..n {
            let p = glam::Vec2::new((x as f32 + 0.5) * cell_w, (y as f32 + 0.5) * cell_h);
            if world.wall_at(p) {
                continue;
            }
            clear_cells += 1;
            let trail = world
                .pheromones
                .sample(PheromoneChannel::Food, p)
                .max(world.pheromones.sample(PheromoneChannel::Home, p));
            if trail >= trail_threshold {
                active[y * n + x] = true;
                trail_cells += 1;
            }
        }
    }

    let mut visited = vec![false; n * n];
    let mut trail_components = 0u32;
    let mut largest_component = 0u32;
    let dirs = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)];
    for y in 0..n {
        for x in 0..n {
            let idx = y * n + x;
            if !active[idx] || visited[idx] {
                continue;
            }
            trail_components += 1;
            let mut size = 0u32;
            let mut queue = VecDeque::new();
            visited[idx] = true;
            queue.push_back((x as i32, y as i32));
            while let Some((cx, cy)) = queue.pop_front() {
                size += 1;
                for (dx, dy) in dirs {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx < 0 || ny < 0 || nx >= n as i32 || ny >= n as i32 {
                        continue;
                    }
                    let ni = ny as usize * n + nx as usize;
                    if active[ni] && !visited[ni] {
                        visited[ni] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            largest_component = largest_component.max(size);
        }
    }

    let mut branch_cells = 0u32;
    let mut dead_end_cells = 0u32;
    for y in 0..n {
        for x in 0..n {
            let idx = y * n + x;
            if !active[idx] {
                continue;
            }
            let mut degree = 0u32;
            for (dx, dy) in dirs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= n as i32 || ny >= n as i32 {
                    continue;
                }
                if active[ny as usize * n + nx as usize] {
                    degree += 1;
                }
            }
            if degree >= 3 {
                branch_cells += 1;
            } else if degree <= 1 {
                dead_end_cells += 1;
            }
        }
    }

    let food_positions: Vec<_> = world.food.iter().map(|f| f.pos).collect();
    let mut workers_considered = 0u32;
    let mut scattered_workers = 0u32;
    let mut carriers_considered = 0u32;
    let mut direct_home_dashes = 0u32;
    const ENDPOINT_R: f32 = 80.0;
    const FOOD_SWARM_R: f32 = 70.0;
    let endpoint_r2 = ENDPOINT_R * ENDPOINT_R;
    let food_swarm_r2 = FOOD_SWARM_R * FOOD_SWARM_R;
    let mut max_food_swarm = 0u32;
    let mut total_workers = 0u32;

    for food_pos in &food_positions {
        let count = world
            .ants
            .iter()
            .filter(|ant| {
                ant.role == crate::entities::Role::Worker
                    && ant.pos.distance_squared(*food_pos) <= food_swarm_r2
            })
            .count() as u32;
        max_food_swarm = max_food_swarm.max(count);
    }

    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker {
            continue;
        }
        total_workers += 1;
        let near_nest = ant.pos.distance_squared(world.nest.pos) <= endpoint_r2;
        let near_food = food_positions
            .iter()
            .any(|food_pos| ant.pos.distance_squared(*food_pos) <= endpoint_r2);
        if near_nest || near_food {
            continue;
        }
        workers_considered += 1;
        let trail_here = world
            .pheromones
            .sample(PheromoneChannel::Food, ant.pos)
            .max(world.pheromones.sample(PheromoneChannel::Home, ant.pos));
        if trail_here < scatter_threshold {
            scattered_workers += 1;
        }
        if ant.carrying_food {
            carriers_considered += 1;
            let home_here = world.pheromones.sample(PheromoneChannel::Home, ant.pos);
            let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
            let to_nest = (world.nest.pos - ant.pos).normalize_or_zero();
            if home_here < scatter_threshold && heading.dot(to_nest) > 0.92 {
                direct_home_dashes += 1;
            }
        }
    }

    BehaviorMetrics {
        trail_components,
        largest_component_frac: if trail_cells > 0 {
            largest_component as f32 / trail_cells as f32
        } else {
            0.0
        },
        branch_cells,
        dead_end_cells,
        trail_coverage: if clear_cells > 0 {
            trail_cells as f32 / clear_cells as f32
        } else {
            0.0
        },
        scatter_rate: if workers_considered > 0 {
            scattered_workers as f32 / workers_considered as f32
        } else {
            0.0
        },
        food_swarm_rate: if total_workers > 0 {
            max_food_swarm as f32 / total_workers as f32
        } else {
            0.0
        },
        home_dash_rate: if carriers_considered > 0 {
            direct_home_dashes as f32 / carriers_considered as f32
        } else {
            0.0
        },
    }
}

/// Path Regression: wall_test sweep optimized for shortest-path emergence.
///
/// Cost rewards Food pheromone INSIDE a corridor around the ideal
/// go-around polyline (nest → wall-corner → food) and PENALIZES it
/// outside. Maximizes ants forming two tight, straight-line paths.
///
/// To modify: edit the PARAMS block below, rebuild (`cargo build --release`)
/// and run `./target/release/ant-backend path_regression`.
/// Runs the wall_test scenario for N ticks then dumps the Food and Home
/// pheromone fields as a single composite PPM image.
/// Shows what the *actual* trail field looks like under the current
/// algo, so we can see if "tight single path" is emerging.
fn dump_wall_test() {
    use std::io::Write;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("wall_test");
    world.config.stable_mode = true;
    world.config.spawn_cooldown_ticks = 999_999_999;
    world.config.food_respawn = false;
    world.config.speed_mult = 1;
    // Big enough pile to sustain trail formation through the dump tick.
    for f in &mut world.food {
        f.amount = 5000.0;
    }
    const DUMP_TICKS: u32 = 16_000;
    for _ in 0..DUMP_TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
    }
    // Sample on a fine grid → write PPM.
    let w: u32 = 480;
    let h: u32 = 270;
    let cell_x = WORLD_W / w as f32;
    let cell_y = WORLD_H / h as f32;
    let mut buf: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let pp = glam::Vec2::new((x as f32 + 0.5) * cell_x, (y as f32 + 0.5) * cell_y);
            let food = world.pheromones.sample(PheromoneChannel::Food, pp);
            let home = world.pheromones.sample(PheromoneChannel::Home, pp);
            let wall = if world.wall_at(pp) { 1.0 } else { 0.0 };
            // Composite. Food = yellow (R+G), Home = blue, Wall = white.
            // Sqrt-ramp so faint trails show.
            let r = ((food / 50.0).sqrt() * 255.0 + wall * 255.0).min(255.0) as u8;
            let g = ((food / 50.0).sqrt() * 200.0 + wall * 255.0).min(255.0) as u8;
            let b = ((home / 50.0).sqrt() * 255.0 + wall * 255.0).min(255.0) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }
    // Overlay ant positions as 2×2 white dots.
    let stride_b = (w * 3) as usize;
    for ant in &world.ants {
        let cx = (ant.pos.x / cell_x) as i32;
        let cy = (ant.pos.y / cell_y) as i32;
        for dy in 0..2 {
            for dx in 0..2 {
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let i = (yy as usize) * stride_b + (xx as usize) * 3;
                buf[i] = 255;
                buf[i + 1] = 255;
                buf[i + 2] = 255;
            }
        }
    }
    let path = "/tmp/claude/wall_test_dump.ppm";
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{} {}\n255\n", w, h).unwrap();
    f.write_all(&buf).unwrap();
    println!(
        "dumped wall_test pheromone field after {} ticks → {}",
        DUMP_TICKS, path
    );
    println!(
        "deliveries={}  food_left={:.0}",
        world.food_delivered_total,
        world.food.iter().map(|f| f.amount).sum::<f32>()
    );
    let behavior = compute_behavior_metrics(&world, 192, 3.0, 0.5);
    println!(
        "topo: cmp={} branch={} dead={} scatter={:.2} swarm={:.2} homeDash={:.2} cover={:.3} largest={:.2}",
        behavior.trail_components,
        behavior.branch_cells,
        behavior.dead_end_cells,
        behavior.scatter_rate,
        behavior.food_swarm_rate,
        behavior.home_dash_rate,
        behavior.trail_coverage,
        behavior.largest_component_frac,
    );
    // ASCII heatmap of the Food channel (the trail) so we can SEE the
    // field shape without an image viewer. 96 cols × 36 rows.
    println!("\n=== FOOD TRAIL (yellow channel) ASCII heatmap ===");
    println!("(legend: ' '=0  .=1  -=3  +=8  *=15  #=25  @=40+  W=wall)");
    let aw: u32 = 96;
    let ah: u32 = 36;
    let acx = WORLD_W / aw as f32;
    let acy = WORLD_H / ah as f32;
    for y in 0..ah {
        let mut line = String::with_capacity(aw as usize + 2);
        for x in 0..aw {
            let pp = glam::Vec2::new((x as f32 + 0.5) * acx, (y as f32 + 0.5) * acy);
            if world.wall_at(pp) {
                line.push('W');
                continue;
            }
            let v = world.pheromones.sample(PheromoneChannel::Food, pp);
            let c = if v < 0.5 {
                ' '
            } else if v < 2.0 {
                '.'
            } else if v < 5.0 {
                '-'
            } else if v < 10.0 {
                '+'
            } else if v < 20.0 {
                '*'
            } else if v < 35.0 {
                '#'
            } else {
                '@'
            };
            line.push(c);
        }
        println!("{}", line);
    }
    // Also dump the nest and food pile coordinates for orientation.
    println!("\nnest@({:.0},{:.0})", world.nest.pos.x, world.nest.pos.y);
    for (i, f) in world.food.iter().enumerate() {
        println!(
            "food[{}]@({:.0},{:.0}) amount={:.0}",
            i, f.pos.x, f.pos.y, f.amount
        );
    }
}

fn dist2_to_polyline(p: glam::Vec2, pts: &[glam::Vec2]) -> f32 {
    pts.windows(2)
        .map(|w| dist2_to_seg(p, w[0], w[1]))
        .fold(f32::INFINITY, f32::min)
}

fn paint_food_polyline(world: &mut World, pts: &[glam::Vec2], samples_per_seg: u32, strength: f32) {
    for pair in pts.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let tangent = (b - a).normalize_or_zero();
        let normal = glam::Vec2::new(-tangent.y, tangent.x);
        for i in 0..samples_per_seg {
            let t = i as f32 / (samples_per_seg - 1).max(1) as f32;
            let p = a.lerp(b, t);
            for offset in [-8.0, 0.0, 8.0] {
                world
                    .pheromones
                    .deposit(PheromoneChannel::Food, p + normal * offset, strength);
            }
        }
    }
}

fn food_mass_near_polyline(
    world: &World,
    grid: u32,
    short_pts: &[glam::Vec2],
    long_pts: &[glam::Vec2],
) -> (f32, f32, f32) {
    let cell_w = WORLD_W / grid as f32;
    let cell_h = WORLD_H / grid as f32;
    let corridor_r2 = 20.0_f32.powi(2);
    let endpoint_r2 = 90.0_f32.powi(2);
    let start = short_pts[0];
    let end = *short_pts.last().unwrap();
    let mut short_mass = 0.0_f32;
    let mut long_mass = 0.0_f32;
    let mut off_mass = 0.0_f32;
    for gx in 0..grid {
        for gy in 0..grid {
            let p = glam::Vec2::new((gx as f32 + 0.5) * cell_w, (gy as f32 + 0.5) * cell_h);
            if p.distance_squared(start) <= endpoint_r2 || p.distance_squared(end) <= endpoint_r2 {
                continue;
            }
            let food = world.pheromones.sample(PheromoneChannel::Food, p);
            if food <= 0.5 {
                continue;
            }
            let d_short = dist2_to_polyline(p, short_pts);
            let d_long = dist2_to_polyline(p, long_pts);
            if d_short <= corridor_r2 && d_short <= d_long {
                short_mass += food;
            } else if d_long <= corridor_r2 {
                long_mass += food;
            } else {
                off_mass += food;
            }
        }
    }
    let denom = short_mass + long_mass + off_mass + 1.0;
    (short_mass / denom, long_mass / denom, off_mass / denom)
}

fn dump_arc_to_line() {
    use std::io::Write;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("arc_to_line");
    world.config.stable_mode = true;
    world.config.spawn_cooldown_ticks = 999_999_999;
    world.config.food_respawn = false;
    world.config.speed_mult = 1;

    let start = world.nest.pos;
    let end = world.food.first().map(|f| f.pos).unwrap_or(start);
    let mid = (start + end) * 0.5 + glam::Vec2::new(0.0, -world.height * 0.30);
    let mut arc_pts = Vec::with_capacity(65);
    for i in 0..65 {
        let t = i as f32 / 64.0;
        let u = 1.0 - t;
        arc_pts.push(start * (u * u) + mid * (2.0 * u * t) + end * (t * t));
    }

    const DUMP_TICKS: u32 = 18_000;
    for _ in 0..DUMP_TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
    }

    let w: u32 = 480;
    let h: u32 = 270;
    let cell_x = WORLD_W / w as f32;
    let cell_y = WORLD_H / h as f32;
    let straight_r2 = 18.0_f32.powi(2);
    let arc_r2 = 18.0_f32.powi(2);
    let mut straight_mass = 0.0_f32;
    let mut arc_mass = 0.0_f32;
    let mut off_mass = 0.0_f32;
    let mut weighted_dist = 0.0_f32;
    let mut total_mass = 0.0_f32;
    let mut buf: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let pp = glam::Vec2::new((x as f32 + 0.5) * cell_x, (y as f32 + 0.5) * cell_y);
            let food = world.pheromones.sample(PheromoneChannel::Food, pp);
            let home = world.pheromones.sample(PheromoneChannel::Home, pp);
            if food > 0.5 {
                total_mass += food;
                let d2_line = dist2_to_seg(pp, start, end);
                let d2_arc = dist2_to_polyline(pp, &arc_pts);
                weighted_dist += food * d2_line.sqrt();
                if d2_line <= straight_r2 {
                    straight_mass += food;
                } else if d2_arc <= arc_r2 {
                    arc_mass += food;
                } else {
                    off_mass += food;
                }
            }
            let r = ((food / 50.0).sqrt() * 255.0).min(255.0) as u8;
            let g = ((food / 50.0).sqrt() * 200.0).min(255.0) as u8;
            let b = ((home / 50.0).sqrt() * 255.0).min(255.0) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }

    let stride_b = (w * 3) as usize;
    for ant in &world.ants {
        let cx = (ant.pos.x / cell_x) as i32;
        let cy = (ant.pos.y / cell_y) as i32;
        for dy in 0..2 {
            for dx in 0..2 {
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let i = (yy as usize) * stride_b + (xx as usize) * 3;
                buf[i] = 255;
                buf[i + 1] = 255;
                buf[i + 2] = 255;
            }
        }
    }

    let path = "/tmp/claude/arc_to_line_dump.ppm";
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{} {}\n255\n", w, h).unwrap();
    f.write_all(&buf).unwrap();

    let straight_ratio = straight_mass / (straight_mass + arc_mass + off_mass + 1.0);
    let arc_ratio = arc_mass / (straight_mass + arc_mass + off_mass + 1.0);
    let off_ratio = off_mass / (straight_mass + arc_mass + off_mass + 1.0);
    let mean_line_dist = weighted_dist / total_mass.max(1.0);
    let behavior = compute_behavior_metrics(&world, 192, 3.0, 0.5);
    println!(
        "dumped arc_to_line pheromone field after {} ticks -> {}",
        DUMP_TICKS, path
    );
    println!(
        "deliveries={} food_left={:.0}",
        world.food_delivered_total,
        world.food.iter().map(|f| f.amount).sum::<f32>()
    );
    println!(
        "straight={:.2} arc={:.2} off={:.2} meanLineDist={:.1}",
        straight_ratio, arc_ratio, off_ratio, mean_line_dist
    );
    println!(
        "topo: cmp={} branch={} dead={} scatter={:.2} swarm={:.2} homeDash={:.2} cover={:.3} largest={:.2}",
        behavior.trail_components,
        behavior.branch_cells,
        behavior.dead_end_cells,
        behavior.scatter_rate,
        behavior.food_swarm_rate,
        behavior.home_dash_rate,
        behavior.trail_coverage,
        behavior.largest_component_frac,
    );

    println!("\n=== FOOD TRAIL (yellow channel) ASCII heatmap ===");
    println!("(legend: ' '=0  .=1  -=3  +=8  *=15  #=25  @=40+)");
    let aw: u32 = 96;
    let ah: u32 = 36;
    let acx = WORLD_W / aw as f32;
    let acy = WORLD_H / ah as f32;
    for y in 0..ah {
        let mut line = String::with_capacity(aw as usize);
        for x in 0..aw {
            let pp = glam::Vec2::new((x as f32 + 0.5) * acx, (y as f32 + 0.5) * acy);
            let v = world.pheromones.sample(PheromoneChannel::Food, pp);
            let c = if v < 0.5 {
                ' '
            } else if v < 2.0 {
                '.'
            } else if v < 5.0 {
                '-'
            } else if v < 10.0 {
                '+'
            } else if v < 20.0 {
                '*'
            } else if v < 35.0 {
                '#'
            } else {
                '@'
            };
            line.push(c);
        }
        println!("{}", line);
    }
    println!(
        "\nnest@({:.0},{:.0}) food@({:.0},{:.0})",
        start.x, start.y, end.x, end.y
    );
}

fn render_world_panel(world: &World, panel_w: u32, panel_h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (panel_w * panel_h * 3) as usize];
    let cell_x = WORLD_W / panel_w as f32;
    let cell_y = WORLD_H / panel_h as f32;
    for y in 0..panel_h {
        for x in 0..panel_w {
            let p = glam::Vec2::new((x as f32 + 0.5) * cell_x, (y as f32 + 0.5) * cell_y);
            let food = world.pheromones.sample(PheromoneChannel::Food, p);
            let home = world.pheromones.sample(PheromoneChannel::Home, p);
            let food_smell = world.pheromones.sample(PheromoneChannel::FoodSmell, p);
            let wall = if world.wall_at(p) { 1.0 } else { 0.0 };
            let i = ((y * panel_w + x) * 3) as usize;
            buf[i] = ((food / 50.0).sqrt() * 255.0 + wall * 255.0).min(255.0) as u8;
            buf[i + 1] =
                ((food / 50.0).sqrt() * 210.0 + (food_smell / 50.0).sqrt() * 90.0 + wall * 255.0)
                    .min(255.0) as u8;
            buf[i + 2] = ((home / 50.0).sqrt() * 255.0 + wall * 255.0).min(255.0) as u8;
        }
    }

    let start = world.nest.pos;
    if let Some(food) = world.food.first() {
        let end = food.pos;
        for i in 0..panel_w {
            let t = i as f32 / (panel_w - 1).max(1) as f32;
            let p = start.lerp(end, t);
            let x = (p.x / cell_x).round() as i32;
            let y = (p.y / cell_y).round() as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let xx = x + dx;
                    let yy = y + dy;
                    if xx < 0 || yy < 0 || xx >= panel_w as i32 || yy >= panel_h as i32 {
                        continue;
                    }
                    let bi = ((yy as u32 * panel_w + xx as u32) * 3) as usize;
                    buf[bi] = buf[bi].max(70);
                    buf[bi + 1] = buf[bi + 1].max(70);
                    buf[bi + 2] = buf[bi + 2].max(70);
                }
            }
        }
    }

    buf
}

fn dump_arc_progress() {
    use std::io::Write;
    const PANEL_W: u32 = 320;
    const PANEL_H: u32 = 180;
    const TICKS: [u32; 5] = [0, 4_500, 9_000, 13_500, 18_000];
    let params = BenchParams {
        name: "default",
        home_diffusion: 0.03,
        food_lay_strength: 1.5,
        outbound_lay_threshold: 0.5,
        bilinear_deposit: false,
    };
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("arc_to_line");
    apply_bench_params(&mut world, params);

    let mut panels: Vec<(u32, Vec<u8>, ArcLineMetrics)> = Vec::new();
    let mut last_tick = 0;
    for tick in TICKS {
        if tick > last_tick {
            run_steps(&mut world, tick - last_tick, 0);
            last_tick = tick;
        }
        panels.push((
            tick,
            render_world_panel(&world, PANEL_W, PANEL_H),
            arc_line_metrics(&world, 192),
        ));
    }

    let out_w = PANEL_W * panels.len() as u32;
    let out_h = PANEL_H;
    let mut out = vec![0u8; (out_w * out_h * 3) as usize];
    for (panel_i, (_, panel, _)) in panels.iter().enumerate() {
        let x0 = panel_i as u32 * PANEL_W;
        for y in 0..PANEL_H {
            for x in 0..PANEL_W {
                let src = ((y * PANEL_W + x) * 3) as usize;
                let dst = ((y * out_w + x0 + x) * 3) as usize;
                out[dst..dst + 3].copy_from_slice(&panel[src..src + 3]);
            }
        }
        if panel_i > 0 {
            for y in 0..PANEL_H {
                let dst = ((y * out_w + x0) * 3) as usize;
                out[dst] = 80;
                out[dst + 1] = 80;
                out[dst + 2] = 80;
            }
        }
    }

    let path = "/tmp/claude/arc_progress.ppm";
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{} {}\n255\n", out_w, out_h).unwrap();
    f.write_all(&out).unwrap();
    println!("dumped arc progress montage -> {}", path);
    for (tick, _, metrics) in panels {
        println!(
            "tick={tick:>5} straight={:.2} arc={:.2} off={:.2} meanLineDist={:.1}",
            metrics.straight_ratio, metrics.arc_ratio, metrics.off_ratio, metrics.mean_line_dist
        );
    }
}

fn dump_food_cycle() {
    use std::io::Write;
    let params = BenchParams {
        name: "dump",
        home_diffusion: 0.03,
        food_lay_strength: 1.5,
        outbound_lay_threshold: 3.0,
        bilinear_deposit: false,
    };
    let mut world = World::new(WORLD_W, WORLD_H);
    apply_bench_params(&mut world, params);
    world.config.food_respawn_amount = 45.0;

    let first_pos = world.nest.pos + glam::Vec2::new(220.0, 0.0);
    let second_pos = world.nest.pos + glam::Vec2::new(-220.0, 130.0);
    world.add_food_at(first_pos, 45.0);
    run_steps(&mut world, 6_000, 0);
    let first_left = world.food.iter().map(|f| f.amount).sum::<f32>();
    run_steps(&mut world, 1_500, 0);
    let first_deliveries = world.food_delivered_total;
    let old_swarm_after_depletion = worker_fraction_near(&world, first_pos, 60.0);
    let phantom_smell = world
        .pheromones
        .sample(PheromoneChannel::FoodSmell, first_pos);
    let before_second = world.food_delivered_total;
    world.add_food_at(second_pos, 45.0);
    run_steps(&mut world, 6_000, 0);
    let second_deliveries = world.food_delivered_total - before_second;
    let second_left = world.food.iter().map(|f| f.amount).sum::<f32>();
    let old_swarm_final = worker_fraction_near(&world, first_pos, 60.0);
    let repellent_at_old_source = world
        .pheromones
        .sample(PheromoneChannel::Repellent, first_pos);

    let w: u32 = 480;
    let h: u32 = 270;
    let cell_x = WORLD_W / w as f32;
    let cell_y = WORLD_H / h as f32;
    let mut buf: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let pp = glam::Vec2::new((x as f32 + 0.5) * cell_x, (y as f32 + 0.5) * cell_y);
            let food = world.pheromones.sample(PheromoneChannel::Food, pp);
            let home = world.pheromones.sample(PheromoneChannel::Home, pp);
            let smell = world.pheromones.sample(PheromoneChannel::FoodSmell, pp);
            let repel = world.pheromones.sample(PheromoneChannel::Repellent, pp);
            let r = ((food / 50.0).sqrt() * 255.0 + (repel / 8.0).sqrt() * 180.0).min(255.0) as u8;
            let g = ((food / 50.0).sqrt() * 200.0 + (smell / 25.0).sqrt() * 160.0).min(255.0) as u8;
            let b = ((home / 50.0).sqrt() * 255.0).min(255.0) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }

    let stride_b = (w * 3) as usize;
    for ant in &world.ants {
        let cx = (ant.pos.x / cell_x) as i32;
        let cy = (ant.pos.y / cell_y) as i32;
        for dy in 0..2 {
            for dx in 0..2 {
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let i = (yy as usize) * stride_b + (xx as usize) * 3;
                buf[i] = 255;
                buf[i + 1] = 255;
                buf[i + 2] = 255;
            }
        }
    }
    for (pos, rgb) in [(first_pos, [255, 0, 255]), (second_pos, [0, 255, 0])] {
        let cx = (pos.x / cell_x) as i32;
        let cy = (pos.y / cell_y) as i32;
        for dy in -3..=3 {
            for dx in -3..=3 {
                if dx != 0 && dy != 0 {
                    continue;
                }
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let i = (yy as usize) * stride_b + (xx as usize) * 3;
                buf[i] = rgb[0];
                buf[i + 1] = rgb[1];
                buf[i + 2] = rgb[2];
            }
        }
    }

    let path = "/tmp/claude/food_cycle_dump.ppm";
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{} {}\n255\n", w, h).unwrap();
    f.write_all(&buf).unwrap();
    println!("dumped food_cycle pheromone field -> {}", path);
    println!(
        "first_deliveries={} first_left={:.0} second_deliveries={} second_left={:.0}",
        first_deliveries, first_left, second_deliveries, second_left
    );
    println!(
        "oldSwarmAfter={:.3} oldSwarmFinal={:.3} phantomSmell={:.2} repelOld={:.2}",
        old_swarm_after_depletion, old_swarm_final, phantom_smell, repellent_at_old_source
    );
    println!(
        "old@({:.0},{:.0}) second@({:.0},{:.0})",
        first_pos.x, first_pos.y, second_pos.x, second_pos.y
    );
}

fn apply_bench_params(world: &mut World, params: BenchParams) {
    world.pheromones.diffusion[0] = params.home_diffusion;
    world.pheromones.diffusion[1] = 0.0;
    world.pheromones.decay[0] = 0.998;
    world.pheromones.decay[1] = 0.998;
    world.pheromones.diffusion[3] = 0.06;
    world.pheromones.decay[3] = 0.9994;
    world.config.stable_mode = true;
    world.config.spawn_cooldown_ticks = 999_999_999;
    world.config.food_respawn = false;
    world.config.speed_mult = 1;
    world.config.food_lay_strength = params.food_lay_strength;
    world.config.outbound_lay_threshold = params.outbound_lay_threshold;
    world.config.bilinear_deposit = params.bilinear_deposit;
}

fn run_steps(world: &mut World, ticks: u32, checkpoint_every: u32) -> Vec<u32> {
    let mut checkpoints = Vec::new();
    for tick in 0..ticks {
        if !world.is_running() {
            break;
        }
        world.step();
        if checkpoint_every > 0 && (tick + 1) % checkpoint_every == 0 {
            checkpoints.push(world.food_delivered_total);
        }
    }
    checkpoints
}

fn ray_hits_wall(world: &World, pos: glam::Vec2, dir: glam::Vec2, max_dist: f32) -> bool {
    let step = 8.0;
    let n = (max_dist / step).ceil() as u32;
    for i in 1..=n {
        let d = (i as f32 * step).min(max_dist);
        if world.wall_at(pos + dir * d) {
            return true;
        }
    }
    false
}

fn wall_blocked_home_aim_tick(world: &World) -> (u32, u32) {
    const DIRECT_DOT: f32 = 0.75;
    const LOOKAHEAD: f32 = 90.0;
    let wall_x = WORLD_W * 0.5;
    let wall_top = WORLD_H * 0.22;
    let wall_bot = WORLD_H * 0.78;
    let mut samples = 0u32;
    let mut blocked = 0u32;
    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker || !ant.carrying_food {
            continue;
        }
        if ant.pos.x <= wall_x + 20.0 || ant.pos.x >= wall_x + 180.0 {
            continue;
        }
        if ant.pos.y < wall_top - 40.0 || ant.pos.y > wall_bot + 40.0 {
            continue;
        }
        let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
        let to_nest = (world.nest.pos - ant.pos).normalize_or_zero();
        if to_nest.length_squared() <= 0.0 {
            continue;
        }
        if !ray_hits_wall(world, ant.pos, heading, LOOKAHEAD) {
            continue;
        }
        samples += 1;
        if heading.dot(to_nest) >= DIRECT_DOT {
            blocked += 1;
        }
    }
    (samples, blocked)
}

fn wall_clear_home_aim_tick(world: &World) -> (u32, u32) {
    const DIRECT_DOT: f32 = 0.92;
    let wall_x = WORLD_W * 0.5;
    let mut samples = 0u32;
    let mut direct = 0u32;
    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker || !ant.carrying_food {
            continue;
        }
        if ant.pos.x >= wall_x - 48.0 || ant.pos.x <= world.nest.pos.x + 80.0 {
            continue;
        }
        if ant.pos.distance(world.nest.pos) <= 160.0 {
            continue;
        }
        samples += 1;
        let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
        let to_nest = (world.nest.pos - ant.pos).normalize_or_zero();
        if to_nest.length_squared() > 0.0 && heading.dot(to_nest) >= DIRECT_DOT {
            direct += 1;
        }
    }
    (samples, direct)
}

fn worker_fraction_near(world: &World, pos: glam::Vec2, radius: f32) -> f32 {
    let r2 = radius * radius;
    let mut workers = 0u32;
    let mut near = 0u32;
    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker {
            continue;
        }
        workers += 1;
        if ant.pos.distance_squared(pos) <= r2 {
            near += 1;
        }
    }
    if workers == 0 {
        0.0
    } else {
        near as f32 / workers as f32
    }
}

fn wall_route_distance2(p: glam::Vec2) -> f32 {
    let nest_p = glam::Vec2::new(WORLD_W * 0.18, WORLD_H * 0.5);
    let food_p = glam::Vec2::new(WORLD_W * 0.82, WORLD_H * 0.5);
    let top_c = glam::Vec2::new(WORLD_W * 0.5, WORLD_H * 0.20);
    let bot_c = glam::Vec2::new(WORLD_W * 0.5, WORLD_H * 0.80);
    [
        (nest_p, top_c),
        (top_c, food_p),
        (nest_p, bot_c),
        (bot_c, food_p),
    ]
    .iter()
    .map(|s| dist2_to_seg(p, s.0, s.1))
    .fold(f32::INFINITY, f32::min)
}

fn sample_wall_clutter(world: &World, samples: &mut WallClutterSamples) {
    const ENDPOINT_R: f32 = 95.0;
    const ROUTE_R: f32 = 70.0;
    const TRAIL_THRESHOLD: f32 = 0.5;
    const CLUMP_R: f32 = 34.0;
    const CLUMP_MIN_NEIGHBORS: u32 = 7;

    let endpoint_r2 = ENDPOINT_R * ENDPOINT_R;
    let route_r2 = ROUTE_R * ROUTE_R;
    let clump_r2 = CLUMP_R * CLUMP_R;
    let wall_x = WORLD_W * 0.5;
    let wall_top = WORLD_H * 0.22;
    let wall_bot = WORLD_H * 0.78;
    let food_positions: Vec<_> = world.food.iter().map(|food| food.pos).collect();
    let mut offroute_positions = Vec::new();

    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker {
            continue;
        }
        if ant.pos.distance_squared(world.nest.pos) <= endpoint_r2 {
            continue;
        }
        if food_positions
            .iter()
            .any(|food_pos| ant.pos.distance_squared(*food_pos) <= endpoint_r2)
        {
            continue;
        }
        samples.workers += 1;

        let route_dist2 = wall_route_distance2(ant.pos);
        if route_dist2 <= route_r2 {
            continue;
        }
        samples.offroute_workers += 1;
        offroute_positions.push(ant.pos);

        let trail_here = world
            .pheromones
            .sample(PheromoneChannel::Food, ant.pos)
            .max(world.pheromones.sample(PheromoneChannel::Home, ant.pos));
        if trail_here >= TRAIL_THRESHOLD {
            samples.offroute_trail_workers += 1;
        }

        let central_wall_basin = ant.pos.x >= wall_x - 210.0
            && ant.pos.x <= wall_x + 210.0
            && ant.pos.y >= wall_top + 45.0
            && ant.pos.y <= wall_bot - 45.0;
        let upper_side_loop_basin = ant.pos.x >= wall_x - 360.0
            && ant.pos.x <= wall_x + 360.0
            && ant.pos.y <= wall_top + 150.0;
        let lower_side_loop_basin = ant.pos.x >= wall_x - 360.0
            && ant.pos.x <= wall_x + 360.0
            && ant.pos.y >= wall_bot - 150.0;
        if central_wall_basin || upper_side_loop_basin || lower_side_loop_basin {
            samples.wall_dead_zone_workers += 1;
        }
    }

    for (i, pos) in offroute_positions.iter().enumerate() {
        let mut neighbors = 0u32;
        for (j, other) in offroute_positions.iter().enumerate() {
            if i == j {
                continue;
            }
            if pos.distance_squared(*other) <= clump_r2 {
                neighbors += 1;
                if neighbors >= CLUMP_MIN_NEIGHBORS {
                    samples.offroute_clumped_workers += 1;
                    break;
                }
            }
        }
    }
}

fn wall_route_metrics(world: &World, grid: u32) -> (f32, f32) {
    let nest_p = glam::Vec2::new(WORLD_W * 0.18, WORLD_H * 0.5);
    let food_p = glam::Vec2::new(WORLD_W * 0.82, WORLD_H * 0.5);
    let top_c = glam::Vec2::new(WORLD_W * 0.5, WORLD_H * 0.20);
    let bot_c = glam::Vec2::new(WORLD_W * 0.5, WORLD_H * 0.80);
    let top_segs = [(nest_p, top_c), (top_c, food_p)];
    let bot_segs = [(nest_p, bot_c), (bot_c, food_p)];
    let corridor_r2 = 15.0_f32.powi(2);
    let cell_w = WORLD_W / grid as f32;
    let cell_h = WORLD_H / grid as f32;
    let mut on_route = 0.0_f32;
    let mut total = 0.0_f32;
    let mut wall_press = 0.0_f32;
    let wall_x_min = WORLD_W * 0.5 - 25.0;
    let wall_x_max = WORLD_W * 0.5 + 25.0;
    let wall_y_min = WORLD_H * 0.30;
    let wall_y_max = WORLD_H * 0.70;
    let dash_y_min = WORLD_H * 0.46;
    let dash_y_max = WORLD_H * 0.54;
    let dash_x_left_min = WORLD_W * 0.25;
    let dash_x_left_max = WORLD_W * 0.45;
    let dash_x_right_min = WORLD_W * 0.55;
    let dash_x_right_max = WORLD_W * 0.75;

    for gx in 0..grid {
        for gy in 0..grid {
            let p = glam::Vec2::new((gx as f32 + 0.5) * cell_w, (gy as f32 + 0.5) * cell_h);
            let food = world.pheromones.sample(PheromoneChannel::Food, p);
            if food <= 0.0 {
                continue;
            }
            total += food;
            let d2 = top_segs
                .iter()
                .map(|s| dist2_to_seg(p, s.0, s.1))
                .chain(bot_segs.iter().map(|s| dist2_to_seg(p, s.0, s.1)))
                .fold(f32::INFINITY, f32::min);
            if d2 <= corridor_r2 {
                on_route += food;
            }
            let at_wall =
                p.x >= wall_x_min && p.x <= wall_x_max && p.y >= wall_y_min && p.y <= wall_y_max;
            let in_dash_band = p.y >= dash_y_min
                && p.y <= dash_y_max
                && ((p.x >= dash_x_left_min && p.x <= dash_x_left_max)
                    || (p.x >= dash_x_right_min && p.x <= dash_x_right_max));
            if at_wall || in_dash_band {
                wall_press += food;
            }
        }
    }
    (on_route / (total + 1.0), wall_press)
}

fn arc_line_metrics(world: &World, grid: u32) -> ArcLineMetrics {
    let start = world.nest.pos;
    let Some(food) = world.food.first() else {
        return ArcLineMetrics::default();
    };
    let end = food.pos;
    let mid = (start + end) * 0.5 + glam::Vec2::new(0.0, -world.height * 0.30);
    let mut arc_pts = Vec::with_capacity(65);
    for i in 0..65 {
        let t = i as f32 / 64.0;
        let u = 1.0 - t;
        arc_pts.push(start * (u * u) + mid * (2.0 * u * t) + end * (t * t));
    }

    let cell_w = WORLD_W / grid as f32;
    let cell_h = WORLD_H / grid as f32;
    let straight_r2 = 18.0_f32.powi(2);
    let arc_r2 = 18.0_f32.powi(2);
    let mut straight_mass = 0.0_f32;
    let mut arc_mass = 0.0_f32;
    let mut off_mass = 0.0_f32;
    let mut weighted_dist = 0.0_f32;
    let mut total_mass = 0.0_f32;
    for gx in 0..grid {
        for gy in 0..grid {
            let p = glam::Vec2::new((gx as f32 + 0.5) * cell_w, (gy as f32 + 0.5) * cell_h);
            let food = world.pheromones.sample(PheromoneChannel::Food, p);
            if food <= 0.5 {
                continue;
            }
            total_mass += food;
            let d2_line = dist2_to_seg(p, start, end);
            weighted_dist += food * d2_line.sqrt();
            if d2_line <= straight_r2 {
                straight_mass += food;
            } else if dist2_to_polyline(p, &arc_pts) <= arc_r2 {
                arc_mass += food;
            } else {
                off_mass += food;
            }
        }
    }
    let denom = straight_mass + arc_mass + off_mass + 1.0;
    ArcLineMetrics {
        straight_ratio: straight_mass / denom,
        arc_ratio: arc_mass / denom,
        off_ratio: off_mass / denom,
        mean_line_dist: weighted_dist / total_mass.max(1.0),
    }
}

fn run_wall_bench(params: BenchParams) -> WallBench {
    const TICKS: u32 = 16_000;
    const GRID: u32 = 192;
    const WALL_TRACE_TICKS: u32 = 120;
    const WALL_TRACE_MIN_PATH: f32 = 45.0;
    const WALL_TRACE_STRAIGHTNESS_LIMIT: f32 = 0.90;
    const WALL_TRACE_HOME_PROGRESS_LIMIT: f32 = 0.70;
    const BEHIND_WALL_TRACE_TICKS: u32 = 3_600;
    const CLUTTER_SAMPLE_START: u32 = 8_000;
    const CLUTTER_SAMPLE_EVERY: u32 = 200;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("wall_test");
    for food in &mut world.food {
        food.amount = 5000.0;
    }
    apply_bench_params(&mut world, params);
    let mut checkpoints = Vec::new();
    let mut wall_return_samples = 0u32;
    let mut blocked_home_aim_samples = 0u32;
    let mut current_blocked_streak = 0u32;
    let mut max_blocked_home_aim_streak = 0u32;
    let mut clear_home_samples = 0u32;
    let mut clear_home_direct_samples = 0u32;
    let mut current_clear_home_streak = 0u32;
    let mut max_clear_home_streak = 0u32;
    let mut clear_traces: HashMap<crate::entities::EntityId, PostPickupTrace> = HashMap::new();
    let mut previous_clear_carriers: HashSet<crate::entities::EntityId> = HashSet::new();
    let mut clear_home_trajectory_samples = 0u32;
    let mut clear_home_straight_traces = 0u32;
    let mut behind_wall_traces: HashMap<crate::entities::EntityId, BehindWallReturnTrace> =
        HashMap::new();
    let mut behind_wall_pickups = 0u32;
    let mut behind_wall_returns = 0u32;
    let mut behind_wall_timeouts = 0u32;
    let mut behind_wall_return_ticks_sum = 0u64;
    let mut max_behind_wall_return_ticks = 0u32;
    let mut behind_wall_wall_aim_ticks = 0u32;
    let mut behind_wall_trace_ticks = 0u32;
    let mut clutter_samples = WallClutterSamples::default();
    let wall_x = WORLD_W * 0.5;
    let wall_top = WORLD_H * 0.22;
    let wall_bot = WORLD_H * 0.78;
    let nest_pos = world.nest.pos;
    let finish_clear_trace = |trace: &PostPickupTrace| -> Option<bool> {
        if trace.path_len < WALL_TRACE_MIN_PATH {
            return None;
        }
        let displacement = trace.start.distance(trace.last);
        let straightness = displacement / trace.path_len.max(1.0);
        let home_progress =
            (trace.start_home_dist - trace.last.distance(nest_pos)) / trace.path_len.max(1.0);
        Some(
            straightness >= WALL_TRACE_STRAIGHTNESS_LIMIT
                && home_progress >= WALL_TRACE_HOME_PROGRESS_LIMIT,
        )
    };
    for tick in 0..TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
        if tick + 1 >= CLUTTER_SAMPLE_START && (tick + 1) % CLUTTER_SAMPLE_EVERY == 0 {
            sample_wall_clutter(&world, &mut clutter_samples);
        }
        for ant in &world.ants {
            if ant.role != crate::entities::Role::Worker || !ant.carrying_food {
                continue;
            }
            if tick + BEHIND_WALL_TRACE_TICKS >= TICKS
                || ant.since_state_change > 1
                || ant.pos.x <= wall_x + 80.0
                || behind_wall_traces.contains_key(&ant.id)
            {
                continue;
            }
            behind_wall_pickups += 1;
            behind_wall_traces.insert(
                ant.id,
                BehindWallReturnTrace {
                    last: ant.pos,
                    age: 0,
                    wall_aim_ticks: 0,
                },
            );
        }
        let mut finished_behind_wall = Vec::new();
        for (id, trace) in behind_wall_traces.iter_mut() {
            let Some(ant) = world
                .ants
                .iter()
                .find(|ant| ant.id == *id && ant.role == crate::entities::Role::Worker)
            else {
                finished_behind_wall.push((*id, false));
                continue;
            };
            if !ant.carrying_food {
                finished_behind_wall.push((*id, ant.pos.distance(nest_pos) <= 80.0));
                continue;
            }
            trace.age += 1;
            trace.last = ant.pos;
            behind_wall_trace_ticks += 1;
            let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
            let to_nest = (nest_pos - ant.pos).normalize_or_zero();
            let near_blocking_wall = ant.pos.x > wall_x + 20.0
                && ant.pos.x < wall_x + 180.0
                && ant.pos.y >= wall_top - 40.0
                && ant.pos.y <= wall_bot + 40.0;
            if near_blocking_wall
                && to_nest.length_squared() > 0.0
                && heading.dot(to_nest) >= 0.75
                && ray_hits_wall(&world, ant.pos, heading, 90.0)
            {
                trace.wall_aim_ticks += 1;
                behind_wall_wall_aim_ticks += 1;
            }
            if trace.age >= BEHIND_WALL_TRACE_TICKS {
                finished_behind_wall.push((*id, false));
            }
        }
        for (id, returned_home) in finished_behind_wall {
            if let Some(trace) = behind_wall_traces.remove(&id) {
                if returned_home {
                    behind_wall_returns += 1;
                    behind_wall_return_ticks_sum += trace.age as u64;
                    max_behind_wall_return_ticks = max_behind_wall_return_ticks.max(trace.age);
                } else if trace.age >= BEHIND_WALL_TRACE_TICKS {
                    behind_wall_timeouts += 1;
                }
            }
        }
        let (samples, blocked) = wall_blocked_home_aim_tick(&world);
        wall_return_samples += samples;
        blocked_home_aim_samples += blocked;
        let tick_rate = if samples > 0 {
            blocked as f32 / samples as f32
        } else {
            0.0
        };
        if samples >= 5 && tick_rate > 0.15 {
            current_blocked_streak += 1;
            max_blocked_home_aim_streak = max_blocked_home_aim_streak.max(current_blocked_streak);
        } else {
            current_blocked_streak = 0;
        }
        let (clear_samples, clear_direct) = wall_clear_home_aim_tick(&world);
        clear_home_samples += clear_samples;
        clear_home_direct_samples += clear_direct;
        let clear_tick_rate = if clear_samples > 0 {
            clear_direct as f32 / clear_samples as f32
        } else {
            0.0
        };
        if clear_samples >= 5 && clear_tick_rate > 0.05 {
            current_clear_home_streak += 1;
            max_clear_home_streak = max_clear_home_streak.max(current_clear_home_streak);
        } else {
            current_clear_home_streak = 0;
        }
        let clear_carriers = world
            .ants
            .iter()
            .filter(|ant| {
                ant.role == crate::entities::Role::Worker
                    && ant.carrying_food
                    && ant.pos.x < wall_x - 48.0
                    && ant.pos.x > nest_pos.x + 80.0
                    && ant.pos.distance(nest_pos) > 160.0
            })
            .map(|ant| ant.id)
            .collect::<HashSet<_>>();
        for ant in &world.ants {
            if !clear_carriers.contains(&ant.id) || previous_clear_carriers.contains(&ant.id) {
                continue;
            }
            clear_traces.insert(
                ant.id,
                PostPickupTrace {
                    start: ant.pos,
                    last: ant.pos,
                    path_len: 0.0,
                    start_home_dist: ant.pos.distance(nest_pos),
                    age: 0,
                },
            );
        }
        let mut finished_clear = Vec::new();
        for (id, trace) in clear_traces.iter_mut() {
            let Some(ant) = world
                .ants
                .iter()
                .find(|ant| ant.id == *id && ant.role == crate::entities::Role::Worker)
            else {
                finished_clear.push(*id);
                continue;
            };
            if !clear_carriers.contains(id) {
                finished_clear.push(*id);
                continue;
            }
            trace.age += 1;
            trace.path_len += ant.pos.distance(trace.last);
            trace.last = ant.pos;
            if trace.age >= WALL_TRACE_TICKS || ant.pos.distance(nest_pos) <= 80.0 {
                finished_clear.push(*id);
            }
        }
        for id in finished_clear {
            if let Some(trace) = clear_traces.remove(&id) {
                if let Some(straight_home) = finish_clear_trace(&trace) {
                    clear_home_trajectory_samples += 1;
                    if straight_home {
                        clear_home_straight_traces += 1;
                    }
                }
            }
        }
        previous_clear_carriers = clear_carriers;
        if (tick + 1) % 4_000 == 0 {
            checkpoints.push(world.food_delivered_total);
        }
    }
    for trace in clear_traces.values() {
        if let Some(straight_home) = finish_clear_trace(trace) {
            clear_home_trajectory_samples += 1;
            if straight_home {
                clear_home_straight_traces += 1;
            }
        }
    }
    let behavior = compute_behavior_metrics(&world, GRID, 3.0, 0.5);
    let (route_ratio, wall_press) = wall_route_metrics(&world, GRID);
    let blocked_home_aim_rate = if wall_return_samples > 0 {
        blocked_home_aim_samples as f32 / wall_return_samples as f32
    } else {
        1.0
    };
    let clear_home_direct_rate = if clear_home_samples > 0 {
        clear_home_direct_samples as f32 / clear_home_samples as f32
    } else {
        1.0
    };
    let behind_wall_return_rate = if behind_wall_pickups > 0 {
        behind_wall_returns as f32 / behind_wall_pickups as f32
    } else {
        0.0
    };
    let behind_wall_wall_aim_rate = if behind_wall_trace_ticks > 0 {
        behind_wall_wall_aim_ticks as f32 / behind_wall_trace_ticks as f32
    } else {
        0.0
    };
    let offroute_clutter_rate = if clutter_samples.workers > 0 {
        clutter_samples.offroute_workers as f32 / clutter_samples.workers as f32
    } else {
        0.0
    };
    let offroute_trail_rate = if clutter_samples.workers > 0 {
        clutter_samples.offroute_trail_workers as f32 / clutter_samples.workers as f32
    } else {
        0.0
    };
    let wall_dead_zone_rate = if clutter_samples.workers > 0 {
        clutter_samples.wall_dead_zone_workers as f32 / clutter_samples.workers as f32
    } else {
        0.0
    };
    let offroute_clump_rate = if clutter_samples.workers > 0 {
        clutter_samples.offroute_clumped_workers as f32 / clutter_samples.workers as f32
    } else {
        0.0
    };
    let mean_behind_wall_return_ticks = if behind_wall_returns > 0 {
        behind_wall_return_ticks_sum as f32 / behind_wall_returns as f32
    } else {
        0.0
    };
    let score = world.food_delivered_total as f32 * 80.0
        + route_ratio * 20_000.0
        + behind_wall_return_rate * 30_000.0
        + behavior.largest_component_frac * 2_000.0
        - wall_press * 200.0
        - blocked_home_aim_rate * 150_000.0
        - max_blocked_home_aim_streak as f32 * 500.0
        - clear_home_direct_rate * 150_000.0
        - max_clear_home_streak as f32 * 500.0
        - behind_wall_wall_aim_rate * 120_000.0
        - behind_wall_timeouts as f32 * 300.0
        - offroute_clutter_rate * 40_000.0
        - offroute_trail_rate * 100_000.0
        - wall_dead_zone_rate * 120_000.0
        - offroute_clump_rate * 100_000.0
        - behavior.home_dash_rate * 100_000.0
        - behavior.branch_cells as f32 * 35.0
        - behavior.dead_end_cells as f32 * 15.0
        - behavior.trail_coverage * 50_000.0
        - behavior.scatter_rate * 8_000.0;
    WallBench {
        deliveries: world.food_delivered_total,
        route_ratio,
        wall_press,
        offroute_clutter_rate,
        offroute_trail_rate,
        wall_dead_zone_rate,
        offroute_clump_rate,
        blocked_home_aim_rate,
        max_blocked_home_aim_streak,
        blocked_home_aim_samples,
        wall_return_samples,
        clear_home_direct_rate,
        clear_home_direct_samples,
        clear_home_samples,
        max_clear_home_streak,
        clear_home_trajectory_samples,
        clear_home_straight_traces,
        behind_wall_pickups,
        behind_wall_returns,
        behind_wall_return_rate,
        behind_wall_timeouts,
        behind_wall_wall_aim_rate,
        mean_behind_wall_return_ticks,
        max_behind_wall_return_ticks,
        behavior,
        checkpoints,
        score,
    }
}

fn run_arc_bench(params: BenchParams) -> ArcBench {
    const TICKS: u32 = 18_000;
    const GRID: u32 = 192;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("arc_to_line");
    apply_bench_params(&mut world, params);
    let checkpoints = run_steps(&mut world, TICKS, 4_500);
    let metrics = arc_line_metrics(&world, GRID);
    let behavior = compute_behavior_metrics(&world, GRID, 3.0, 0.5);
    let score = world.food_delivered_total as f32 * 35.0 + metrics.straight_ratio * 40_000.0
        - metrics.arc_ratio * 15_000.0
        - metrics.off_ratio * 18_000.0
        - metrics.mean_line_dist * 45.0
        - behavior.home_dash_rate * 60_000.0
        - behavior.branch_cells as f32 * 20.0
        - behavior.dead_end_cells as f32 * 10.0
        - behavior.trail_coverage * 35_000.0;
    ArcBench {
        deliveries: world.food_delivered_total,
        metrics,
        behavior,
        checkpoints,
        score,
    }
}

fn run_multi_path_bench(params: BenchParams) -> MultiPathBench {
    const TICKS: u32 = 12_000;
    const GRID: u32 = 192;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("arc_to_line");
    world.pheromones = crate::pheromone::PheromoneField::new(WORLD_W, WORLD_H, 8.0);
    for food in &mut world.food {
        food.amount = 5000.0;
    }
    apply_bench_params(&mut world, params);

    let start = world.nest.pos;
    let end = world.food.first().unwrap().pos;
    let short_path = [start, end];
    let long_path = [
        start,
        (start + end) * 0.5 + glam::Vec2::new(0.0, world.height * 0.32),
        end,
    ];
    paint_food_polyline(&mut world, &short_path, 700, 8.0);
    paint_food_polyline(&mut world, &long_path, 500, 8.0);

    let checkpoints = run_steps(&mut world, TICKS, 3_000);
    let (short_ratio, long_ratio, off_ratio) =
        food_mass_near_polyline(&world, GRID, &short_path, &long_path);
    let behavior = compute_behavior_metrics(&world, GRID, 3.0, 0.5);
    let score = world.food_delivered_total as f32 * 35.0 + short_ratio * 45_000.0
        - long_ratio * 25_000.0
        - off_ratio * 18_000.0
        - behavior.scatter_rate * 8_000.0
        - behavior.branch_cells as f32 * 15.0;

    MultiPathBench {
        deliveries: world.food_delivered_total,
        short_ratio,
        long_ratio,
        off_ratio,
        behavior,
        checkpoints,
        score,
    }
}

fn run_loop_decay_bench(params: BenchParams) -> LoopDecayBench {
    const TICKS: u32 = 4_000;
    const GRID: u32 = 192;
    let mut world = World::new(WORLD_W, WORLD_H);
    world.load_scenario("fresh");
    apply_bench_params(&mut world, params);
    world.food.clear();
    world.pheromones = crate::pheromone::PheromoneField::new(WORLD_W, WORLD_H, 8.0);
    apply_bench_params(&mut world, params);

    let center = glam::Vec2::new(WORLD_W * 0.68, WORLD_H * 0.5);
    let radius = 170.0;
    let samples = 360;
    let mut initial_food_mass = 0.0_f32;
    for i in 0..samples {
        let t = i as f32 / samples as f32 * std::f32::consts::TAU;
        let p = center + glam::Vec2::new(t.cos(), t.sin()) * radius;
        for offset in [-8.0, 0.0, 8.0] {
            let q = center + (p - center).normalize_or_zero() * (radius + offset);
            world.pheromones.deposit(PheromoneChannel::Food, q, 10.0);
            initial_food_mass += 10.0;
        }
    }

    run_steps(&mut world, TICKS, 0);

    let cell_w = WORLD_W / GRID as f32;
    let cell_h = WORLD_H / GRID as f32;
    let mut final_food_mass = 0.0_f32;
    let mut loop_workers = 0u32;
    let mut total_workers = 0u32;
    for gx in 0..GRID {
        for gy in 0..GRID {
            let p = glam::Vec2::new((gx as f32 + 0.5) * cell_w, (gy as f32 + 0.5) * cell_h);
            let d = p.distance(center);
            if (d - radius).abs() <= 24.0 {
                final_food_mass += world.pheromones.sample(PheromoneChannel::Food, p);
            }
        }
    }
    for ant in &world.ants {
        if ant.role != crate::entities::Role::Worker {
            continue;
        }
        total_workers += 1;
        let d = ant.pos.distance(center);
        if (d - radius).abs() <= 36.0 {
            loop_workers += 1;
        }
    }
    let final_mass_ratio = final_food_mass / initial_food_mass.max(1.0);
    let loop_swarm_rate = if total_workers > 0 {
        loop_workers as f32 / total_workers as f32
    } else {
        0.0
    };
    let behavior = compute_behavior_metrics(&world, GRID, 3.0, 0.5);
    let score = -(final_mass_ratio * 40_000.0)
        - loop_swarm_rate * 40_000.0
        - behavior.home_dash_rate * 40_000.0;

    LoopDecayBench {
        initial_food_mass,
        final_food_mass,
        final_mass_ratio,
        loop_swarm_rate,
        behavior,
        score,
    }
}

fn run_food_cycle_bench(params: BenchParams) -> FoodCycleBench {
    const FIRST_TICKS: u32 = 6_000;
    const DEAD_TICKS: u32 = 1_500;
    const SECOND_TICKS: u32 = 6_000;
    let mut world = World::new(WORLD_W, WORLD_H);
    apply_bench_params(&mut world, params);
    world.config.food_respawn = false;
    world.config.food_respawn_amount = 45.0;

    let first_pos = world.nest.pos + glam::Vec2::new(220.0, 0.0);
    let second_pos = world.nest.pos + glam::Vec2::new(-220.0, 130.0);
    world.add_food_at(first_pos, 45.0);

    let mut checkpoints = run_steps(&mut world, FIRST_TICKS, 2_000);
    let first_left = world.food.iter().map(|f| f.amount).sum::<f32>();

    run_steps(&mut world, DEAD_TICKS, 0);
    let first_deliveries = world.food_delivered_total;
    let old_swarm_after_depletion = worker_fraction_near(&world, first_pos, 60.0);
    let phantom_smell = world
        .pheromones
        .sample(PheromoneChannel::FoodSmell, first_pos);
    let repellent_at_old_source = world
        .pheromones
        .sample(PheromoneChannel::Repellent, first_pos);

    let before_second = world.food_delivered_total;
    world.add_food_at(second_pos, 45.0);
    checkpoints.extend(run_steps(&mut world, SECOND_TICKS, 2_000));
    let second_deliveries = world.food_delivered_total - before_second;
    let second_left = world.food.iter().map(|f| f.amount).sum::<f32>();
    let old_swarm_final = worker_fraction_near(&world, first_pos, 60.0);

    let score = first_deliveries as f32 * 70.0 + second_deliveries as f32 * 100.0
        - first_left * 200.0
        - second_left * 80.0
        - old_swarm_after_depletion * 50_000.0
        - old_swarm_final * 30_000.0
        - phantom_smell * 1_500.0;

    FoodCycleBench {
        first_deliveries,
        second_deliveries,
        first_left,
        second_left,
        old_swarm_after_depletion,
        old_swarm_final,
        phantom_smell,
        repellent_at_old_source,
        checkpoints,
        score,
    }
}

fn run_post_pickup_bench(params: BenchParams) -> PostPickupBench {
    const TICKS: u32 = 3_000;
    const RECENT_PICKUP_TICKS: u32 = 900;
    const TRACE_TICKS: u32 = 240;
    const MIN_HOME_DIST: f32 = 80.0;
    const DIRECT_DOT: f32 = 0.92;
    const TRACE_MIN_PATH: f32 = 35.0;
    const TRACE_STRAIGHTNESS_LIMIT: f32 = 0.90;
    const TRACE_HOME_PROGRESS_LIMIT: f32 = 0.70;

    let mut world = World::new(WORLD_W, WORLD_H);
    apply_bench_params(&mut world, params);
    world.config.food_respawn = false;
    world.config.food_respawn_amount = 120.0;
    let food_pos = world.nest.pos + glam::Vec2::new(360.0, 0.0);
    world.add_food_at(food_pos, 120.0);

    let mut samples = 0u32;
    let mut direct_home_samples = 0u32;
    let mut current_direct_streak = 0u32;
    let mut max_direct_streak = 0u32;
    let mut traces: HashMap<crate::entities::EntityId, PostPickupTrace> = HashMap::new();
    let mut previous_carriers: HashSet<crate::entities::EntityId> = HashSet::new();
    let mut trajectory_samples = 0u32;
    let mut straight_home_traces = 0u32;
    let mut max_trace_straightness = 0.0_f32;
    let mut max_trace_home_progress = 0.0_f32;
    let nest_pos = world.nest.pos;

    let finish_trace = |trace: &PostPickupTrace| -> Option<(bool, f32, f32)> {
        if trace.path_len < TRACE_MIN_PATH {
            return None;
        }
        let displacement = trace.start.distance(trace.last);
        let straightness = displacement / trace.path_len.max(1.0);
        let home_progress =
            (trace.start_home_dist - trace.last.distance(nest_pos)) / trace.path_len.max(1.0);
        let straight_home =
            straightness >= TRACE_STRAIGHTNESS_LIMIT && home_progress >= TRACE_HOME_PROGRESS_LIMIT;
        Some((straight_home, straightness, home_progress))
    };

    for _ in 0..TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
        let current_carriers = world
            .ants
            .iter()
            .filter(|ant| ant.role == crate::entities::Role::Worker && ant.carrying_food)
            .map(|ant| ant.id)
            .collect::<HashSet<_>>();

        let mut tick_samples = 0u32;
        let mut tick_direct = 0u32;
        for ant in &world.ants {
            if ant.role != crate::entities::Role::Worker {
                continue;
            }
            if ant.carrying_food && !previous_carriers.contains(&ant.id) {
                traces.insert(
                    ant.id,
                    PostPickupTrace {
                        start: ant.pos,
                        last: ant.pos,
                        path_len: 0.0,
                        start_home_dist: ant.pos.distance(world.nest.pos),
                        age: 0,
                    },
                );
            }
            if !ant.carrying_food
                || ant.since_state_change > RECENT_PICKUP_TICKS
                || ant.pos.distance(world.nest.pos) <= MIN_HOME_DIST
            {
                continue;
            }
            samples += 1;
            tick_samples += 1;
            let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
            let to_nest = (world.nest.pos - ant.pos).normalize_or_zero();
            if heading.dot(to_nest) >= DIRECT_DOT {
                direct_home_samples += 1;
                tick_direct += 1;
            }
        }
        let tick_direct_rate = if tick_samples > 0 {
            tick_direct as f32 / tick_samples as f32
        } else {
            0.0
        };
        if tick_samples >= 5 && tick_direct_rate > 0.08 {
            current_direct_streak += 1;
            max_direct_streak = max_direct_streak.max(current_direct_streak);
        } else {
            current_direct_streak = 0;
        }

        let mut finished = Vec::new();
        for (id, trace) in traces.iter_mut() {
            let Some(ant) = world
                .ants
                .iter()
                .find(|ant| ant.id == *id && ant.role == crate::entities::Role::Worker)
            else {
                finished.push(*id);
                continue;
            };
            if !ant.carrying_food {
                finished.push(*id);
                continue;
            }
            trace.age += 1;
            trace.path_len += ant.pos.distance(trace.last);
            trace.last = ant.pos;
            if trace.age >= TRACE_TICKS || ant.pos.distance(world.nest.pos) <= MIN_HOME_DIST {
                finished.push(*id);
            }
        }
        for id in finished {
            if let Some(trace) = traces.remove(&id) {
                if let Some((straight_home, straightness, home_progress)) = finish_trace(&trace) {
                    trajectory_samples += 1;
                    if straight_home {
                        straight_home_traces += 1;
                    }
                    max_trace_straightness = max_trace_straightness.max(straightness);
                    max_trace_home_progress = max_trace_home_progress.max(home_progress);
                }
            }
        }
        previous_carriers = current_carriers;
    }

    for trace in traces.values() {
        if let Some((straight_home, straightness, home_progress)) = finish_trace(trace) {
            trajectory_samples += 1;
            if straight_home {
                straight_home_traces += 1;
            }
            max_trace_straightness = max_trace_straightness.max(straightness);
            max_trace_home_progress = max_trace_home_progress.max(home_progress);
        }
    }

    let direct_home_rate = if samples > 0 {
        direct_home_samples as f32 / samples as f32
    } else {
        1.0
    };
    let straight_home_rate = if trajectory_samples > 0 {
        straight_home_traces as f32 / trajectory_samples as f32
    } else {
        1.0
    };
    let score = -(direct_home_rate * 100_000.0)
        - straight_home_rate * 150_000.0
        - max_direct_streak as f32 * 1_000.0
        + samples.min(500) as f32
        + trajectory_samples.min(100) as f32;
    PostPickupBench {
        samples,
        direct_home_samples,
        direct_home_rate,
        max_direct_streak,
        trajectory_samples,
        straight_home_traces,
        straight_home_rate,
        max_trace_straightness,
        max_trace_home_progress,
        score,
    }
}

fn run_lost_carrier_bench(params: BenchParams) -> LostCarrierBench {
    const TICKS: u32 = 700;
    const CARRIERS: u32 = 120;
    const SAMPLE_AFTER_TICKS: u32 = 30;
    const SAMPLE_MIN_FOOD_DIST: f32 = 40.0;
    const BACKTRACK_DOT: f32 = 0.45;
    const MIN_LEFT_SOURCE_DIST: f32 = 120.0;
    const RETURN_NEAR_SOURCE_DIST: f32 = 90.0;

    let mut world = World::new(WORLD_W, WORLD_H);
    apply_bench_params(&mut world, params);
    world.config.food_respawn = false;
    world.nest.pos = glam::Vec2::new(WORLD_W * 0.18, WORLD_H * 0.5);
    world.nest.food_stored = 0.0;
    world.food_delivered_total = 0;
    world.food.clear();
    world.ants.clear();
    world.brains.clear();
    world.clear_walls();

    let queen_id = world.spawn_ant(world.nest.pos, crate::entities::Role::Queen, 0);
    world.nest.queen_id = Some(queen_id);
    let food_pos = world.nest.pos + glam::Vec2::new(360.0, 0.0);
    world.add_food_at(food_pos, 5000.0);

    let mut traces: HashMap<crate::entities::EntityId, LostCarrierTrace> = HashMap::new();
    for i in 0..CARRIERS {
        let a = i as f32 * 2.399_963_1;
        let r = 7.0 + (i % 8) as f32 * 2.0;
        let start = food_pos + glam::Vec2::new(a.cos(), a.sin()) * r;
        let id = world.spawn_ant(start, crate::entities::Role::Worker, 0);
        if let Some(ant) = world.ants.iter_mut().find(|ant| ant.id == id) {
            ant.carrying_food = true;
            ant.pickup_home_dist = start.distance(world.nest.pos);
            ant.heading = std::f32::consts::PI;
            ant.target_heading = ant.heading;
            ant.since_state_change = 0;
            ant.breadcrumbs.clear();
            ant.return_path.clear();
            traces.insert(
                id,
                LostCarrierTrace {
                    last: ant.pos,
                    path_len: 0.0,
                    max_food_dist: ant.pos.distance(food_pos),
                    final_food_dist: ant.pos.distance(food_pos),
                    active: true,
                    completed_home: false,
                },
            );
        }
    }

    let mut samples = 0u32;
    let mut backtrack_samples = 0u32;
    for tick in 0..TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
        for ant in &world.ants {
            let Some(trace) = traces.get_mut(&ant.id) else {
                continue;
            };
            if !trace.active {
                continue;
            }
            let previous_food_dist = trace.final_food_dist;
            trace.path_len += ant.pos.distance(trace.last);
            trace.last = ant.pos;
            trace.final_food_dist = ant.pos.distance(food_pos);
            trace.max_food_dist = trace.max_food_dist.max(trace.final_food_dist);

            if !ant.carrying_food {
                trace.active = false;
                trace.completed_home = true;
                continue;
            }
            if tick < SAMPLE_AFTER_TICKS {
                continue;
            }
            if ant.pos.distance(world.nest.pos) <= 120.0 {
                continue;
            }
            if trace.final_food_dist >= ant.pos.distance(world.nest.pos) {
                continue;
            }
            if trace.final_food_dist <= SAMPLE_MIN_FOOD_DIST {
                continue;
            }
            let to_food = (food_pos - ant.pos).normalize_or_zero();
            if to_food.length_squared() <= 0.0 {
                continue;
            }
            samples += 1;
            let heading = glam::Vec2::new(ant.heading.cos(), ant.heading.sin());
            if trace.final_food_dist < previous_food_dist - 0.2
                && heading.dot(to_food) >= BACKTRACK_DOT
            {
                backtrack_samples += 1;
            }
        }
    }

    let mut returned_to_food_traces = 0u32;
    let mut max_return_drop = 0.0_f32;
    let mut trace_count = 0u32;
    for trace in traces.values() {
        if trace.path_len < MIN_LEFT_SOURCE_DIST {
            continue;
        }
        trace_count += 1;
        let return_drop = (trace.max_food_dist - trace.final_food_dist).max(0.0);
        max_return_drop = max_return_drop.max(return_drop);
        if !trace.completed_home
            && trace.max_food_dist >= MIN_LEFT_SOURCE_DIST
            && trace.final_food_dist <= RETURN_NEAR_SOURCE_DIST
        {
            returned_to_food_traces += 1;
        }
    }

    let backtrack_rate = if samples > 0 {
        backtrack_samples as f32 / samples as f32
    } else {
        1.0
    };
    let score = samples.min(2_000) as f32
        - backtrack_rate * 120_000.0
        - returned_to_food_traces as f32 * 5_000.0
        - max_return_drop * 30.0;

    LostCarrierBench {
        samples,
        backtrack_samples,
        backtrack_rate,
        traces: trace_count,
        returned_to_food_traces,
        max_return_drop,
        score,
    }
}

fn run_cluster_bench(params: BenchParams) -> ClusterBench {
    const TICKS: u32 = 2_400;
    const WORKERS: u32 = 180;
    const START_RADIUS: f32 = 6.0;
    const CLUSTER_RADIUS: f32 = 24.0;
    const CLUSTER_NEIGHBORS: u32 = 6;
    const SAMPLE_EVERY: u32 = 20;
    const MOVE_FLOOR2: f32 = 0.16;
    const REVERSAL_DOT: f32 = -0.65;
    const TRAPPED_DISPLACEMENT: f32 = 48.0;
    const TRAPPED_REVERSAL_RATE: f32 = 0.18;

    let mut world = World::new(WORLD_W, WORLD_H);
    apply_bench_params(&mut world, params);
    world.config.food_respawn = false;
    world.nest.pos = glam::Vec2::new(WORLD_W * 0.12, WORLD_H * 0.5);
    world.nest.food_stored = 0.0;
    world.food_delivered_total = 0;
    world.food.clear();
    world.corpses.clear();
    world.ants.clear();
    world.brains.clear();
    world.clear_walls();
    world.pheromones = crate::pheromone::PheromoneField::new(WORLD_W, WORLD_H, 8.0);
    apply_bench_params(&mut world, params);

    let queen_id = world.spawn_ant(world.nest.pos, crate::entities::Role::Queen, 0);
    world.nest.queen_id = Some(queen_id);
    let wall_x = WORLD_W * 0.50;
    let mut yy = WORLD_H * 0.34;
    while yy <= WORLD_H * 0.66 {
        world.paint_walls(wall_x, yy, 8.0, true);
        yy += 6.0;
    }
    let center = glam::Vec2::new(wall_x + 20.0, WORLD_H * 0.5);
    let mut traces: HashMap<crate::entities::EntityId, ClusterTrace> = HashMap::new();
    for i in 0..WORKERS {
        let a = i as f32 * 2.399_963_1;
        let r = START_RADIUS * ((i % 19) as f32 / 18.0).sqrt();
        let start = center + glam::Vec2::new(a.cos(), a.sin()) * r;
        let id = world.spawn_ant(start, crate::entities::Role::Worker, 0);
        if let Some(ant) = world.ants.iter_mut().find(|ant| ant.id == id) {
            ant.heading = a;
            ant.target_heading = a;
            ant.breadcrumbs.clear();
            ant.return_path.clear();
        }
        traces.insert(
            id,
            ClusterTrace {
                start,
                last: start,
                prev_move: glam::Vec2::ZERO,
                moves: 0,
                reversals: 0,
            },
        );
    }

    let mut cluster_samples = 0u32;
    let mut clustered_samples = 0u32;
    let mut final_cluster_rate = 1.0_f32;
    for tick in 0..TICKS {
        if !world.is_running() {
            break;
        }
        world.step();
        for ant in &world.ants {
            if ant.role != crate::entities::Role::Worker {
                continue;
            }
            let Some(trace) = traces.get_mut(&ant.id) else {
                continue;
            };
            let mv = ant.pos - trace.last;
            if mv.length_squared() >= MOVE_FLOOR2 {
                if trace.prev_move.length_squared() >= MOVE_FLOOR2 {
                    let dot = trace.prev_move.normalize().dot(mv.normalize());
                    if dot <= REVERSAL_DOT {
                        trace.reversals += 1;
                    }
                }
                trace.prev_move = mv;
                trace.moves += 1;
            }
            trace.last = ant.pos;
        }
        if (tick + 1) % SAMPLE_EVERY == 0 {
            let (clustered, total) =
                clustered_worker_count(&world, CLUSTER_RADIUS, CLUSTER_NEIGHBORS);
            clustered_samples += clustered;
            cluster_samples += total;
            final_cluster_rate = if total > 0 {
                clustered as f32 / total as f32
            } else {
                1.0
            };
        }
    }

    let workers = traces.len() as u32;
    let mut displacement_sum = 0.0_f32;
    let mut move_samples = 0u32;
    let mut reversal_samples = 0u32;
    let mut trapped = 0u32;
    for trace in traces.values() {
        let displacement = trace.last.distance(trace.start);
        displacement_sum += displacement;
        move_samples += trace.moves;
        reversal_samples += trace.reversals;
        let reversal_rate = if trace.moves > 0 {
            trace.reversals as f32 / trace.moves as f32
        } else {
            1.0
        };
        if displacement <= TRAPPED_DISPLACEMENT && reversal_rate >= TRAPPED_REVERSAL_RATE {
            trapped += 1;
        }
    }
    let mean_displacement = displacement_sum / workers.max(1) as f32;
    let cluster_sample_rate = if cluster_samples > 0 {
        clustered_samples as f32 / cluster_samples as f32
    } else {
        1.0
    };
    let reversal_rate = if move_samples > 0 {
        reversal_samples as f32 / move_samples as f32
    } else {
        1.0
    };
    let trapped_oscillation_rate = trapped as f32 / workers.max(1) as f32;
    let score = mean_displacement * 20.0
        - cluster_sample_rate * 4_000.0
        - final_cluster_rate * 6_000.0
        - reversal_rate * 8_000.0
        - trapped_oscillation_rate * 10_000.0;

    ClusterBench {
        workers,
        mean_displacement,
        cluster_sample_rate,
        final_cluster_rate,
        reversal_rate,
        trapped_oscillation_rate,
        score,
    }
}

fn clustered_worker_count(world: &World, radius: f32, min_neighbors: u32) -> (u32, u32) {
    let r2 = radius * radius;
    let workers = world
        .ants
        .iter()
        .filter(|ant| ant.role == crate::entities::Role::Worker)
        .collect::<Vec<_>>();
    let mut clustered = 0u32;
    for (i, ant) in workers.iter().enumerate() {
        let mut neighbors = 0u32;
        for (j, other) in workers.iter().enumerate() {
            if i == j {
                continue;
            }
            if ant.pos.distance_squared(other.pos) <= r2 {
                neighbors += 1;
                if neighbors >= min_neighbors {
                    clustered += 1;
                    break;
                }
            }
        }
    }
    (clustered, workers.len() as u32)
}

fn acceptance_violations(row: &BenchRow) -> Vec<&'static str> {
    let mut violations = Vec::new();
    if row.post_pickup.samples < 20 {
        violations.push("post_pickup produced too few carrier samples");
    }
    if row.post_pickup.direct_home_rate > 0.005 {
        violations.push("post_pickup direct-home dash rate is too high");
    }
    if row.post_pickup.max_direct_streak > 3 {
        violations.push("post_pickup has a sustained direct-home streak");
    }
    if row.post_pickup.trajectory_samples < 10 {
        violations.push("post_pickup produced too few trajectory samples");
    }
    if row.post_pickup.straight_home_traces > 0 {
        violations.push("post_pickup has straight homebound pickup trajectories");
    }
    if row.lost_carrier.samples < 1_000 || row.lost_carrier.traces < 50 {
        violations.push("lost_carrier produced too few lost-carrier samples");
    }
    if row.lost_carrier.backtrack_rate > 0.08 {
        violations.push("lost_carrier carriers turn back toward food too often");
    }
    if row.lost_carrier.returned_to_food_traces > 0 {
        violations.push("lost_carrier carriers returned toward the food source");
    }
    if row.wall.deliveries < 250 {
        violations.push("wall_test produced too few deliveries");
    }
    if row.wall.route_ratio < 0.03 {
        violations.push("wall_test did not form enough around-wall route");
    }
    if row.wall.wall_press > 250.0 {
        violations.push("wall_test has too much wall or straight-through pressure");
    }
    if row.wall.blocked_home_aim_rate > 0.05 {
        violations.push("wall_test carriers aim home through a blocking wall too often");
    }
    if row.wall.max_blocked_home_aim_streak > 8 {
        violations.push("wall_test has a sustained blocked-wall home-aim streak");
    }
    if row.wall.clear_home_direct_rate > 0.01 {
        violations.push("wall_test carriers beeline home after clearing the wall");
    }
    if row.wall.max_clear_home_streak > 3 {
        violations.push("wall_test has a sustained clear-line home beeline");
    }
    if row.wall.clear_home_straight_traces > 0 {
        violations.push("wall_test has straight homebound clear-wall trajectories");
    }
    if row.wall.behind_wall_pickups < 20 {
        violations.push("wall_test produced too few behind-wall pickup traces");
    }
    if row.wall.behind_wall_return_rate < 0.20 {
        violations.push("wall_test behind-wall carriers do not find home often enough");
    }
    if row.wall.behind_wall_returns > 0 && row.wall.mean_behind_wall_return_ticks > 3_000.0 {
        violations.push("wall_test behind-wall carriers return home too slowly");
    }
    if row.wall.behind_wall_wall_aim_rate > 0.03 {
        violations.push("wall_test behind-wall carriers aim into the wall too often");
    }
    if row.wall.behavior.scatter_rate > 0.85 {
        violations.push("wall_test workers do not converge to trails enough");
    }
    if row.wall.offroute_trail_rate > 0.20 {
        violations.push("wall_test has too much off-route trail-following clutter");
    }
    if row.wall.wall_dead_zone_rate > 0.35 {
        violations.push("wall_test has too many workers milling in wall side/dead zones");
    }
    if row.wall.offroute_clump_rate > 0.08 {
        violations.push("wall_test has too much off-route clumped milling");
    }
    if row.cycle.first_left > 0.0 || row.cycle.second_left > 0.0 {
        violations.push("food_cycle did not consume both piles");
    }
    if row.cycle.second_deliveries < 20 {
        violations.push("food_cycle second pile did not produce enough deliveries");
    }
    if row
        .cycle
        .old_swarm_after_depletion
        .max(row.cycle.old_swarm_final)
        > 0.03
    {
        violations.push("food_cycle old-source swarm is too high");
    }
    if row.cycle.phantom_smell > 0.5 {
        violations.push("food_cycle phantom FoodSmell remained after depletion");
    }
    if row.arc.metrics.straight_ratio < 0.75 {
        violations.push("arc_to_line did not collapse enough toward the straight chord");
    }
    if row.arc.metrics.arc_ratio > 0.10 {
        violations.push("arc_to_line retained too much curved bootstrap trail");
    }
    if row.arc.metrics.mean_line_dist > 25.0 {
        violations.push("arc_to_line mean distance from chord is too high");
    }
    if row.arc.behavior.scatter_rate > 0.80 {
        violations.push("arc_to_line workers do not converge to the shortened path enough");
    }
    if row.multi_path.deliveries < 250 {
        violations.push("multi_path produced too few deliveries");
    }
    if row.multi_path.short_ratio < 0.55 {
        violations.push("multi_path did not favor the shorter path enough");
    }
    if row.multi_path.long_ratio > 0.35 {
        violations.push("multi_path retained too much longer-path pheromone");
    }
    if row.multi_path.behavior.scatter_rate > 0.75 {
        violations.push("multi_path workers do not converge to active paths enough");
    }
    if row.loop_decay.final_mass_ratio > 0.08 {
        violations.push("loop_decay retained too much closed-loop Food pheromone");
    }
    if row.loop_decay.loop_swarm_rate > 0.35 {
        violations.push("loop_decay trapped an unrealistic number of ants on a closed loop");
    }
    if row.cluster.cluster_sample_rate > 0.35 {
        violations.push("cluster_escape kept too many workers in dense clusters");
    }
    if row.cluster.final_cluster_rate > 0.15 {
        violations.push("cluster_escape still has a dense final cluster");
    }
    if row.cluster.reversal_rate > 0.16 {
        violations.push("cluster_escape has too much back-and-forth motion");
    }
    if row.cluster.trapped_oscillation_rate > 0.05 {
        violations.push("cluster_escape leaves workers oscillating in place");
    }
    violations
}

fn run_path_regression() {
    println!("=== Ant Behavior Bench Suite ===");
    println!("Useful benches kept:");
    println!(
        "  wall_test    : obstacle routing, off-route clutter, behind-wall returns, no hidden home-vector dash"
    );
    println!("  arc_to_line  : path-shortening from a curved bootstrap trail toward the chord");
    println!("  multi_path   : duplicate routes to one food source favor shorter/faster path");
    println!(
        "  loop_decay   : closed Food loop without food decays instead of becoming a magic path"
    );
    println!("  food_cycle   : depleted pile recovery, second food placement, no dead-source ball");
    println!("  post_pickup  : hard gate against direct-home dash immediately after pickup");
    println!(
        "  lost_carrier : no-Home-signal carriers keep searching instead of returning to food"
    );
    println!("  cluster_escape: dense worker clump disperses instead of ping-ponging in place");
    println!("Removed from the main score:");
    println!("  density      : too indirect; visual overlap belongs in targeted UI/perf checks");
    println!("  pesticide    : real feature, but unrelated to trail-route tuning");
    println!("  generic dissipation: replaced by the depleted-food food_cycle scenario\n");

    let candidates = [
        BenchParams {
            name: "default",
            home_diffusion: 0.03,
            food_lay_strength: 1.5,
            outbound_lay_threshold: 0.5,
            bilinear_deposit: false,
        },
        BenchParams {
            name: "higher-threshold",
            home_diffusion: 0.03,
            food_lay_strength: 1.5,
            outbound_lay_threshold: 3.0,
            bilinear_deposit: false,
        },
        BenchParams {
            name: "stronger-lay",
            home_diffusion: 0.03,
            food_lay_strength: 3.0,
            outbound_lay_threshold: 0.5,
            bilinear_deposit: false,
        },
        BenchParams {
            name: "bilinear",
            home_diffusion: 0.03,
            food_lay_strength: 1.5,
            outbound_lay_threshold: 3.0,
            bilinear_deposit: true,
        },
        BenchParams {
            name: "no-home-diff",
            home_diffusion: 0.0,
            food_lay_strength: 1.5,
            outbound_lay_threshold: 0.5,
            bilinear_deposit: false,
        },
    ];

    use rayon::prelude::*;
    let total_t0 = std::time::Instant::now();
    let mut rows: Vec<BenchRow> = candidates
        .par_iter()
        .map(|params| {
            let t0 = std::time::Instant::now();
            let wall = run_wall_bench(*params);
            let arc = run_arc_bench(*params);
            let multi_path = run_multi_path_bench(*params);
            let loop_decay = run_loop_decay_bench(*params);
            let cycle = run_food_cycle_bench(*params);
            let post_pickup = run_post_pickup_bench(*params);
            let lost_carrier = run_lost_carrier_bench(*params);
            let cluster = run_cluster_bench(*params);
            BenchRow {
                params: *params,
                total: wall.score
                    + arc.score
                    + multi_path.score
                    + loop_decay.score
                    + cycle.score
                    + post_pickup.score
                    + lost_carrier.score
                    + cluster.score,
                wall,
                arc,
                multi_path,
                loop_decay,
                cycle,
                post_pickup,
                lost_carrier,
                cluster,
                dur: t0.elapsed().as_secs_f32(),
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        let a_pass = acceptance_violations(a).is_empty();
        let b_pass = acceptance_violations(b).is_empty();
        b_pass
            .cmp(&a_pass)
            .then_with(|| b.total.partial_cmp(&a.total).unwrap())
    });

    println!(
        "{:<16} {:>9} | {:>4} {:>5} {:>7} {:>6} {:>3} {:>5} {:>5} {:>5} {:>6} {:>6} | {:>4} {:>5} {:>5} {:>5} {:>6} | {:>3} {:>3} {:>5} {:>5} | {:>5} {:>5} {:>5} {:>3} | {:>5} {:>7}",
        "candidate",
        "total",
        "wDel",
        "route",
        "wallPr",
        "wAim",
        "wSt",
        "wClr",
        "bRet",
        "scat",
        "hDash",
        "br",
        "aDel",
        "line",
        "arc",
        "off",
        "dist",
        "c1",
        "c2",
        "old",
        "smell",
        "pN",
        "pDash",
        "pLine",
        "stk",
        "gate",
        "sec"
    );
    println!("{}", "-".repeat(184));
    for row in &rows {
        let violations = acceptance_violations(row);
        let gate = if violations.is_empty() {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<16} {:>9.0} | {:>4} {:>5.2} {:>7.0} {:>6.2} {:>3} {:>5.2} {:>5.2} {:>5.2} {:>6.2} {:>6} | {:>4} {:>5.2} {:>5.2} {:>5.2} {:>6.1} | {:>3} {:>3} {:>5.2} {:>5.2} | {:>5} {:>5.2} {:>5.2} {:>3} | {:>5} {:>7.1}",
            row.params.name,
            row.total,
            row.wall.deliveries,
            row.wall.route_ratio,
            row.wall.wall_press,
            row.wall.blocked_home_aim_rate,
            row.wall.max_blocked_home_aim_streak,
            row.wall.clear_home_direct_rate,
            row.wall.behind_wall_return_rate,
            row.wall.behavior.scatter_rate,
            row.wall.behavior.home_dash_rate,
            row.wall.behavior.branch_cells,
            row.arc.deliveries,
            row.arc.metrics.straight_ratio,
            row.arc.metrics.arc_ratio,
            row.arc.metrics.off_ratio,
            row.arc.metrics.mean_line_dist,
            row.cycle.first_deliveries,
            row.cycle.second_deliveries,
            row.cycle.old_swarm_after_depletion.max(row.cycle.old_swarm_final),
            row.cycle.phantom_smell,
            row.post_pickup.samples,
            row.post_pickup.direct_home_rate,
            row.post_pickup.straight_home_rate,
            row.post_pickup.max_direct_streak,
            gate,
            row.dur,
        );
        println!(
            "    scores wall={:.0} arc={:.0} multi={:.0} loop={:.0} cycle={:.0} postPickup={:.0} lostCarrier={:.0} cluster={:.0}",
            row.wall.score,
            row.arc.score,
            row.multi_path.score,
            row.loop_decay.score,
            row.cycle.score,
            row.post_pickup.score,
            row.lost_carrier.score,
            row.cluster.score,
        );
        println!(
            "    wall checkpoints={} blockedAim={}/{} clearHome={}/{} clearSt={} clearLine={}/{} behindWall={}/{} mean={} max={} timeout={} wallAim={:.3} offRoute={:.2} offTrail={:.2} wallDz={:.2} offClump={:.2}  arc checkpoints={}  multi short={:.2} long={:.2} off={:.2} scat={:.2} del={} checkpoints={}  loop mass={:.1}/{:.1} final={:.3} swarm={:.2} scat={:.2}  cycle checkpoints={}  cycle left={:.0}/{:.0} repelOld={:.1}  postPickup direct={}/{} line={}/{} maxStraight={:.2} maxHomeProg={:.2}  lostCarrier back={}/{} ret={}/{} maxDrop={:.1}  cluster workers={} disp={:.1} sample={:.2} final={:.2} rev={:.2} trap={:.2}  arcTopo: branch={} scatter={:.2} hDash={:.2} cover={:.3}  params: hd={:.3} lay={:.1} obt={:.1} bilin={}",
            row.wall
                .checkpoints
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("->"),
            row.wall.blocked_home_aim_samples,
            row.wall.wall_return_samples,
            row.wall.clear_home_direct_samples,
            row.wall.clear_home_samples,
            row.wall.max_clear_home_streak,
            row.wall.clear_home_straight_traces,
            row.wall.clear_home_trajectory_samples,
            row.wall.behind_wall_returns,
            row.wall.behind_wall_pickups,
            row.wall.mean_behind_wall_return_ticks.round() as u32,
            row.wall.max_behind_wall_return_ticks,
            row.wall.behind_wall_timeouts,
            row.wall.behind_wall_wall_aim_rate,
            row.wall.offroute_clutter_rate,
            row.wall.offroute_trail_rate,
            row.wall.wall_dead_zone_rate,
            row.wall.offroute_clump_rate,
            row.arc
                .checkpoints
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("->"),
            row.multi_path.short_ratio,
            row.multi_path.long_ratio,
            row.multi_path.off_ratio,
            row.multi_path.behavior.scatter_rate,
            row.multi_path.deliveries,
            row.multi_path
                .checkpoints
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("->"),
            row.loop_decay.final_food_mass,
            row.loop_decay.initial_food_mass,
            row.loop_decay.final_mass_ratio,
            row.loop_decay.loop_swarm_rate,
            row.loop_decay.behavior.scatter_rate,
            row.cycle
                .checkpoints
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("->"),
            row.cycle.first_left,
            row.cycle.second_left,
            row.cycle.repellent_at_old_source,
            row.post_pickup.direct_home_samples,
            row.post_pickup.samples,
            row.post_pickup.straight_home_traces,
            row.post_pickup.trajectory_samples,
            row.post_pickup.max_trace_straightness,
            row.post_pickup.max_trace_home_progress,
            row.lost_carrier.backtrack_samples,
            row.lost_carrier.samples,
            row.lost_carrier.returned_to_food_traces,
            row.lost_carrier.traces,
            row.lost_carrier.max_return_drop,
            row.cluster.workers,
            row.cluster.mean_displacement,
            row.cluster.cluster_sample_rate,
            row.cluster.final_cluster_rate,
            row.cluster.reversal_rate,
            row.cluster.trapped_oscillation_rate,
            row.arc.behavior.branch_cells,
            row.arc.behavior.scatter_rate,
            row.arc.behavior.home_dash_rate,
            row.arc.behavior.trail_coverage,
            row.params.home_diffusion,
            row.params.food_lay_strength,
            row.params.outbound_lay_threshold,
            row.params.bilinear_deposit,
        );
        if !violations.is_empty() {
            println!("    gate failures: {}", violations.join("; "));
        }
    }

    if let Some(best) = rows.first() {
        println!(
            "\nWinner: {}  total={:.0}  hd={:.3} lay={:.1} obt={:.1} bilin={}",
            best.params.name,
            best.total,
            best.params.home_diffusion,
            best.params.food_lay_strength,
            best.params.outbound_lay_threshold,
            best.params.bilinear_deposit,
        );
    }
    let default_row = rows
        .iter()
        .find(|row| row.params.name == "default")
        .expect("default bench row missing");
    let default_violations = acceptance_violations(default_row);
    if default_violations.is_empty() {
        println!("Acceptance gates: PASS for default");
    } else {
        eprintln!(
            "Acceptance gates: FAIL for default: {}",
            default_violations.join("; ")
        );
        std::process::exit(1);
    }
    println!(
        "Total: {} candidates, {:.1}s wall time",
        rows.len(),
        total_t0.elapsed().as_secs_f32()
    );
}

fn run_wall_regression() {
    let params = BenchParams {
        name: "default",
        home_diffusion: 0.03,
        food_lay_strength: 1.5,
        outbound_lay_threshold: 0.5,
        bilinear_deposit: false,
    };
    let t0 = std::time::Instant::now();
    let wall = run_wall_bench(params);
    println!(
        "wall_test score={:.0} deliveries={} route={:.2} wallPr={:.0} bRet={:.2} timeouts={} scatter={:.2} offRoute={:.2} offTrail={:.2} wallDz={:.2} offClump={:.2} branch={} dead={} cover={:.3} largest={:.2} wAim={:.3} wClr={:.3} wallAim={:.3} elapsed={:.1}s",
        wall.score,
        wall.deliveries,
        wall.route_ratio,
        wall.wall_press,
        wall.behind_wall_return_rate,
        wall.behind_wall_timeouts,
        wall.behavior.scatter_rate,
        wall.offroute_clutter_rate,
        wall.offroute_trail_rate,
        wall.wall_dead_zone_rate,
        wall.offroute_clump_rate,
        wall.behavior.branch_cells,
        wall.behavior.dead_end_cells,
        wall.behavior.trail_coverage,
        wall.behavior.largest_component_frac,
        wall.blocked_home_aim_rate,
        wall.clear_home_direct_rate,
        wall.behind_wall_wall_aim_rate,
        t0.elapsed().as_secs_f32(),
    );
    println!(
        "checkpoints={} blockedAim={}/{} clearHome={}/{} clearLine={}/{} behindWall={}/{} mean={} max={}",
        wall.checkpoints
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("->"),
        wall.blocked_home_aim_samples,
        wall.wall_return_samples,
        wall.clear_home_direct_samples,
        wall.clear_home_samples,
        wall.clear_home_straight_traces,
        wall.clear_home_trajectory_samples,
        wall.behind_wall_returns,
        wall.behind_wall_pickups,
        wall.mean_behind_wall_return_ticks.round() as u32,
        wall.max_behind_wall_return_ticks,
    );
}

fn run_cluster_regression() {
    let params = BenchParams {
        name: "default",
        home_diffusion: 0.03,
        food_lay_strength: 1.5,
        outbound_lay_threshold: 0.5,
        bilinear_deposit: false,
    };
    let t0 = std::time::Instant::now();
    let cluster = run_cluster_bench(params);
    println!(
        "cluster_escape score={:.0} workers={} disp={:.1} sample={:.2} final={:.2} rev={:.2} trap={:.2} elapsed={:.1}s",
        cluster.score,
        cluster.workers,
        cluster.mean_displacement,
        cluster.cluster_sample_rate,
        cluster.final_cluster_rate,
        cluster.reversal_rate,
        cluster.trapped_oscillation_rate,
        t0.elapsed().as_secs_f32(),
    );
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    use tokio::sync::broadcast::error::{RecvError, TryRecvError};
    let mut snap_rx = state.snap_tx.subscribe();
    loop {
        tokio::select! {
            recv = snap_rx.recv() => {
                let mut msg = match recv {
                    Ok(m) => m,
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => return,
                };
                // Drain anything else queued — we only care about the
                // freshest snapshot. This keeps interactive UI commands
                // (place food, paint wall) feeling instant even when the
                // client/network is slow.
                loop {
                    match snap_rx.try_recv() {
                        Ok(newer) => msg = newer,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Lagged(_)) => continue,
                        Err(TryRecvError::Closed) => return,
                    }
                }
                if socket.send(Message::Text(msg)).await.is_err() {
                    return;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(s))) => {
                        if let Ok(cmd) = serde_json::from_str::<command::Command>(&s) {
                            let _ = state.cmd_tx.send(cmd);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => return,
                    _ => {}
                }
            }
        }
    }
}
