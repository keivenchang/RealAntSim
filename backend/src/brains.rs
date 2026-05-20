//! Rule-based brains for queen / worker / soldier.
//!
//! Each is intentionally short (~30 lines) and side-effect free apart from its
//! own internal state. Rewriting any one of these does not affect the others
//! or the sim. To add a neural-net brain, mirror this file: same trait, same
//! `decide` signature.

use crate::brain::{Action, Brain, Perception, PheromoneChannel, WorkerBrainKind};
use crate::entities::Role;
use glam::Vec2;
use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f32::consts::{PI, TAU};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Angular blend: rotate `from` toward `to` by fraction `t` (radians-aware).
fn blend_angle(from: f32, to: f32, t: f32) -> f32 {
    let mut d = (to - from) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    from + d * t
}

fn heading_of(v: glam::Vec2) -> f32 {
    v.y.atan2(v.x)
}

fn angle_delta(from: f32, to: f32) -> f32 {
    let mut d = (to - from) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    d
}

pub fn make_worker_brain(kind: WorkerBrainKind) -> Box<dyn Brain> {
    match kind {
        WorkerBrainKind::Classic => Box::new(WorkerBrain::default()),
        WorkerBrainKind::Weighted => Box::new(WeightedWorkerBrain::default()),
        WorkerBrainKind::Neural => Box::new(NeuralWorkerBrain::default()),
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker state is intentionally small. It keeps stale-trail exposure, a local
/// carrier search heading, and a leaky 30x30 map. The map is deliberately
/// coarse: it bends over-homebound carriers but does not replay a perfect
/// world route.
pub struct WorkerBrain {
    /// Counts ticks where the ant is sitting on a strong Food trail with no
    /// nearby FoodSmell — a "stale trail" pointing to a now-empty pile.
    /// When this exceeds STALE_TRAIL_LIMIT, the ant lays Repellent at the
    /// spot to warn nest-mates (Pharaoh-ant-style no-entry signal).
    stale_trail_ticks: u32,
    /// Heading commitment used only while a food carrier has lost Home
    /// signal. It prevents random jitter from slowly turning the ant back
    /// toward the food plume it just left.
    carrier_search_heading: Option<f32>,
    carrier_wall_side: f32,
    carrier_wall_follow_ticks: u32,
    wall_side: f32,
    wall_follow_ticks: u32,
    local_map: CoarseMap,
}

impl Default for WorkerBrain {
    fn default() -> Self {
        Self {
            stale_trail_ticks: 0,
            carrier_search_heading: None,
            carrier_wall_side: 0.0,
            carrier_wall_follow_ticks: 0,
            wall_side: 0.0,
            wall_follow_ticks: 0,
            local_map: CoarseMap::default(),
        }
    }
}

/// Pharaoh-ant "no entry" signal thresholds.
///   - Trail intensity above this counts as "strong trail here".
const STALE_TRAIL_INTENSITY: f32 = 5.0;
///   - FoodSmell below this means no actual pile is in range.
const STALE_SMELL_THRESHOLD: f32 = 1.0;
///   - Ticks of strong-trail-without-smell before we declare the trail stale
///     and lay Repellent. ~2 sec at 30 Hz.
const STALE_TRAIL_LIMIT: u32 = 35;
const WEIGHTED_STALE_TRAIL_LIMIT: u32 = 60;
const STALE_TRAIL_REPELLENT_MULT: f32 = 1.0;

// johnBuffer-style worker brain. Three forward sensors pick the brightest
// trail. Carriers follow Home (laid by outbound, densest near nest);
// outbound ants follow Food (laid by carriers, densest near food). There is
// deliberately no "vector to nest" fallback for carriers: if no Home trail is
// detectable, they keep momentum plus jitter until they rediscover a signal.

/// Distance ahead (matches grid resolution for 1-cell precision).
const JB_SENSOR_FLOOR: f32 = 0.5; // below this, "no signal" → random walk
const JB_TURN_PER_TICK: f32 = 0.35; // ≈ 20° per tick when committed to a side
const JB_WANDER_JITTER: f32 = 0.30; // random heading change when no signal
const JB_FORWARD_SPEED: f32 = 1.0;
const JB_PICKUP_RANGE: f32 = 3.0;
const JB_REPELLENT_SENSOR_WEIGHT: f32 = 2.0;
const JB_REPELLENT_BIAS_WEIGHT: f32 = 1.5;
const JB_SHORTCUT_SMELL_WEIGHT: f32 = 8.0;
const JB_TRAIL_GRADIENT_WEIGHT: f32 = 2.0;
const JB_FOOD_SMELL_SEARCH_WEIGHT: f32 = 1.5;
const FOOD_SMELL_ROUTE_THRESHOLD: f32 = 1.0;
const FOOD_TRAIL_SHORTCUT_FLOOR: f32 = 0.2;
const HOME_SIGNAL_DEPOSIT_FLOOR: f32 = 0.5;
const CARRIER_BOOTSTRAP_TICKS: u32 = 900;
const CARRIER_BOOTSTRAP_SCALE: f32 = 0.04;
const CARRIER_ROUTE_MEMORY_SCALE: f32 = 1.0;
const CARRIER_SEARCH_BLEND: f32 = 0.65;
const CARRIER_SEARCH_MOMENTUM_WEIGHT: f32 = 0.8;
const CARRIER_SEARCH_REPELLENT_WEIGHT: f32 = 1.0;
const CARRIER_SEARCH_JITTER_SCALE: f32 = 0.08;
const CARRIER_SEARCH_FAR_PICKUP_DIST: f32 = 300.0;
const CLASSIC_OPEN_LONG_FOOD_LAY_SCALE: f32 = 0.05;
const CARRIER_WALL_FOLLOW_TICKS: u32 = 120;
const TRAIL_WALL_FOLLOW_TICKS: u32 = 24;
const LOCAL_MAP_DEFAULT_N: usize = 30;
const LOCAL_MAP_MIN_N: usize = 10;
const LOCAL_MAP_MAX_N: usize = 80;
const LOCAL_MAP_DEFAULT_NOISE: f32 = 0.10;
const LOCAL_MAP_DEFAULT_SUCCESS_DILATE: f32 = 64.0;
const LOCAL_MAP_NOISE_REFERENCE_N: f32 = 30.0;
const LOCAL_MAP_COST_REFERENCE_N: f32 = 30.0;
const LOCAL_MAP_WALL_PROBE_DIST: f32 = 28.0;
const LOCAL_MAP_VISIT_DECAY: f32 = 0.9990;
const LOCAL_MAP_WALL_DECAY: f32 = 0.9998;
const LOCAL_MAP_REPELLENT_DECAY: f32 = 0.9970;
const LOCAL_MAP_SUCCESS_DECAY: f32 = 0.9995;
const LOCAL_MAP_VISIT_WEIGHT: f32 = 0.06;
const LOCAL_MAP_WALL_WEIGHT: f32 = 3.5;
const LOCAL_MAP_REPELLENT_WEIGHT: f32 = 1.2;
const LOCAL_MAP_AVOID_WEIGHT: f32 = 0.15;
const LOCAL_MAP_RETURN_WEIGHT: f32 = 1.25;
const LOCAL_MAP_PLAN_FOOD_WEIGHT: f32 = 3.4;
const LOCAL_MAP_PLAN_HOME_WEIGHT: f32 = 2.2;
const LOCAL_MAP_PLAN_RECOMPUTE_TICKS: u32 = 12;
const LOCAL_MAP_MAX_TRIP_CELLS: usize = 720;
const LOCAL_MAP_HOME_WEAVE_MIN_PICKUP_DIST: f32 = 180.0;
const LOCAL_MAP_HOME_WEAVE_MIN_CONF: f32 = 80.0;
const LOCAL_MAP_HOME_WEAVE_DOT: f32 = -1.01;
const LOCAL_MAP_HOME_WEAVE_ANGLE: f32 = 0.85;
const LOCAL_MAP_WALL_HOME_WEAVE_ANGLE: f32 = 0.45;
const LOCAL_MAP_HOME_WEAVE_BLEND: f32 = 0.90;
const LOCAL_MAP_HOME_WEAVE_PERIOD: u32 = 24;
const LOCAL_MAP_AWAY_FOOD_WEIGHT: f32 = 0.35;
const LOCAL_MAP_PATH_HINT_WEIGHT: f32 = 0.0;
const LOCAL_MAP_ROUGH_HOME_WEIGHT: f32 = 1.0;
const NEURAL_BASE_OBS_DIM: usize = 24;
const NEURAL_MAP_OBS_DIM: usize = 16;
const NEURAL_OBS_DIM: usize = NEURAL_BASE_OBS_DIM + NEURAL_MAP_OBS_DIM;
const NEURAL_MAP_CUE_FLOOR: f32 = 0.05;
const NEIGHBOR_AVOID_WEIGHT: f32 = 0.0;
const NEIGHBOR_AVOID_CARRIER_WEIGHT: f32 = 0.0;
const NEIGHBOR_AVOID_MIN_DIST2: f32 = 4.0;
const NEIGHBOR_AVOID_ROUTE_SIGNAL: f32 = 0.5;
const CLASSIC_CROWD_ESCAPE_MIN_NEIGHBORS: usize = 4;
const CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR: f32 = 0.35;
const CLASSIC_CROWD_ESCAPE_TURN_BLEND: f32 = 0.85;
const CLASSIC_CROWD_ESCAPE_SPEED: f32 = 1.18;

static LOCAL_MAP_GRID_OVERRIDE: AtomicU32 = AtomicU32::new(0);
static LOCAL_MAP_NOISE_MILLI_OVERRIDE: AtomicU32 = AtomicU32::new(u32::MAX);

struct LocalMapConfig {
    n: usize,
    plan_noise: f32,
    scale_noise: bool,
    cost_norm: bool,
    wall_dilate_radius: f32,
    success_dilate_radius: f32,
}

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_f32_clamped(name: &str, default: f32, min: f32, max: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn env_u32_clamped(name: &str, default: u32, min: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn local_map_config() -> LocalMapConfig {
    let grid_override = LOCAL_MAP_GRID_OVERRIDE.load(Ordering::Relaxed);
    let grid = if grid_override > 0 {
        grid_override as usize
    } else {
        std::env::var("REALANTSIM_MAP_GRID")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(LOCAL_MAP_DEFAULT_N)
    }
    .clamp(LOCAL_MAP_MIN_N, LOCAL_MAP_MAX_N);

    let noise_override = LOCAL_MAP_NOISE_MILLI_OVERRIDE.load(Ordering::Relaxed);
    let noise = if noise_override != u32::MAX {
        noise_override as f32 / 1000.0
    } else {
        std::env::var("REALANTSIM_MAP_NOISE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(LOCAL_MAP_DEFAULT_NOISE)
    }
    .clamp(0.0, 2.0);
    LocalMapConfig {
        n: grid,
        plan_noise: noise,
        scale_noise: env_bool("REALANTSIM_MAP_NOISE_SCALE"),
        cost_norm: env_bool("REALANTSIM_MAP_COST_NORM"),
        wall_dilate_radius: env_f32_clamped("REALANTSIM_MAP_WALL_DILATE", 0.0, 0.0, 160.0),
        success_dilate_radius: env_f32_clamped(
            "REALANTSIM_MAP_SUCCESS_DILATE",
            LOCAL_MAP_DEFAULT_SUCCESS_DILATE,
            0.0,
            160.0,
        ),
    }
}

#[derive(Clone, Copy, Default)]
struct LocalMapCell {
    visited: f32,
    wall: f32,
    repellent: f32,
    food_seen: f32,
    home_seen: f32,
    success: f32,
    back_x: f32,
    back_y: f32,
}

#[derive(Clone)]
struct CoarseMap {
    n: usize,
    plan_noise: f32,
    scale_noise: bool,
    cost_norm: bool,
    wall_dilate_radius: f32,
    success_dilate_radius: f32,
    cells: Vec<LocalMapCell>,
    last_pos: Option<Vec2>,
    path_home_vec: Vec2,
    path_home_conf: f32,
    outbound_cells: Vec<usize>,
    known_food_idx: Option<usize>,
    cached_plan_from: Option<usize>,
    cached_plan_goal: Option<usize>,
    cached_plan_tick: u32,
    cached_plan_dir: Vec2,
    success_conf: f32,
}

impl Default for CoarseMap {
    fn default() -> Self {
        let config = local_map_config();
        Self {
            n: config.n,
            plan_noise: config.plan_noise,
            scale_noise: config.scale_noise,
            cost_norm: config.cost_norm,
            wall_dilate_radius: config.wall_dilate_radius,
            success_dilate_radius: config.success_dilate_radius,
            cells: vec![LocalMapCell::default(); config.n * config.n],
            last_pos: None,
            path_home_vec: Vec2::ZERO,
            path_home_conf: 0.0,
            outbound_cells: Vec::with_capacity(LOCAL_MAP_MAX_TRIP_CELLS),
            known_food_idx: None,
            cached_plan_from: None,
            cached_plan_goal: None,
            cached_plan_tick: 0,
            cached_plan_dir: Vec2::ZERO,
            success_conf: 0.0,
        }
    }
}

impl CoarseMap {
    fn world_cell(&self, p: &Perception, pos: Vec2) -> Option<(i32, i32)> {
        if p.world_width <= 1.0 || p.world_height <= 1.0 {
            return None;
        }
        let nx = (pos.x / p.world_width).clamp(0.0, 0.999_999);
        let ny = (pos.y / p.world_height).clamp(0.0, 0.999_999);
        Some((
            (nx * self.n as f32).floor() as i32,
            (ny * self.n as f32).floor() as i32,
        ))
    }

    fn idx(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.n as i32 || y >= self.n as i32 {
            return None;
        }
        Some(y as usize * self.n + x as usize)
    }

    fn xy(&self, idx: usize) -> (i32, i32) {
        ((idx % self.n) as i32, (idx / self.n) as i32)
    }

    fn idx_of_pos(&self, p: &Perception, pos: Vec2) -> Option<usize> {
        let (x, y) = self.world_cell(p, pos)?;
        self.idx(x, y)
    }

    fn cell_center(&self, p: &Perception, idx: usize) -> Vec2 {
        let (x, y) = self.xy(idx);
        Vec2::new(
            (x as f32 + 0.5) * p.world_width / self.n as f32,
            (y as f32 + 0.5) * p.world_height / self.n as f32,
        )
    }

    fn cell_size(&self, p: &Perception) -> f32 {
        (p.world_width / self.n as f32).min(p.world_height / self.n as f32)
    }

    fn cost_cell_scale(&self, p: &Perception) -> f32 {
        if !self.cost_norm {
            return 1.0;
        }
        let reference_cell = (p.world_width / LOCAL_MAP_COST_REFERENCE_N)
            .min(p.world_height / LOCAL_MAP_COST_REFERENCE_N);
        (self.cell_size(p) / reference_cell.max(1.0)).clamp(0.2, 2.5)
    }

    fn noise_cell_scale(&self, p: &Perception) -> f32 {
        let reference_cell = (p.world_width / LOCAL_MAP_NOISE_REFERENCE_N)
            .min(p.world_height / LOCAL_MAP_NOISE_REFERENCE_N);
        (self.cell_size(p) / reference_cell.max(1.0)).clamp(0.2, 2.5)
    }

    fn radius_cells(&self, p: &Perception, radius: f32) -> i32 {
        if radius <= 0.0 {
            return 0;
        }
        (radius / self.cell_size(p).max(1.0)).ceil().max(0.0) as i32
    }

    fn decay(&mut self) {
        for cell in &mut self.cells {
            cell.visited *= LOCAL_MAP_VISIT_DECAY;
            cell.wall *= LOCAL_MAP_WALL_DECAY;
            cell.repellent *= LOCAL_MAP_REPELLENT_DECAY;
            cell.food_seen *= LOCAL_MAP_SUCCESS_DECAY;
            cell.home_seen *= LOCAL_MAP_SUCCESS_DECAY;
            cell.success *= LOCAL_MAP_SUCCESS_DECAY;
            if cell.visited < 0.01 {
                cell.visited = 0.0;
            }
            if cell.wall < 0.01 {
                cell.wall = 0.0;
            }
            if cell.repellent < 0.01 {
                cell.repellent = 0.0;
            }
            if cell.food_seen < 0.01 {
                cell.food_seen = 0.0;
            }
            if cell.home_seen < 0.01 {
                cell.home_seen = 0.0;
            }
            if cell.success < 0.01 {
                cell.success = 0.0;
            }
            cell.back_x *= LOCAL_MAP_VISIT_DECAY;
            cell.back_y *= LOCAL_MAP_VISIT_DECAY;
            if cell.back_x.abs() < 0.01 {
                cell.back_x = 0.0;
            }
            if cell.back_y.abs() < 0.01 {
                cell.back_y = 0.0;
            }
        }
        self.success_conf *= LOCAL_MAP_SUCCESS_DECAY;
    }

    fn mark_idx(&mut self, i: usize, visit: f32, wall: f32, repellent: f32) {
        let cell = &mut self.cells[i];
        cell.visited = cell.visited.max(visit).min(6.0);
        cell.wall = cell.wall.max(wall).min(8.0);
        cell.repellent = cell.repellent.max(repellent).min(6.0);
    }

    fn mark_pos_dilated(
        &mut self,
        p: &Perception,
        pos: Vec2,
        visit: f32,
        wall: f32,
        repellent: f32,
        radius: f32,
    ) {
        let Some(center_idx) = self.idx_of_pos(p, pos) else {
            return;
        };
        let radius_cells = self.radius_cells(p, radius);
        if radius_cells <= 0 {
            self.mark_idx(center_idx, visit, wall, repellent);
            return;
        }
        let (cx, cy) = self.xy(center_idx);
        let r2 = radius_cells * radius_cells;
        for dy in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let Some(idx) = self.idx(cx + dx, cy + dy) else {
                    continue;
                };
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let falloff = (1.0 - dist / (radius_cells as f32 + 1.0)).clamp(0.35, 1.0);
                self.mark_idx(idx, visit * falloff, wall * falloff, repellent * falloff);
            }
        }
    }

    fn mark_back_hint(&mut self, i: usize, back: Vec2) {
        let old = Vec2::new(self.cells[i].back_x, self.cells[i].back_y);
        let blended = (old * 0.35 + back * 0.65).normalize_or_zero();
        self.cells[i].back_x = blended.x;
        self.cells[i].back_y = blended.y;
    }

    fn sample_cell(&self, x: i32, y: i32) -> LocalMapCell {
        let Some(i) = self.idx(x, y) else {
            return LocalMapCell::default();
        };
        self.cells[i]
    }

    fn remember_outbound_cell(&mut self, idx: usize) {
        if self.outbound_cells.last().copied() == Some(idx) {
            return;
        }
        if self.outbound_cells.len() >= LOCAL_MAP_MAX_TRIP_CELLS {
            self.outbound_cells.remove(0);
        }
        self.outbound_cells.push(idx);
    }

    fn reinforce_success_idx(&mut self, p: &Perception, idx: usize) {
        let radius_cells = self.radius_cells(p, self.success_dilate_radius);
        if radius_cells <= 0 {
            let cell = &mut self.cells[idx];
            cell.success = (cell.success + 2.0).min(20.0);
            cell.visited = cell.visited.max(3.0);
            return;
        }
        let (cx, cy) = self.xy(idx);
        let r2 = radius_cells * radius_cells;
        for dy in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let Some(nidx) = self.idx(cx + dx, cy + dy) else {
                    continue;
                };
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let falloff = (1.0 - dist / (radius_cells as f32 + 1.0)).clamp(0.25, 1.0);
                let cell = &mut self.cells[nidx];
                cell.success = (cell.success + 2.0 * falloff).min(20.0);
                cell.visited = cell.visited.max(3.0 * falloff);
            }
        }
    }

    fn reinforce_successful_trip(&mut self, p: &Perception) {
        if let Some(&food_idx) = self.outbound_cells.last() {
            self.known_food_idx = Some(food_idx);
            self.cells[food_idx].food_seen = (self.cells[food_idx].food_seen + 4.0).min(10.0);
        }
        let outbound_cells = self.outbound_cells.clone();
        for idx in outbound_cells {
            self.reinforce_success_idx(p, idx);
        }
        self.success_conf = (self.success_conf + 1.0).min(10.0);
    }

    fn update(&mut self, p: &Perception) {
        self.decay();
        let Some(cur_idx) = self.idx_of_pos(p, p.self_pos) else {
            self.last_pos = Some(p.self_pos);
            return;
        };
        if p.at_nest && !p.carrying_food {
            self.path_home_vec = Vec2::ZERO;
            self.path_home_conf = 0.0;
            self.outbound_cells.clear();
        }
        if let Some(last_pos) = self.last_pos {
            let back = last_pos - p.self_pos;
            if !p.carrying_food && back.length_squared() >= 0.05 {
                self.path_home_vec += back;
                let len = self.path_home_vec.length();
                if len > 900.0 {
                    self.path_home_vec = self.path_home_vec / len * 900.0;
                }
                self.path_home_conf = (self.path_home_conf + back.length()).min(900.0);
                self.mark_back_hint(cur_idx, back.normalize_or_zero());
            }
        }
        self.path_home_vec *= 0.999;
        self.path_home_conf *= 0.999;
        self.last_pos = Some(p.self_pos);
        self.mark_idx(cur_idx, 1.0, 0.0, p.repellent_here * 0.35);
        if p.food_here > 0.2 {
            self.cells[cur_idx].food_seen = self.cells[cur_idx].food_seen.max(p.food_here.min(6.0));
        }
        let home_signal = p
            .sensor_left
            .home
            .max(p.sensor_center.home)
            .max(p.sensor_right.home);
        if home_signal > 0.2 || p.at_nest {
            self.cells[cur_idx].home_seen = self.cells[cur_idx].home_seen.max(home_signal.min(6.0));
        }
        if !p.carrying_food {
            self.remember_outbound_cell(cur_idx);
        }
        if let Some(&(food_pos, _)) = p.nearby_food.first() {
            if let Some(food_idx) = self.idx_of_pos(p, food_pos) {
                self.known_food_idx = Some(food_idx);
                self.cells[food_idx].food_seen = 10.0;
            }
        }

        for (offset, sample) in [
            (-0.6_f32, p.sensor_left),
            (0.0_f32, p.sensor_center),
            (0.6_f32, p.sensor_right),
        ] {
            if sample.wall {
                let pos = p.self_pos
                    + dir_from_heading(p.self_heading + offset) * LOCAL_MAP_WALL_PROBE_DIST;
                self.mark_pos_dilated(p, pos, 0.0, 8.0, 0.0, self.wall_dilate_radius);
            }
        }
        if p.wall_ahead {
            let ahead = p.self_pos + dir_from_heading(p.self_heading) * LOCAL_MAP_WALL_PROBE_DIST;
            self.mark_pos_dilated(p, ahead, 0.0, 8.0, 0.0, self.wall_dilate_radius);
        }
    }

    fn direction_cost(&self, p: &Perception, heading: f32) -> f32 {
        let dir = dir_from_heading(heading);
        let mut cost = 0.0;
        for (step_i, dist) in [32.0, 64.0, 96.0].iter().enumerate() {
            let probe = p.self_pos + dir * *dist;
            let Some((wx, wy)) = self.world_cell(p, probe) else {
                continue;
            };
            let cell = self.sample_cell(wx, wy);
            let w = 1.0 / (step_i as f32 + 1.0);
            cost += (cell.wall * LOCAL_MAP_WALL_WEIGHT
                + cell.visited * LOCAL_MAP_VISIT_WEIGHT
                + cell.repellent * LOCAL_MAP_REPELLENT_WEIGHT)
                * w;
        }
        cost
    }

    fn better_wall_side(&self, p: &Perception, default_side: f32) -> f32 {
        let left_h = p.self_heading - std::f32::consts::FRAC_PI_2;
        let right_h = p.self_heading + std::f32::consts::FRAC_PI_2;
        let left_cost = self.direction_cost(p, left_h);
        let right_cost = self.direction_cost(p, right_h);
        if (left_cost - right_cost).abs() < 0.05 {
            default_side
        } else if left_cost < right_cost {
            -1.0
        } else {
            1.0
        }
    }

    fn avoidance_vector(&self, p: &Perception) -> Vec2 {
        let mut acc = Vec2::ZERO;
        for y in 0..self.n {
            for x in 0..self.n {
                let idx = y * self.n + x;
                let cell = self.cells[idx];
                let cost = cell.wall * LOCAL_MAP_WALL_WEIGHT
                    + cell.repellent * LOCAL_MAP_REPELLENT_WEIGHT
                    + cell.visited * LOCAL_MAP_VISIT_WEIGHT;
                if cost <= 0.01 {
                    continue;
                }
                let center = self.cell_center(p, idx);
                let away = p.self_pos - center;
                let cell_size = (p.world_width / self.n as f32)
                    .min(p.world_height / self.n as f32)
                    .max(1.0);
                let dist2 = away.length_squared().max(cell_size * cell_size);
                acc += away.normalize_or_zero() * (cost / dist2.sqrt());
            }
        }
        acc.normalize_or_zero()
    }

    fn path_home_hint(&self, p: &Perception) -> Vec2 {
        if self.path_home_conf < LOCAL_MAP_HOME_WEAVE_MIN_CONF
            || self.path_home_vec.length_squared() <= 0.0
        {
            return Vec2::ZERO;
        }
        let noise = deterministic_centered(p.self_id, p.tick / 20, 91) * 0.35;
        rotate_dir(self.path_home_vec.normalize_or_zero(), noise)
    }

    fn rough_home_hint(&self, p: &Perception) -> Vec2 {
        let rough = p.nest_pos - p.self_pos;
        if rough.length_squared() <= 0.0 {
            return Vec2::ZERO;
        }
        let noise = 0.0;
        rotate_dir(rough.normalize_or_zero(), noise)
    }

    fn return_hint(&self, p: &Perception) -> Vec2 {
        let Some((cx, cy)) = self.world_cell(p, p.self_pos) else {
            return Vec2::ZERO;
        };
        let mut acc = Vec2::ZERO;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cell = self.sample_cell(cx + dx, cy + dy);
                let back = Vec2::new(cell.back_x, cell.back_y);
                if back.length_squared() <= 0.0 {
                    continue;
                }
                let wall_penalty = 1.0 / (1.0 + cell.wall);
                let visit_weight = (1.0 + cell.visited * 0.2).min(2.0);
                acc += back * wall_penalty * visit_weight;
            }
        }
        acc.normalize_or_zero()
    }

    fn traversal_cost(&self, p: &Perception, idx: usize, goal: usize) -> Option<i32> {
        let cell = self.cells[idx];
        if idx != goal && cell.wall >= 1.5 {
            return None;
        }
        let cost_scale = self.cost_cell_scale(p);
        let unknown: f32 = if cell.visited < 0.05 && cell.success < 0.05 {
            180.0 * cost_scale
        } else {
            0.0
        };
        let noise_scale = if self.scale_noise {
            self.noise_cell_scale(p)
        } else {
            1.0
        };
        let noise = deterministic_centered(
            p.self_id,
            idx as u32,
            (goal as u32).wrapping_add(self.n as u32 * 17),
        ) * self.plan_noise
            * noise_scale
            * 220.0;
        let cost: f32 = 100.0 * cost_scale
            + unknown
            + cell.wall * 260.0 * cost_scale
            + cell.repellent * 90.0 * cost_scale
            + noise
            - cell.success * 6.0 * cost_scale
            - cell.food_seen.min(6.0) * 3.0 * cost_scale
            - cell.home_seen.min(6.0) * 2.0 * cost_scale;
        Some(cost.clamp(20.0, 2_000.0) as i32)
    }

    fn plan_hint(&mut self, p: &Perception, goal: usize) -> Vec2 {
        let Some(start) = self.idx_of_pos(p, p.self_pos) else {
            return Vec2::ZERO;
        };
        if start == goal {
            return Vec2::ZERO;
        }
        if self.cached_plan_from == Some(start)
            && self.cached_plan_goal == Some(goal)
            && p.tick.saturating_sub(self.cached_plan_tick) < LOCAL_MAP_PLAN_RECOMPUTE_TICKS
        {
            return self.cached_plan_dir;
        }

        let mut dist = vec![i32::MAX; self.cells.len()];
        let mut prev = vec![usize::MAX; self.cells.len()];
        let mut heap = BinaryHeap::new();
        dist[start] = 0;
        heap.push((Reverse(0_i32), start));
        const NEIGHBORS: [(i32, i32, i32); 8] = [
            (1, 0, 100),
            (-1, 0, 100),
            (0, 1, 100),
            (0, -1, 100),
            (1, 1, 141),
            (1, -1, 141),
            (-1, 1, 141),
            (-1, -1, 141),
        ];
        while let Some((Reverse(cost), idx)) = heap.pop() {
            if idx == goal {
                break;
            }
            if cost != dist[idx] {
                continue;
            }
            let (x, y) = self.xy(idx);
            for (dx, dy, step_cost) in NEIGHBORS {
                let nx = x + dx;
                let ny = y + dy;
                let Some(next) = self.idx(nx, ny) else {
                    continue;
                };
                if dx != 0 && dy != 0 {
                    let side_a = self
                        .idx(x + dx, y)
                        .map(|i| self.cells[i].wall >= 1.5)
                        .unwrap_or(true);
                    let side_b = self
                        .idx(x, y + dy)
                        .map(|i| self.cells[i].wall >= 1.5)
                        .unwrap_or(true);
                    if side_a || side_b {
                        continue;
                    }
                }
                let Some(cell_cost) = self.traversal_cost(p, next, goal) else {
                    continue;
                };
                let step_scale = self.cost_cell_scale(p);
                let scaled_step_cost = if self.cost_norm {
                    (step_cost as f32 * step_scale).round().max(1.0) as i32
                } else {
                    step_cost
                };
                let next_cost = cost
                    .saturating_add(scaled_step_cost)
                    .saturating_add(cell_cost);
                if next_cost < dist[next] {
                    dist[next] = next_cost;
                    prev[next] = idx;
                    heap.push((Reverse(next_cost), next));
                }
            }
        }
        if prev[goal] == usize::MAX {
            self.cached_plan_from = Some(start);
            self.cached_plan_goal = Some(goal);
            self.cached_plan_tick = p.tick;
            self.cached_plan_dir = Vec2::ZERO;
            return Vec2::ZERO;
        }

        let mut step = goal;
        while prev[step] != start {
            let parent = prev[step];
            if parent == usize::MAX {
                return Vec2::ZERO;
            }
            step = parent;
        }
        let dir = (self.cell_center(p, step) - p.self_pos).normalize_or_zero();
        self.cached_plan_from = Some(start);
        self.cached_plan_goal = Some(goal);
        self.cached_plan_tick = p.tick;
        self.cached_plan_dir = dir;
        dir
    }

    fn food_plan_hint(&mut self, p: &Perception) -> Vec2 {
        if !p.has_walls {
            return Vec2::ZERO;
        }
        if self.success_conf < 0.5 {
            return Vec2::ZERO;
        }
        let Some(goal) = self.known_food_idx else {
            return Vec2::ZERO;
        };
        self.plan_hint(p, goal)
    }

    fn home_plan_hint(&mut self, p: &Perception) -> Vec2 {
        if !p.has_walls {
            return Vec2::ZERO;
        }
        let Some(goal) = self.idx_of_pos(p, p.nest_pos) else {
            return Vec2::ZERO;
        };
        self.plan_hint(p, goal)
    }

    fn carrier_heading(&mut self, p: &Perception, heading: f32) -> Option<f32> {
        if p.pickup_home_dist < LOCAL_MAP_HOME_WEAVE_MIN_PICKUP_DIST {
            return None;
        }
        let path_hint = self.path_home_hint(p);
        let plan_hint = self.home_plan_hint(p);
        let rough_hint = self.rough_home_hint(p);
        let rough_weight = if p.has_walls {
            LOCAL_MAP_ROUGH_HOME_WEIGHT * 0.35
        } else {
            LOCAL_MAP_ROUGH_HOME_WEIGHT
        };
        let home_hint = (path_hint * LOCAL_MAP_PATH_HINT_WEIGHT
            + plan_hint * LOCAL_MAP_PLAN_HOME_WEIGHT
            + rough_hint * rough_weight)
            .normalize_or_zero();
        if home_hint.length_squared() <= 0.0 {
            return None;
        }
        let current = dir_from_heading(heading);
        if current.dot(home_hint) < LOCAL_MAP_HOME_WEAVE_DOT {
            return None;
        }

        let side = if p.has_walls {
            if p.self_id & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        } else {
            let phase = (p.tick / LOCAL_MAP_HOME_WEAVE_PERIOD).max(1);
            if (p.self_id ^ phase) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        };
        let weave_angle = if p.has_walls {
            LOCAL_MAP_WALL_HOME_WEAVE_ANGLE
        } else {
            LOCAL_MAP_HOME_WEAVE_ANGLE
        };
        let mut target = rotate_dir(home_hint, side * weave_angle);
        let avoid = self.avoidance_vector(p);
        if avoid.length_squared() > 0.0 {
            target = (target + avoid * LOCAL_MAP_AVOID_WEIGHT).normalize_or_zero();
        }
        if target.length_squared() <= 0.0 {
            return None;
        }
        Some(blend_angle(
            heading,
            heading_of(target),
            LOCAL_MAP_HOME_WEAVE_BLEND,
        ))
    }

    fn neural_features(&mut self, p: &Perception) -> [f32; NEURAL_MAP_OBS_DIM] {
        let food_plan = world_to_local(self.food_plan_hint(p), p.self_heading);
        let home_plan = world_to_local(self.home_plan_hint(p), p.self_heading);
        let return_hint = world_to_local(self.return_hint(p), p.self_heading);
        let avoid = world_to_local(self.avoidance_vector(p), p.self_heading);
        let rough_home = world_to_local(self.rough_home_hint(p), p.self_heading);
        let path_home = world_to_local(self.path_home_hint(p), p.self_heading);
        let cell = self
            .idx_of_pos(p, p.self_pos)
            .map(|idx| self.cells[idx])
            .unwrap_or_default();
        [
            food_plan.x,
            food_plan.y,
            home_plan.x,
            home_plan.y,
            return_hint.x,
            return_hint.y,
            avoid.x,
            avoid.y,
            rough_home.x,
            rough_home.y,
            path_home.x,
            path_home.y,
            (cell.wall / 8.0).clamp(0.0, 1.0),
            (cell.repellent / 6.0).clamp(0.0, 1.0),
            ((cell.success + self.success_conf) / 20.0).clamp(0.0, 1.0),
            if p.has_walls { 1.0 } else { 0.0 },
        ]
    }
}

fn rotate_dir(v: Vec2, angle: f32) -> Vec2 {
    let (sin, cos) = angle.sin_cos();
    Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
}

fn neighbor_avoidance(p: &Perception) -> Vec2 {
    if p.at_nest
        || p.food_smell_here > NEIGHBOR_AVOID_ROUTE_SIGNAL
        || p.food_here > NEIGHBOR_AVOID_ROUTE_SIGNAL
        || p.sensor_left
            .food
            .max(p.sensor_center.food)
            .max(p.sensor_right.food)
            > NEIGHBOR_AVOID_ROUTE_SIGNAL
        || p.sensor_left
            .home
            .max(p.sensor_center.home)
            .max(p.sensor_right.home)
            > NEIGHBOR_AVOID_ROUTE_SIGNAL
    {
        return Vec2::ZERO;
    }

    let mut acc = Vec2::ZERO;
    for ant in &p.nearby_ants {
        if ant.colony != p.self_colony {
            continue;
        }
        let away = p.self_pos - ant.pos;
        let d2 = away.length_squared();
        if d2 <= NEIGHBOR_AVOID_MIN_DIST2 {
            continue;
        }
        acc += away.normalize_or_zero() / d2.sqrt();
    }
    acc.normalize_or_zero()
}

fn route_signal_floor(p: &Perception) -> f32 {
    p.food_here
        .max(p.food_smell_here)
        .max(p.sensor_left.food)
        .max(p.sensor_center.food)
        .max(p.sensor_right.food)
        .max(p.sensor_left.home)
        .max(p.sensor_center.home)
        .max(p.sensor_right.home)
}

fn open_no_signal_meander(p: &Perception) -> f32 {
    if p.carrying_food
        || p.has_walls
        || p.at_nest
        || p.food_piles > 0
        || !p.nearby_food.is_empty()
        || p.repellent_here > 0.05
        || route_signal_floor(p) > 0.05
    {
        return 0.0;
    }

    let tick = p.tick as f32;
    let ant_phase = p.self_id as f32;
    let fast_arc = (tick * 0.055 + ant_phase * 1.73).sin() * 0.045;
    let slow_arc = (tick * 0.017 + ant_phase * 2.41).sin() * 0.020;
    fast_arc + slow_arc
}

fn wall_sensor_signal(p: &Perception) -> bool {
    p.wall_ahead || p.sensor_left.wall || p.sensor_center.wall || p.sensor_right.wall
}

fn friendly_neighbors_within(p: &Perception, radius: f32) -> usize {
    let radius2 = radius * radius;
    p.nearby_ants
        .iter()
        .filter(|ant| {
            ant.colony == p.self_colony && ant.pos.distance_squared(p.self_pos) <= radius2
        })
        .count()
}

fn dense_wall_crowd_context(p: &Perception) -> bool {
    p.has_walls
        && !p.at_nest
        && p.nearby_food.is_empty()
        && wall_sensor_signal(p)
        && friendly_neighbors_within(p, 34.0) >= 8
}

fn wall_crowd_escape_context(p: &Perception) -> bool {
    p.near_food_wall_pocket || dense_wall_crowd_context(p)
}

fn classic_crowd_escape(p: &Perception) -> Vec2 {
    if !p.has_walls || p.at_nest || !p.nearby_food.is_empty() {
        return Vec2::ZERO;
    }
    let route_signal = route_signal_floor(p);
    let escape_context = wall_crowd_escape_context(p);
    if !escape_context
        && (p.food_smell_here > CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR
            || p.food_here > CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR
            || route_signal > CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR)
    {
        return Vec2::ZERO;
    }

    let mut neighbors = 0usize;
    let mut away = Vec2::ZERO;
    for ant in &p.nearby_ants {
        if ant.colony != p.self_colony {
            continue;
        }
        let delta = p.self_pos - ant.pos;
        let d2 = delta.length_squared();
        if d2 <= 0.001 {
            continue;
        }
        neighbors += 1;
        away += delta.normalize_or_zero() / d2.sqrt().max(1.0);
    }
    let min_neighbors = if escape_context {
        CLASSIC_CROWD_ESCAPE_MIN_NEIGHBORS
    } else {
        CLASSIC_CROWD_ESCAPE_MIN_NEIGHBORS
    };
    if neighbors < min_neighbors {
        return Vec2::ZERO;
    }
    away.normalize_or_zero()
}

impl Brain for WorkerBrain {
    fn decide(&mut self, p: &Perception, _rng: &mut SmallRng) -> Vec<Action> {
        self.local_map.update(p);
        if !p.carrying_food {
            self.carrier_search_heading = None;
            self.carrier_wall_side = 0.0;
            self.carrier_wall_follow_ticks = 0;
        }
        // ---- Pickup at food, if outbound and within range -----------------
        if !p.carrying_food {
            if let Some(&(food_pos, _)) = p.nearby_food.first() {
                if (food_pos - p.self_pos).length() < JB_PICKUP_RANGE {
                    return vec![Action::PickupFood];
                }
            }
        }
        // ---- Drop at nest, if carrying ------------------------------------
        if p.carrying_food && p.at_nest {
            self.local_map.reinforce_successful_trip(p);
            return vec![Action::DropFood];
        }
        // Pharaoh-ant-style "no entry" signal: an outbound ant that spends
        // time on a strong Food trail but sees no FoodSmell has likely found
        // a stale/dead route. Mark the spot and reverse instead of letting
        // nestmates keep reinforcing a path to nowhere.
        if !p.carrying_food
            && p.food_here >= STALE_TRAIL_INTENSITY
            && p.food_smell_here < STALE_SMELL_THRESHOLD
            && p.nearby_food.is_empty()
        {
            self.stale_trail_ticks = self.stale_trail_ticks.saturating_add(1);
            let stale_limit = if p.has_walls && p.food_piles > 0 {
                60
            } else {
                STALE_TRAIL_LIMIT
            };
            if self.stale_trail_ticks >= stale_limit {
                self.stale_trail_ticks = 0;
                let jitter_seed =
                    (p.self_id.wrapping_mul(1103515245) ^ p.tick.wrapping_mul(12345)) as i32;
                let r01 = ((jitter_seed & 0xffff) as f32 / 65535.0) - 0.5;
                let stale_heading = p.self_heading + PI + r01 * 0.8;
                return vec![
                    Action::LayPheromone {
                        channel: PheromoneChannel::Repellent,
                        strength: p.stuck_repel_strength * STALE_TRAIL_REPELLENT_MULT,
                    },
                    Action::SetHeading(stale_heading),
                    Action::Forward(JB_FORWARD_SPEED),
                ];
            }
        } else {
            self.stale_trail_ticks = 0;
        }

        // ---- 3-sensor trail tracking --------------------------------------
        // Carrying → follow Home channel (densest near nest)
        // Outbound → follow Food channel (densest near food)
        let (l_raw, c_raw, r_raw) = if p.carrying_food {
            (
                p.sensor_left.home,
                p.sensor_center.home,
                p.sensor_right.home,
            )
        } else {
            (
                p.sensor_left.food,
                p.sensor_center.food,
                p.sensor_right.food,
            )
        };
        let l = (l_raw - p.sensor_left.repellent * JB_REPELLENT_SENSOR_WEIGHT).max(0.0);
        let c = (c_raw - p.sensor_center.repellent * JB_REPELLENT_SENSOR_WEIGHT).max(0.0);
        let r = (r_raw - p.sensor_right.repellent * JB_REPELLENT_SENSOR_WEIGHT).max(0.0);
        let max = l.max(c).max(r);
        let center_blocked = p.wall_ahead;
        let near_wall_signal = wall_sensor_signal(p);
        let multi_food_wall_plume =
            p.multi_food_wall_context && !p.carrying_food && p.food_smell_here > 0.5;
        let wall_food_smell_plume = wall_crowd_escape_context(p) || multi_food_wall_plume;

        let new_heading = if max >= JB_SENSOR_FLOOR {
            if p.carrying_food {
                self.carrier_search_heading = None;
                self.carrier_wall_follow_ticks = 0;
            }
            // Trail detected — turn toward the strongest sensor. If the
            // center ray hits a wall, commit to a side briefly; otherwise a
            // dense colony keeps sampling the same blocked trail mouth and
            // forms the wall-throat pile seen in the GUI.
            let trail_heading = if center_blocked {
                let default_side = if p.self_id & 1 == 0 { -1.0 } else { 1.0 };
                self.wall_side = self.local_map.better_wall_side(
                    p,
                    if self.wall_side == 0.0 {
                        default_side
                    } else {
                        self.wall_side
                    },
                );
                self.wall_follow_ticks = TRAIL_WALL_FOLLOW_TICKS;
                p.self_heading + self.wall_side * JB_TURN_PER_TICK * 1.5
            } else if p.has_walls && self.wall_follow_ticks > 0 {
                self.wall_follow_ticks -= 1;
                p.self_heading + self.wall_side * JB_TURN_PER_TICK * 0.45
            } else if c >= l && c >= r {
                p.self_heading
            } else if l > r {
                p.self_heading - JB_TURN_PER_TICK
            } else {
                p.self_heading + JB_TURN_PER_TICK
            };
            if (!p.has_walls || !wall_food_smell_plume)
                && !p.carrying_food
                && !p.wall_ahead
                && p.gradient_food_smell.length_squared() > 0.0
            {
                let trail_dir = Vec2::new(trail_heading.cos(), trail_heading.sin());
                let shortcut = (trail_dir + p.gradient_food_smell * JB_SHORTCUT_SMELL_WEIGHT)
                    .normalize_or_zero();
                if shortcut.length_squared() > 0.0 {
                    shortcut.y.atan2(shortcut.x)
                } else {
                    trail_heading
                }
            } else {
                trail_heading
            }
        } else {
            // No trail signal at sensors. Outbound ants can still bias toward
            // FoodSmell because food piles emit it. Carriers deliberately get
            // no Home-gradient fallback here: if the forward sensors cannot
            // see a Home trail, the ant must search rather than smell the nest.
            let jitter_seed =
                (p.self_id.wrapping_mul(2654435761) ^ p.tick.wrapping_mul(0x9E3779B1)) as i32;
            let r01 = ((jitter_seed & 0xffff) as f32 / 65535.0) - 0.5;
            let jitter = r01 * JB_WANDER_JITTER * 2.0;
            if p.carrying_food {
                if p.pickup_home_dist < CARRIER_SEARCH_FAR_PICKUP_DIST {
                    self.carrier_search_heading = None;
                    p.self_heading + jitter
                } else {
                    if self.carrier_wall_side == 0.0 {
                        self.carrier_wall_side = if p.self_id & 1 == 0 { -1.0 } else { 1.0 };
                    }
                    let mut search_heading = self.carrier_search_heading.unwrap_or(p.self_heading);
                    if p.wall_ahead {
                        self.carrier_wall_side =
                            self.local_map.better_wall_side(p, self.carrier_wall_side);
                        self.carrier_wall_follow_ticks = CARRIER_WALL_FOLLOW_TICKS;
                        search_heading += self.carrier_wall_side * JB_TURN_PER_TICK;
                    } else if self.carrier_wall_follow_ticks > 0 {
                        self.carrier_wall_follow_ticks -= 1;
                        search_heading += jitter * CARRIER_SEARCH_JITTER_SCALE;
                    } else {
                        let search_dir = Vec2::new(search_heading.cos(), search_heading.sin())
                            * CARRIER_SEARCH_MOMENTUM_WEIGHT;
                        let return_hint = self.local_map.return_hint(p) * LOCAL_MAP_RETURN_WEIGHT;
                        let planned_home =
                            self.local_map.home_plan_hint(p) * LOCAL_MAP_PLAN_HOME_WEIGHT;
                        let map_avoid = self.local_map.avoidance_vector(p) * LOCAL_MAP_AVOID_WEIGHT;
                        let away_food = if p.gradient_food_smell.length_squared() > 0.0 {
                            -p.gradient_food_smell * LOCAL_MAP_AWAY_FOOD_WEIGHT
                        } else {
                            Vec2::ZERO
                        };
                        let repellent_avoid = if p.repellent_here > 0.2 {
                            -p.gradient_repellent * CARRIER_SEARCH_REPELLENT_WEIGHT
                        } else {
                            Vec2::ZERO
                        };
                        let neighbor_avoid = neighbor_avoidance(p) * NEIGHBOR_AVOID_CARRIER_WEIGHT;
                        let search = (search_dir
                            + return_hint
                            + planned_home
                            + map_avoid
                            + away_food
                            + repellent_avoid
                            + neighbor_avoid)
                            .normalize_or_zero();
                        if search.length_squared() > 0.0 {
                            search_heading = blend_angle(
                                search_heading,
                                heading_of(search),
                                CARRIER_SEARCH_BLEND,
                            );
                        }
                        search_heading += jitter * CARRIER_SEARCH_JITTER_SCALE;
                    }
                    self.carrier_search_heading = Some(search_heading);
                    blend_angle(p.self_heading, search_heading, 0.75)
                }
            } else if p.has_walls
                && p.food_piles == 0
                && route_signal_floor(p) < CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR
                && p.nearby_food.is_empty()
                && !p.at_nest
            {
                let heading_dir = dir_from_heading(p.self_heading);
                let jitter_dir = dir_from_heading(
                    p.self_heading + deterministic_centered(p.self_id, p.tick, 241) * 0.35,
                );
                let default_side = if (p.self_id ^ p.tick) & 1 == 0 {
                    -1.0
                } else {
                    1.0
                };
                let wall_bias = if p.wall_ahead {
                    self.wall_side = self.local_map.better_wall_side(
                        p,
                        if self.wall_side == 0.0 {
                            default_side
                        } else {
                            self.wall_side
                        },
                    );
                    dir_from_heading(p.self_heading + self.wall_side * std::f32::consts::FRAC_PI_2)
                        * 3.0
                } else {
                    Vec2::ZERO
                };
                let desired = (heading_dir * 0.65
                    + self.local_map.avoidance_vector(p) * LOCAL_MAP_AVOID_WEIGHT
                    + neighbor_avoidance(p) * 3.0
                    + classic_crowd_escape(p) * 2.4
                    + wall_bias
                    + jitter_dir * 0.55)
                    .normalize_or_zero();
                if desired.length_squared() > 0.0 {
                    blend_angle(p.self_heading, heading_of(desired), 0.35)
                } else {
                    p.self_heading + jitter * 0.15
                }
            } else {
                let attraction = if p.wall_ahead {
                    let default_side = if (p.self_id ^ p.tick) & 1 == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    self.wall_side = self.local_map.better_wall_side(
                        p,
                        if self.wall_side == 0.0 {
                            default_side
                        } else {
                            self.wall_side
                        },
                    );
                    self.wall_follow_ticks = TRAIL_WALL_FOLLOW_TICKS;
                    let tangent = dir_from_heading(
                        p.self_heading + self.wall_side * std::f32::consts::FRAC_PI_2,
                    );
                    let crowd_escape = classic_crowd_escape(p);
                    let jitter_dir = dir_from_heading(
                        p.self_heading + deterministic_centered(p.self_id, p.tick, 211) * 0.8,
                    );
                    let repellent = if p.repellent_here > 0.2 {
                        -p.gradient_repellent * JB_REPELLENT_BIAS_WEIGHT
                    } else {
                        Vec2::ZERO
                    };
                    let target =
                        (tangent * 2.2 + crowd_escape * 1.4 + repellent + jitter_dir * 0.25)
                            .normalize_or_zero();
                    let heading = if target.length_squared() > 0.0 {
                        blend_angle(
                            p.self_heading,
                            heading_of(target),
                            CLASSIC_CROWD_ESCAPE_TURN_BLEND,
                        )
                    } else {
                        p.self_heading + self.wall_side * JB_TURN_PER_TICK * 1.5
                    };
                    return vec![
                        Action::SetHeadingImmediate(heading),
                        Action::Forward(CLASSIC_CROWD_ESCAPE_SPEED),
                    ];
                } else {
                    let planned_food =
                        self.local_map.food_plan_hint(p) * LOCAL_MAP_PLAN_FOOD_WEIGHT;
                    let food_smell_search =
                        if wall_food_smell_plume || (p.has_walls && near_wall_signal) {
                            Vec2::ZERO
                        } else {
                            p.gradient_food_smell * JB_FOOD_SMELL_SEARCH_WEIGHT
                        };
                    let crowd_escape = classic_crowd_escape(p) * 1.2;
                    p.gradient_to_food * JB_TRAIL_GRADIENT_WEIGHT
                        + food_smell_search
                        + planned_food
                        + crowd_escape
                        + neighbor_avoidance(p) * NEIGHBOR_AVOID_WEIGHT
                };
                let repellent_bias = if p.repellent_here > 0.2 {
                    -p.gradient_repellent * JB_REPELLENT_BIAS_WEIGHT
                } else {
                    Vec2::ZERO
                };
                let bias = (attraction + repellent_bias).normalize_or_zero();
                if bias.length_squared() > 0.0 {
                    let bias_h = bias.y.atan2(bias.x);
                    let mut diff = (bias_h - p.self_heading + PI).rem_euclid(TAU) - PI;
                    if diff > PI {
                        diff -= TAU;
                    }
                    // Cap the bias turn rate (no snap-to-target = no GPS-style
                    // beeline). Ants gradually align with the wall-aware gradient.
                    let step = diff.clamp(-JB_TURN_PER_TICK * 0.5, JB_TURN_PER_TICK * 0.5);
                    let jitter_scale = if p.has_walls
                        && route_signal_floor(p) < CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR
                    {
                        0.15
                    } else {
                        0.5
                    };
                    p.self_heading + step + jitter * jitter_scale
                } else {
                    let jitter_scale = if p.has_walls
                        && route_signal_floor(p) < CLASSIC_CROWD_ESCAPE_SIGNAL_FLOOR
                    {
                        0.15
                    } else {
                        1.0
                    };
                    p.self_heading + open_no_signal_meander(p) + jitter * jitter_scale
                }
            }
        };

        let immediate_heading = if p.carrying_food {
            self.local_map.carrier_heading(p, new_heading)
        } else {
            None
        };
        let new_heading = immediate_heading.unwrap_or(new_heading);

        // ---- Time-decayed deposit -----------------------------------------
        // Carriers lay Food (densest near food = source). Outbound lay Home
        // (densest near nest = source). Strength fades from full at the
        // source to zero at the destination.
        let freshness =
            (1.0 - p.since_state_change as f32 / p.deposit_decay_horizon as f32).max(0.0);
        let lay_strength = p.food_lay_strength * freshness;
        let mut actions = vec![
            Action::SetHeading(new_heading),
            Action::Forward(JB_FORWARD_SPEED),
        ];
        if immediate_heading.is_some() {
            actions.push(Action::SetHeadingImmediate(new_heading));
        }
        if lay_strength > 0.01 {
            let lay = if p.carrying_food {
                let home_signal = p
                    .sensor_left
                    .home
                    .max(p.sensor_center.home)
                    .max(p.sensor_right.home);
                let strength = if home_signal >= HOME_SIGNAL_DEPOSIT_FLOOR {
                    lay_strength
                } else if p.has_return_route {
                    lay_strength * CARRIER_ROUTE_MEMORY_SCALE
                } else if p.since_state_change <= CARRIER_BOOTSTRAP_TICKS
                    && p.food_smell_here < FOOD_SMELL_ROUTE_THRESHOLD
                {
                    lay_strength * CARRIER_BOOTSTRAP_SCALE
                } else {
                    0.0
                };
                let strength = if p.has_return_route
                    && !p.has_walls
                    && p.pickup_home_dist >= CARRIER_SEARCH_FAR_PICKUP_DIST
                {
                    strength * CLASSIC_OPEN_LONG_FOOD_LAY_SCALE
                } else {
                    strength
                };
                if p.food_here < p.food_sat_cap && strength > 0.01 {
                    Some((PheromoneChannel::Food, strength))
                } else {
                    None
                }
            } else {
                // Outbound Home reinforcement stays on established Food
                // trails, but also marks genuine FoodSmell shortcut plumes.
                // That lets a curved bootstrap route collapse toward a
                // shorter chord once ants start cutting the corner.
                let on_route = p.food_here >= p.outbound_lay_threshold
                    || (p.food_here >= FOOD_TRAIL_SHORTCUT_FLOOR
                        && p.food_smell_here >= FOOD_SMELL_ROUTE_THRESHOLD);
                if on_route {
                    Some((PheromoneChannel::Home, lay_strength))
                } else {
                    None
                }
            };
            if let Some((channel, strength)) = lay {
                actions.push(Action::LayPheromone { channel, strength });
            }
        }
        actions
    }
}

// ---------------------------------------------------------------------------
// Experimental weighted worker
// ---------------------------------------------------------------------------

/// Alternate worker brain that turns the same perception into weighted
/// steering vectors instead of selecting a dominant left/center/right branch.
/// It is intentionally separate from `WorkerBrain`: Classic stays the tuned
/// benchmark baseline while this gives us a simpler parameter surface to test.
pub struct WeightedWorkerBrain {
    stale_trail_ticks: u32,
    carrier_search_heading: Option<f32>,
    carrier_wall_side: f32,
    wall_side: f32,
    wall_follow_ticks: u32,
    wall_follow_angle: f32,
    wall_follow_scale: f32,
    local_map: CoarseMap,
}

impl Default for WeightedWorkerBrain {
    fn default() -> Self {
        Self {
            stale_trail_ticks: 0,
            carrier_search_heading: None,
            carrier_wall_side: 0.0,
            wall_side: 0.0,
            wall_follow_ticks: 0,
            wall_follow_angle: 0.42,
            wall_follow_scale: 0.0,
            local_map: CoarseMap::default(),
        }
    }
}

const WEIGHTED_SENSOR_ANGLE: f32 = 0.6;
const WEIGHTED_TRAIL_FLOOR: f32 = 0.35;
const WEIGHTED_OUTBOUND_BLEND: f32 = 0.35;
const WEIGHTED_CARRIER_BLEND: f32 = 0.30;
const WEIGHTED_TRAIL_WEIGHT: f32 = 2.4;
const WEIGHTED_HOME_TRAIL_WEIGHT: f32 = 2.8;
const WEIGHTED_MOMENTUM_WEIGHT: f32 = 0.65;
const WEIGHTED_FOOD_SMELL_WEIGHT: f32 = 2.2;
const WEIGHTED_FOOD_TRAIL_GRADIENT_WEIGHT: f32 = 1.1;
const WEIGHTED_REPELLENT_WEIGHT: f32 = 1.6;
const WEIGHTED_WALL_WEIGHT: f32 = 3.0;
const WEIGHTED_JITTER_WEIGHT: f32 = 0.20;
const WEIGHTED_MAP_HOME_WEIGHT: f32 = 2.2;
const WEIGHTED_MAP_FOOD_WEIGHT: f32 = 2.6;
const WEIGHTED_MAP_RETURN_WEIGHT: f32 = 1.25;
const WEIGHTED_MAP_AVOID_WEIGHT: f32 = 0.15;
const WEIGHTED_OPEN_RETURN_GUARD_TICKS: u32 = 900;
const WEIGHTED_OPEN_RETURN_DOT: f32 = 0.75;
const WEIGHTED_OPEN_RETURN_WEAVE: f32 = 1.25;
const WEIGHTED_OPEN_RETURN_BLEND: f32 = 0.85;
const WEIGHTED_OPEN_RETURN_NO_ROUTE_WEAVE: f32 = 1.05;
const WEIGHTED_OPEN_RETURN_NO_ROUTE_BLEND: f32 = 0.75;
const WEIGHTED_OPEN_LONG_FOOD_LAY_SCALE: f32 = 0.40;

struct WeightedRuntimeConfig {
    repellent_weight: f32,
    wall_weight: f32,
    wall_food_smell_scale: f32,
    wall_no_signal_jitter_mult: f32,
    wall_no_signal_neighbor_avoid_weight: f32,
    wall_route_memory_scale: f32,
    wall_follow_ticks: u32,
    wall_follow_angle: f32,
    wall_follow_scale: f32,
    wall_crowd_follow_ticks: u32,
    wall_crowd_follow_angle: f32,
    wall_crowd_follow_scale: f32,
    wall_crowd_escape_blend: f32,
    wall_crowd_tangent_weight: f32,
    open_return_dot: f32,
    open_return_weave: f32,
    open_return_blend: f32,
    open_return_no_route_weave: f32,
    open_return_no_route_blend: f32,
    open_long_food_lay_scale: f32,
}

fn weighted_runtime_config() -> &'static WeightedRuntimeConfig {
    static CONFIG: OnceLock<WeightedRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(|| WeightedRuntimeConfig {
        repellent_weight: env_f32_clamped(
            "REALANTSIM_WEIGHTED_REPELLENT_WEIGHT",
            WEIGHTED_REPELLENT_WEIGHT,
            0.0,
            8.0,
        ),
        wall_weight: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_WEIGHT",
            WEIGHTED_WALL_WEIGHT,
            0.0,
            12.0,
        ),
        wall_food_smell_scale: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE",
            0.90,
            0.0,
            3.0,
        ),
        wall_no_signal_jitter_mult: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT",
            2.2,
            0.0,
            5.0,
        ),
        wall_no_signal_neighbor_avoid_weight: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_NEIGHBOR_AVOID_WEIGHT",
            2.0,
            0.0,
            6.0,
        ),
        wall_route_memory_scale: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE",
            1.57,
            0.0,
            3.0,
        ),
        wall_follow_ticks: env_u32_clamped("REALANTSIM_WEIGHTED_WALL_FOLLOW_TICKS", 18, 0, 96),
        wall_follow_angle: env_f32_clamped("REALANTSIM_WEIGHTED_WALL_FOLLOW_ANGLE", 0.42, 0.0, PI),
        wall_follow_scale: env_f32_clamped("REALANTSIM_WEIGHTED_WALL_FOLLOW_SCALE", 0.30, 0.0, 1.5),
        wall_crowd_follow_ticks: env_u32_clamped(
            "REALANTSIM_WEIGHTED_WALL_CROWD_FOLLOW_TICKS",
            TRAIL_WALL_FOLLOW_TICKS,
            0,
            96,
        ),
        wall_crowd_follow_angle: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_CROWD_FOLLOW_ANGLE",
            0.42,
            0.0,
            PI,
        ),
        wall_crowd_follow_scale: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_CROWD_FOLLOW_SCALE",
            0.35,
            0.0,
            1.5,
        ),
        wall_crowd_escape_blend: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_CROWD_ESCAPE_BLEND",
            0.96,
            0.0,
            1.0,
        ),
        wall_crowd_tangent_weight: env_f32_clamped(
            "REALANTSIM_WEIGHTED_WALL_CROWD_TANGENT_WEIGHT",
            1.5,
            0.0,
            6.0,
        ),
        open_return_dot: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_RETURN_DOT",
            WEIGHTED_OPEN_RETURN_DOT,
            -1.0,
            1.0,
        ),
        open_return_weave: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_RETURN_WEAVE",
            WEIGHTED_OPEN_RETURN_WEAVE,
            0.0,
            PI,
        ),
        open_return_blend: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_RETURN_BLEND",
            WEIGHTED_OPEN_RETURN_BLEND,
            0.0,
            1.0,
        ),
        open_return_no_route_weave: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_RETURN_NO_ROUTE_WEAVE",
            WEIGHTED_OPEN_RETURN_NO_ROUTE_WEAVE,
            0.0,
            PI,
        ),
        open_return_no_route_blend: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_RETURN_NO_ROUTE_BLEND",
            WEIGHTED_OPEN_RETURN_NO_ROUTE_BLEND,
            0.0,
            1.0,
        ),
        open_long_food_lay_scale: env_f32_clamped(
            "REALANTSIM_WEIGHTED_OPEN_LONG_FOOD_LAY_SCALE",
            WEIGHTED_OPEN_LONG_FOOD_LAY_SCALE,
            0.0,
            2.0,
        ),
    })
}

fn deterministic_centered(id: u32, tick: u32, salt: u32) -> f32 {
    let x = id
        .wrapping_mul(747_796_405)
        .wrapping_add(tick.wrapping_mul(2_891_336_453))
        .wrapping_add(salt.wrapping_mul(277_803_737));
    ((x >> 16) as f32 / 65_535.0) - 0.5
}

fn dir_from_heading(heading: f32) -> Vec2 {
    Vec2::new(heading.cos(), heading.sin())
}

fn world_to_local(v: Vec2, heading: f32) -> Vec2 {
    let fwd = dir_from_heading(heading);
    let left = Vec2::new(-fwd.y, fwd.x);
    Vec2::new(v.dot(fwd), v.dot(left))
}

fn vec_feature_strength(features: &[f32], start: usize) -> f32 {
    Vec2::new(features[start], features[start + 1]).length()
}

fn neural_map_cue_strength(features: &[f32; NEURAL_MAP_OBS_DIM]) -> f32 {
    vec_feature_strength(features, 0)
        .max(vec_feature_strength(features, 2))
        .max(vec_feature_strength(features, 4))
        .max(vec_feature_strength(features, 6))
        .max(vec_feature_strength(features, 10))
        .max(features[12])
        .max(features[13])
        .max(features[14])
}

fn weighted_sensor_vector(p: &Perception, carrying_food: bool) -> (Vec2, f32) {
    let heading = p.self_heading;
    let dirs = [
        dir_from_heading(heading - WEIGHTED_SENSOR_ANGLE),
        dir_from_heading(heading),
        dir_from_heading(heading + WEIGHTED_SENSOR_ANGLE),
    ];
    let raw = if carrying_food {
        [
            p.sensor_left.home,
            p.sensor_center.home,
            p.sensor_right.home,
        ]
    } else {
        [
            p.sensor_left.food,
            p.sensor_center.food,
            p.sensor_right.food,
        ]
    };
    let repel = [
        p.sensor_left.repellent,
        p.sensor_center.repellent,
        p.sensor_right.repellent,
    ];
    let mut acc = Vec2::ZERO;
    let mut max_signal = 0.0_f32;
    for i in 0..3 {
        let score = (raw[i] - repel[i] * JB_REPELLENT_SENSOR_WEIGHT).max(0.0);
        acc += dirs[i] * score;
        max_signal = max_signal.max(score);
    }
    (acc.normalize_or_zero(), max_signal)
}

#[derive(Deserialize)]
struct NeuralLayerWeights {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

#[derive(Deserialize)]
struct NeuralWeightsFile {
    obs_dim: usize,
    turn_scale: f32,
    layers: Vec<NeuralLayerWeights>,
}

struct NeuralLayer {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

struct NeuralNet {
    obs_dim: usize,
    turn_scale: f32,
    layers: Vec<NeuralLayer>,
}

impl NeuralNet {
    fn from_file(file: NeuralWeightsFile) -> Option<Self> {
        if file.obs_dim == 0 || file.layers.len() != 3 {
            return None;
        }
        let mut input_dim = file.obs_dim;
        let mut layers = Vec::with_capacity(file.layers.len());
        for layer in file.layers {
            if layer.weight.is_empty() || layer.bias.len() != layer.weight.len() {
                return None;
            }
            if layer.weight.iter().any(|row| row.len() != input_dim) {
                return None;
            }
            input_dim = layer.bias.len();
            layers.push(NeuralLayer {
                weight: layer.weight,
                bias: layer.bias,
            });
        }
        if input_dim != 1 {
            return None;
        }
        Some(Self {
            obs_dim: file.obs_dim,
            turn_scale: file.turn_scale,
            layers,
        })
    }

    fn predict_turn(&self, obs: &[f32]) -> Option<f32> {
        if obs.len() != self.obs_dim {
            return None;
        }
        let mut x = obs.to_vec();
        for (layer_i, layer) in self.layers.iter().enumerate() {
            let mut y = Vec::with_capacity(layer.bias.len());
            for (row, bias) in layer.weight.iter().zip(&layer.bias) {
                let v = row
                    .iter()
                    .zip(&x)
                    .fold(*bias, |acc, (weight, input)| acc + weight * input);
                y.push(v);
            }
            if layer_i + 1 == self.layers.len() {
                return Some(y[0].tanh() * self.turn_scale);
            }
            for v in &mut y {
                *v = v.tanh();
            }
            x = y;
        }
        None
    }
}

static NEURAL_WORKER_NET: OnceLock<Option<NeuralNet>> = OnceLock::new();

fn neural_worker_net() -> Option<&'static NeuralNet> {
    NEURAL_WORKER_NET
        .get_or_init(|| {
            let path = std::env::var("REALANTSIM_NEURAL_WORKER_WEIGHTS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("..")
                        .join("assets")
                        .join("neural_worker_weights.json")
                });
            let text = std::fs::read_to_string(path).ok()?;
            let file = serde_json::from_str::<NeuralWeightsFile>(&text).ok()?;
            NeuralNet::from_file(file)
        })
        .as_ref()
}

fn neural_observation(
    p: &Perception,
    map_features: [f32; NEURAL_MAP_OBS_DIM],
) -> [f32; NEURAL_OBS_DIM] {
    let food_grad = world_to_local(p.gradient_to_food, p.self_heading);
    let smell_grad = world_to_local(p.gradient_food_smell, p.self_heading);
    let repel_grad = world_to_local(p.gradient_repellent, p.self_heading);
    let base = [
        if p.carrying_food { 1.0 } else { 0.0 },
        if p.wall_ahead { 1.0 } else { 0.0 },
        1.0,
        0.0,
        if p.at_nest { 1.0 } else { 0.0 },
        (p.pickup_home_dist / 600.0).clamp(0.0, 1.0),
        (p.sensor_left.food / 8.0).clamp(0.0, 1.0),
        (p.sensor_center.food / 8.0).clamp(0.0, 1.0),
        (p.sensor_right.food / 8.0).clamp(0.0, 1.0),
        (p.sensor_left.home / 8.0).clamp(0.0, 1.0),
        (p.sensor_center.home / 8.0).clamp(0.0, 1.0),
        (p.sensor_right.home / 8.0).clamp(0.0, 1.0),
        (p.sensor_left.repellent / 8.0).clamp(0.0, 1.0),
        (p.sensor_center.repellent / 8.0).clamp(0.0, 1.0),
        (p.sensor_right.repellent / 8.0).clamp(0.0, 1.0),
        (p.food_here / 4.0).clamp(0.0, 1.0),
        (p.food_smell_here / 4.0).clamp(0.0, 1.0),
        (p.repellent_here / 4.0).clamp(0.0, 1.0),
        food_grad.x,
        food_grad.y,
        smell_grad.x,
        smell_grad.y,
        repel_grad.x,
        repel_grad.y,
    ];
    let mut obs = [0.0; NEURAL_OBS_DIM];
    obs[..NEURAL_BASE_OBS_DIM].copy_from_slice(&base);
    obs[NEURAL_BASE_OBS_DIM..].copy_from_slice(&map_features);
    obs
}

#[derive(Clone)]
pub struct TeacherSample {
    pub obs: [f32; NEURAL_OBS_DIM],
    pub target_turn: f32,
}

pub struct TeacherSampleBuffer {
    max_samples: usize,
    samples: Mutex<Vec<TeacherSample>>,
}

impl TeacherSampleBuffer {
    pub fn new(max_samples: usize) -> Arc<Self> {
        Arc::new(Self {
            max_samples,
            samples: Mutex::new(Vec::with_capacity(max_samples.min(1_000_000))),
        })
    }

    pub fn len(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    pub fn snapshot(&self) -> Vec<TeacherSample> {
        self.samples.lock().unwrap().clone()
    }

    fn push(&self, sample: TeacherSample) {
        let mut samples = self.samples.lock().unwrap();
        if samples.len() < self.max_samples {
            samples.push(sample);
        }
    }
}

pub struct TeacherWorkerBrain {
    inner: WorkerBrain,
    samples: Arc<TeacherSampleBuffer>,
}

impl TeacherWorkerBrain {
    fn new(samples: Arc<TeacherSampleBuffer>) -> Self {
        Self {
            inner: WorkerBrain::default(),
            samples,
        }
    }
}

impl Brain for TeacherWorkerBrain {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action> {
        let actions = self.inner.decide(p, rng);
        if let Some(target_heading) = actions.iter().find_map(|action| match action {
            Action::SetHeading(h) => Some(*h),
            Action::SetHeadingImmediate(h) => Some(*h),
            _ => None,
        }) {
            let map_features = self.inner.local_map.neural_features(p);
            self.samples.push(TeacherSample {
                obs: neural_observation(p, map_features),
                target_turn: angle_delta(p.self_heading, target_heading).clamp(-0.7, 0.7),
            });
        }
        actions
    }
}

pub fn make_teacher_worker_brain(samples: Arc<TeacherSampleBuffer>) -> Box<dyn Brain> {
    Box::new(TeacherWorkerBrain::new(samples))
}

impl Brain for WeightedWorkerBrain {
    fn decide(&mut self, p: &Perception, _rng: &mut SmallRng) -> Vec<Action> {
        let cfg = weighted_runtime_config();
        self.local_map.update(p);
        if !p.carrying_food {
            self.carrier_search_heading = None;
            self.carrier_wall_side = 0.0;
        }

        if !p.carrying_food {
            if let Some(&(food_pos, _)) = p.nearby_food.first() {
                if (food_pos - p.self_pos).length() < JB_PICKUP_RANGE {
                    return vec![Action::PickupFood];
                }
            }
        }
        if p.carrying_food && p.at_nest {
            self.local_map.reinforce_successful_trip(p);
            return vec![Action::DropFood];
        }

        if !p.carrying_food
            && p.food_here >= STALE_TRAIL_INTENSITY
            && p.food_smell_here < STALE_SMELL_THRESHOLD
            && p.nearby_food.is_empty()
        {
            self.stale_trail_ticks = self.stale_trail_ticks.saturating_add(1);
            if self.stale_trail_ticks >= WEIGHTED_STALE_TRAIL_LIMIT {
                self.stale_trail_ticks = 0;
                let r01 = deterministic_centered(p.self_id, p.tick, 17);
                let stale_heading = if p.gradient_to_food.length_squared() > 0.0 {
                    let away = -p.gradient_to_food;
                    heading_of(rotate_dir(away, r01 * 0.5))
                } else {
                    p.self_heading + PI + r01 * 0.8
                };
                return vec![
                    Action::LayPheromone {
                        channel: PheromoneChannel::Repellent,
                        strength: p.stuck_repel_strength * STALE_TRAIL_REPELLENT_MULT,
                    },
                    Action::SetHeading(stale_heading),
                    Action::Forward(JB_FORWARD_SPEED),
                ];
            }
        } else {
            self.stale_trail_ticks = 0;
        }

        let heading_dir = dir_from_heading(p.self_heading);
        let jitter = deterministic_centered(p.self_id, p.tick, 41) * JB_WANDER_JITTER;
        let jitter_dir = dir_from_heading(p.self_heading + jitter);
        let repellent_avoid = if p.repellent_here > 0.2 {
            -p.gradient_repellent * cfg.repellent_weight
        } else {
            Vec2::ZERO
        };
        let food_smell_weight = WEIGHTED_FOOD_SMELL_WEIGHT
            * if wall_crowd_escape_context(p) {
                0.0
            } else if p.multi_food_wall_context {
                0.10
            } else if p.has_walls {
                cfg.wall_food_smell_scale
            } else {
                1.0
            };
        let map_avoid = self.local_map.avoidance_vector(p) * WEIGHTED_MAP_AVOID_WEIGHT;

        let mut new_heading = if p.carrying_food {
            let (home_sensor, home_signal) = weighted_sensor_vector(p, true);
            let map_home = self.local_map.home_plan_hint(p) * WEIGHTED_MAP_HOME_WEIGHT;
            let map_return = self.local_map.return_hint(p) * WEIGHTED_MAP_RETURN_WEIGHT;
            let short_open_home = if !p.has_walls
                && p.pickup_home_dist >= LOCAL_MAP_HOME_WEAVE_MIN_PICKUP_DIST
                && p.pickup_home_dist < CARRIER_SEARCH_FAR_PICKUP_DIST
            {
                let rough_home = self.local_map.rough_home_hint(p);
                if rough_home.length_squared() > 0.0 {
                    let phase = (p.tick / LOCAL_MAP_HOME_WEAVE_PERIOD).max(1);
                    let side = if (p.self_id ^ phase) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    rotate_dir(rough_home, side * LOCAL_MAP_HOME_WEAVE_ANGLE)
                        * WEIGHTED_MAP_HOME_WEIGHT
                } else {
                    Vec2::ZERO
                }
            } else {
                Vec2::ZERO
            };
            let desired = if home_signal >= WEIGHTED_TRAIL_FLOOR {
                self.carrier_search_heading = None;
                home_sensor * WEIGHTED_HOME_TRAIL_WEIGHT
                    + heading_dir * WEIGHTED_MOMENTUM_WEIGHT
                    + repellent_avoid
                    + map_home
                    + map_return
                    + short_open_home
                    + map_avoid
            } else {
                if self.carrier_wall_side == 0.0 {
                    self.carrier_wall_side = if p.self_id & 1 == 0 { -1.0 } else { 1.0 };
                }
                let committed = self
                    .carrier_search_heading
                    .map(dir_from_heading)
                    .unwrap_or(heading_dir);
                let away_from_food_smell = if p.pickup_home_dist >= CARRIER_SEARCH_FAR_PICKUP_DIST
                    && p.gradient_food_smell.length_squared() > 0.0
                {
                    -p.gradient_food_smell * food_smell_weight
                } else {
                    Vec2::ZERO
                };
                let wall_bias = if p.wall_ahead {
                    self.carrier_wall_side =
                        self.local_map.better_wall_side(p, self.carrier_wall_side);
                    dir_from_heading(
                        p.self_heading + self.carrier_wall_side * std::f32::consts::FRAC_PI_2,
                    ) * cfg.wall_weight
                } else {
                    Vec2::ZERO
                };
                let jitter_weight = if p.has_walls && p.gradient_food_smell.length_squared() <= 0.0
                {
                    WEIGHTED_JITTER_WEIGHT * cfg.wall_no_signal_jitter_mult
                } else {
                    WEIGHTED_JITTER_WEIGHT
                };
                let no_signal_neighbor_avoid = if p.has_walls
                    && home_signal < WEIGHTED_TRAIL_FLOOR
                    && p.gradient_food_smell.length_squared() <= 0.0
                {
                    neighbor_avoidance(p) * cfg.wall_no_signal_neighbor_avoid_weight
                } else {
                    Vec2::ZERO
                };
                committed * 1.4
                    + map_home
                    + map_return
                    + short_open_home
                    + away_from_food_smell
                    + repellent_avoid
                    + map_avoid
                    + no_signal_neighbor_avoid
                    + wall_bias
                    + jitter_dir * jitter_weight
            };
            let desired_dir = desired.normalize_or_zero();
            if desired_dir.length_squared() > 0.0 {
                let target = heading_of(desired_dir);
                self.carrier_search_heading = Some(target);
                blend_angle(p.self_heading, target, WEIGHTED_CARRIER_BLEND)
            } else {
                p.self_heading + jitter
            }
        } else {
            let (food_sensor, food_signal) = weighted_sensor_vector(p, false);
            let map_food = self.local_map.food_plan_hint(p) * WEIGHTED_MAP_FOOD_WEIGHT;
            let trail_pull = if food_signal >= WEIGHTED_TRAIL_FLOOR {
                food_sensor * WEIGHTED_TRAIL_WEIGHT
            } else {
                Vec2::ZERO
            };
            let wall_side = if (p.self_id ^ p.tick) & 1 == 0 {
                -1.0
            } else {
                1.0
            };
            let wall_crowd_context = wall_crowd_escape_context(p);
            let wall_bias = if p.wall_ahead {
                self.wall_side = self.local_map.better_wall_side(
                    p,
                    if self.wall_side == 0.0 {
                        wall_side
                    } else {
                        self.wall_side
                    },
                );
                if wall_crowd_context {
                    self.wall_follow_ticks = cfg.wall_crowd_follow_ticks;
                    self.wall_follow_angle = cfg.wall_crowd_follow_angle;
                    self.wall_follow_scale = cfg.wall_crowd_follow_scale;
                } else {
                    self.wall_follow_ticks = cfg.wall_follow_ticks;
                    self.wall_follow_angle = cfg.wall_follow_angle;
                    self.wall_follow_scale = cfg.wall_follow_scale;
                }
                dir_from_heading(p.self_heading + self.wall_side * std::f32::consts::FRAC_PI_2)
                    * cfg.wall_weight
            } else if p.has_walls && self.wall_follow_ticks > 0 {
                self.wall_follow_ticks -= 1;
                dir_from_heading(p.self_heading + self.wall_side * self.wall_follow_angle)
                    * cfg.wall_weight
                    * self.wall_follow_scale
            } else {
                Vec2::ZERO
            };
            let wall_crowd_tangent = if wall_crowd_context {
                let split_side = if p.self_id & 1 == 0 { -1.0 } else { 1.0 };
                let vertical_side = if (p.self_pos.y - p.nest_pos.y).abs() > 38.0 {
                    (p.self_pos.y - p.nest_pos.y).signum()
                } else {
                    split_side
                };
                Vec2::new(0.0, vertical_side) * cfg.wall_weight * cfg.wall_crowd_tangent_weight
            } else {
                Vec2::ZERO
            };
            let desired = heading_dir * WEIGHTED_MOMENTUM_WEIGHT
                + trail_pull
                + p.gradient_to_food * WEIGHTED_FOOD_TRAIL_GRADIENT_WEIGHT
                + p.gradient_food_smell * food_smell_weight
                + repellent_avoid
                + map_food
                + map_avoid
                + if p.has_walls
                    && food_signal < WEIGHTED_TRAIL_FLOOR
                    && p.gradient_food_smell.length_squared() <= 0.0
                {
                    neighbor_avoidance(p) * cfg.wall_no_signal_neighbor_avoid_weight
                } else {
                    Vec2::ZERO
                }
                + wall_bias
                + wall_crowd_tangent
                + jitter_dir
                    * if p.has_walls
                        && food_signal < WEIGHTED_TRAIL_FLOOR
                        && p.gradient_food_smell.length_squared() <= 0.0
                    {
                        WEIGHTED_JITTER_WEIGHT * cfg.wall_no_signal_jitter_mult
                    } else {
                        WEIGHTED_JITTER_WEIGHT
                    };
            let desired_dir = desired.normalize_or_zero();
            if desired_dir.length_squared() > 0.0 {
                blend_angle(
                    p.self_heading,
                    heading_of(desired_dir),
                    WEIGHTED_OUTBOUND_BLEND,
                )
            } else {
                p.self_heading + jitter
            }
        };
        if p.carrying_food
            && !p.has_walls
            && WEIGHTED_OPEN_RETURN_GUARD_TICKS > 0
            && p.since_state_change <= WEIGHTED_OPEN_RETURN_GUARD_TICKS
            && p.pickup_home_dist >= CARRIER_SEARCH_FAR_PICKUP_DIST
        {
            let to_nest = (p.nest_pos - p.self_pos).normalize_or_zero();
            let proposed = dir_from_heading(new_heading);
            if to_nest.length_squared() > 0.0 && proposed.dot(to_nest) >= cfg.open_return_dot {
                let side = if (p.self_id ^ (p.tick / 24)) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let (weave, blend) = if p.has_return_route {
                    (cfg.open_return_weave, cfg.open_return_blend)
                } else {
                    (
                        cfg.open_return_no_route_weave,
                        cfg.open_return_no_route_blend,
                    )
                };
                let woven = rotate_dir(to_nest, side * weave);
                new_heading = blend_angle(new_heading, heading_of(woven), blend);
            }
        }

        if wall_crowd_escape_context(p) {
            let crowd_escape = classic_crowd_escape(p);
            if crowd_escape.length_squared() > 0.0 {
                new_heading = blend_angle(
                    new_heading,
                    heading_of(crowd_escape),
                    cfg.wall_crowd_escape_blend,
                );
            }
        }

        let freshness =
            (1.0 - p.since_state_change as f32 / p.deposit_decay_horizon as f32).max(0.0);
        let lay_strength = p.food_lay_strength * freshness;
        let mut actions = vec![
            Action::SetHeading(new_heading),
            Action::Forward(JB_FORWARD_SPEED),
        ];
        if lay_strength > 0.01 {
            let lay = if p.carrying_food {
                let home_signal = p
                    .sensor_left
                    .home
                    .max(p.sensor_center.home)
                    .max(p.sensor_right.home);
                let strength = if home_signal >= HOME_SIGNAL_DEPOSIT_FLOOR {
                    lay_strength
                } else if p.has_return_route {
                    lay_strength
                        * if p.has_walls {
                            cfg.wall_route_memory_scale
                        } else {
                            CARRIER_ROUTE_MEMORY_SCALE
                        }
                } else if p.since_state_change <= CARRIER_BOOTSTRAP_TICKS
                    && p.food_smell_here < FOOD_SMELL_ROUTE_THRESHOLD
                {
                    lay_strength * CARRIER_BOOTSTRAP_SCALE
                } else {
                    0.0
                };
                let strength =
                    if !p.has_walls && p.pickup_home_dist >= CARRIER_SEARCH_FAR_PICKUP_DIST {
                        strength * cfg.open_long_food_lay_scale
                    } else {
                        strength
                    };
                if p.food_here < p.food_sat_cap && strength > 0.01 {
                    Some((PheromoneChannel::Food, strength))
                } else {
                    None
                }
            } else {
                let on_route = p.food_here >= p.outbound_lay_threshold
                    || (p.food_here >= FOOD_TRAIL_SHORTCUT_FLOOR
                        && p.food_smell_here >= FOOD_SMELL_ROUTE_THRESHOLD);
                if on_route {
                    Some((PheromoneChannel::Home, lay_strength))
                } else {
                    None
                }
            };
            if let Some((channel, strength)) = lay {
                actions.push(Action::LayPheromone { channel, strength });
            }
        }
        actions
    }
}

/// Tiny-MLP worker brain. The neural net only predicts a bounded local turn;
/// pickup/drop and pheromone-deposit safety gates remain explicit so a bad
/// prototype cannot bypass the no-GPS and stale-trail rules.
pub struct NeuralWorkerBrain {
    classic: WorkerBrain,
}

impl Default for NeuralWorkerBrain {
    fn default() -> Self {
        Self {
            classic: WorkerBrain::default(),
        }
    }
}

struct NeuralRuntimeConfig {
    blend: f32,
    min_ticks_since_pickup: u32,
    min_pickup_dist: f32,
    max_pickup_dist: f32,
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn neural_runtime_config() -> &'static NeuralRuntimeConfig {
    static CONFIG: OnceLock<NeuralRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(|| NeuralRuntimeConfig {
        blend: env_f32("REALANTSIM_NEURAL_BLEND", 0.55).clamp(0.0, 1.0),
        min_ticks_since_pickup: env_u32("REALANTSIM_NEURAL_MIN_TICKS", 0),
        min_pickup_dist: env_f32(
            "REALANTSIM_NEURAL_MIN_PICKUP_DIST",
            CARRIER_SEARCH_FAR_PICKUP_DIST,
        ),
        max_pickup_dist: env_f32("REALANTSIM_NEURAL_MAX_PICKUP_DIST", 500.0),
    })
}

impl Brain for NeuralWorkerBrain {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action> {
        let mut classic_actions = self.classic.decide(p, rng);
        let Some(net) = neural_worker_net() else {
            return classic_actions;
        };

        let cfg = neural_runtime_config();
        let home_signal = p
            .sensor_left
            .home
            .max(p.sensor_center.home)
            .max(p.sensor_right.home);
        let use_neural_local_return = p.carrying_food
            && !p.has_return_route
            && home_signal < WEIGHTED_TRAIL_FLOOR
            && p.since_state_change >= cfg.min_ticks_since_pickup
            && p.pickup_home_dist >= cfg.min_pickup_dist
            && p.pickup_home_dist < cfg.max_pickup_dist;
        if !use_neural_local_return {
            return classic_actions;
        }

        let map_features = self.classic.local_map.neural_features(p);
        if neural_map_cue_strength(&map_features) < NEURAL_MAP_CUE_FLOOR {
            return classic_actions;
        }
        let obs = neural_observation(p, map_features);
        let Some(turn_delta) = net.predict_turn(&obs) else {
            return classic_actions;
        };

        let neural_heading = p.self_heading + turn_delta;
        for action in &mut classic_actions {
            match action {
                Action::SetHeading(classic_heading)
                | Action::SetHeadingImmediate(classic_heading) => {
                    *classic_heading = blend_angle(*classic_heading, neural_heading, cfg.blend);
                    break;
                }
                _ => {}
            }
        }
        classic_actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Action, ForwardSample, Perception};
    use rand::SeedableRng;

    fn sample() -> ForwardSample {
        ForwardSample {
            food: 0.0,
            repellent: 0.0,
            home: 0.0,
            wall: false,
        }
    }

    fn base_perception(carrying_food: bool) -> Perception {
        Perception {
            self_id: 7,
            self_pos: Vec2::new(100.0, 100.0),
            self_heading: PI * 0.5,
            world_width: 1920.0,
            world_height: 1080.0,
            self_colony: 0,
            carrying_food,
            pickup_home_dist: 0.0,
            has_return_route: false,
            at_nest: false,
            nest_pos: Vec2::new(0.0, 0.0),
            colony_food: 0.0,
            food_piles: 0,
            multi_food_wall_context: false,
            near_food_wall_pocket: false,
            nearby_food: Vec::new(),
            nearby_ants: Vec::new(),
            gradient_to_food: Vec2::ZERO,
            gradient_alarm: Vec2::ZERO,
            gradient_food_smell: Vec2::ZERO,
            gradient_repellent: Vec2::ZERO,
            food_here: 0.0,
            food_smell_here: 0.0,
            repellent_here: 0.0,
            wall_ahead: false,
            has_walls: false,
            sensor_left: sample(),
            sensor_center: sample(),
            sensor_right: sample(),
            tick: 11,
            spawn_cooldown_ticks: 30,
            soldier_ratio: 0.05,
            colony_size: 100,
            max_colony_size: 2000,
            food_lay_strength: 3.0,
            food_sat_cap: 50.0,
            outbound_lay_threshold: 3.0,
            stuck_repel_strength: 3.0,
            since_state_change: 0,
            deposit_decay_horizon: 900,
        }
    }

    fn set_heading(actions: &[Action]) -> f32 {
        actions
            .iter()
            .find_map(|action| match action {
                Action::SetHeading(h) => Some(*h),
                Action::SetHeadingImmediate(h) => Some(*h),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn stale_food_trail_lays_repellent_and_turns_back() {
        let mut brain = WorkerBrain::default();
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(false);
        p.self_heading = 0.0;
        p.food_here = STALE_TRAIL_INTENSITY;
        p.sensor_center.food = STALE_TRAIL_INTENSITY;

        let mut last_actions = Vec::new();
        for _ in 0..STALE_TRAIL_LIMIT {
            last_actions = brain.decide(&p, &mut rng);
        }

        let laid_repellent = last_actions.iter().any(|action| {
            matches!(
                action,
                Action::LayPheromone {
                    channel: PheromoneChannel::Repellent,
                    strength,
                } if (*strength - p.stuck_repel_strength * STALE_TRAIL_REPELLENT_MULT).abs() < 0.001
            )
        });
        assert!(laid_repellent);

        let h = set_heading(&last_actions).rem_euclid(TAU);
        assert!(h > 2.6 && h < 3.7);
    }

    #[test]
    fn carrying_without_home_sensor_signal_ignores_nest_position() {
        let mut rng = SmallRng::seed_from_u64(0);

        let mut p_left = base_perception(true);
        p_left.nest_pos = Vec2::new(10_000.0, 100.0);
        let h_left = set_heading(&WorkerBrain::default().decide(&p_left, &mut rng));

        let mut p_right = base_perception(true);
        p_right.nest_pos = Vec2::new(-10_000.0, 100.0);
        let h_right = set_heading(&WorkerBrain::default().decide(&p_right, &mut rng));

        assert_eq!(h_left.to_bits(), h_right.to_bits());
    }

    #[test]
    fn food_trail_sensor_choice_avoids_repellent_branch() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(false);
        p.self_heading = 0.0;
        p.sensor_left.food = 10.0;
        p.sensor_left.repellent = 10.0;
        p.sensor_right.food = 6.0;

        let h = set_heading(&WorkerBrain::default().decide(&p, &mut rng));

        assert!((h - JB_TURN_PER_TICK).abs() < 0.001);
    }

    #[test]
    fn lost_carrier_stops_laying_food_after_bootstrap_window() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(true);
        p.self_heading = 0.0;
        p.gradient_food_smell = -Vec2::X;
        p.since_state_change = CARRIER_BOOTSTRAP_TICKS + 1;

        let actions = WorkerBrain::default().decide(&p, &mut rng);

        assert!(!actions.iter().any(|action| matches!(
            action,
            Action::LayPheromone {
                channel: PheromoneChannel::Food,
                ..
            }
        )));
    }

    #[test]
    fn carrier_on_home_signal_lays_food() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(true);
        p.since_state_change = CARRIER_BOOTSTRAP_TICKS + 1;
        p.deposit_decay_horizon = CARRIER_BOOTSTRAP_TICKS + 500;
        p.sensor_center.home = HOME_SIGNAL_DEPOSIT_FLOOR;

        let actions = WorkerBrain::default().decide(&p, &mut rng);

        assert!(actions.iter().any(|action| matches!(
            action,
            Action::LayPheromone {
                channel: PheromoneChannel::Food,
                ..
            }
        )));
    }

    #[test]
    fn lost_carrier_uses_food_smell_as_away_signal() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut brain = WorkerBrain::default();
        let mut p = base_perception(true);
        p.self_heading = PI;
        p.nest_pos = p.self_pos;
        p.pickup_home_dist = CARRIER_SEARCH_FAR_PICKUP_DIST + 60.0;
        p.gradient_food_smell = Vec2::X;
        p.food_smell_here = 8.0;

        let mut h = p.self_heading;
        for tick in 0..90 {
            p.tick = tick;
            p.self_heading = h;
            h = set_heading(&brain.decide(&p, &mut rng));
        }

        let heading = Vec2::new(h.cos(), h.sin());
        assert!(heading.dot(Vec2::X) < -0.4);
    }

    #[test]
    fn outbound_food_smell_cuts_across_curved_food_trail() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(false);
        p.self_heading = 0.0;
        p.sensor_center.food = 10.0;
        p.gradient_food_smell = Vec2::Y;

        let h = set_heading(&WorkerBrain::default().decide(&p, &mut rng));

        assert!(h > 1.4 && h < 1.6);
    }

    #[test]
    fn outbound_no_sensor_signal_prefers_food_trail_gradient_over_smell() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut p = base_perception(false);
        p.self_heading = 0.0;
        p.gradient_to_food = Vec2::Y;
        p.gradient_food_smell = Vec2::X;

        let h = set_heading(&WorkerBrain::default().decide(&p, &mut rng));

        assert!(h > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Soldier
// ---------------------------------------------------------------------------

/// Soldier behavior, in priority order:
///   1. Hostile in range → engage closest (lay Alarm).
///   2. Alarm signal nearby → rush toward it.
///   3. On an active Food trail within patrol_radius → drift along the trail
///      (escort foragers like Atta/Camponotus soldiers do in real life).
///   4. Far from nest (>patrol_radius) → head home.
///   5. Else → idle wander near the nest.
pub struct SoldierBrain {
    pub patrol_radius: f32,
}

impl Default for SoldierBrain {
    fn default() -> Self {
        // Big enough to follow a foraging corridor partway out, but bounded
        // so they don't wander to the corners of the world.
        Self {
            patrol_radius: 300.0,
        }
    }
}

impl Brain for SoldierBrain {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action> {
        // 1) Hostile in range — engage.
        let hostile = p
            .nearby_ants
            .iter()
            .filter(|a| a.colony != p.self_colony)
            .min_by(|a, b| {
                a.pos
                    .distance_squared(p.self_pos)
                    .partial_cmp(&b.pos.distance_squared(p.self_pos))
                    .unwrap()
            });
        if let Some(t) = hostile {
            let d = t.pos.distance(p.self_pos);
            if d < 3.0 {
                return vec![
                    Action::Attack { target_id: t.id },
                    Action::LayPheromone {
                        channel: PheromoneChannel::Alarm,
                        strength: 2.0,
                    },
                ];
            }
            let target = heading_of((t.pos - p.self_pos).normalize_or_zero());
            return vec![Action::SetHeading(target), Action::Forward(1.5)];
        }

        // 2) Alarm signal — rush toward it (recruitment to nest-mate in distress).
        if p.gradient_alarm.length_squared() > 0.0 {
            let target = heading_of(p.gradient_alarm);
            return vec![Action::SetHeading(target), Action::Forward(1.3)];
        }

        let to_nest = p.nest_pos - p.self_pos;
        let dist_to_nest = to_nest.length();

        // 4) Too far — drift back toward the nest.
        if dist_to_nest > self.patrol_radius {
            let target = heading_of(to_nest.normalize_or_zero());
            return vec![
                Action::SetHeading(blend_angle(p.self_heading, target, 0.25)),
                Action::Forward(1.0),
            ];
        }

        // 3) On or near a Food trail — escort it (follow gradient outward).
        // Soldier prefers the OUTWARD trail direction (same nest-rejection
        // rule workers use) so they don't all crowd at the nest end.
        let away_from_nest = -to_nest.normalize_or_zero();
        let g = p.gradient_to_food;
        let trail_outward = g.length_squared() > 0.0 && g.dot(away_from_nest) > 0.0;
        if trail_outward && (p.food_here > 1.0 || p.food_smell_here > 0.5) {
            let target = heading_of(g);
            let jitter = rng.gen_range(-0.15..0.15);
            return vec![
                Action::SetHeading(blend_angle(p.self_heading, target + jitter, 0.2)),
                Action::Forward(0.9),
            ];
        }

        // 5) No trail in sight, within patrol radius — idle wander.
        let jitter = rng.gen_range(-0.3..0.3);
        vec![
            Action::SetHeading(p.self_heading + jitter),
            Action::Forward(0.7),
        ]
    }
}

// ---------------------------------------------------------------------------
// Queen
// ---------------------------------------------------------------------------

/// Stationary. Spawns ants on a cooldown, costs 1 stored food per spawn.
/// Both the cooldown and the worker/soldier mix are read from `Perception`
/// each tick (sourced from `SimConfig`) so they can be tuned at runtime.
pub struct QueenBrain {
    pub ticks_since_spawn: u32,
}

impl Default for QueenBrain {
    fn default() -> Self {
        Self {
            ticks_since_spawn: 0,
        }
    }
}

impl Brain for QueenBrain {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action> {
        self.ticks_since_spawn += 1;
        if self.ticks_since_spawn >= p.spawn_cooldown_ticks
            && p.colony_food >= 1.0
            && p.colony_size < p.max_colony_size
        {
            self.ticks_since_spawn = 0;
            let role = if rng.gen::<f32>() < p.soldier_ratio {
                Role::Soldier
            } else {
                Role::Worker
            };
            return vec![Action::Spawn { role }];
        }
        vec![Action::Idle]
    }
}
