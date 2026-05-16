use crate::brain::{Action, Brain, ForwardSample, NearbyAnt, Perception, PheromoneChannel};
use crate::brains::{QueenBrain, SoldierBrain, WorkerBrain};
use crate::entities::{Ant, Corpse, EntityId, Food, Nest, Role};
use crate::pheromone::PheromoneField;
use glam::Vec2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f32::consts::TAU;

const PERCEPTION_RADIUS: f32 = 14.0;
// Coarser pheromone cells (8 world units instead of 4) so the field stays
// at ~32 K cells for the larger 1920×1080 world. Same bandwidth as before.
const PHEROMONE_CELL: f32 = 8.0;
const PICKUP_RADIUS: f32 = 3.0;
/// johnBuffer-style 3-sensor probe geometry. Sensors sit at SENSOR_DIST
/// units ahead of the ant, fanned out by ±SENSOR_HALF_ANGLE radians.
const SENSOR_DIST: f32 = 24.0;
const SENSOR_HALF_ANGLE: f32 = 0.6; // ≈ 34°
const ATTACK_RADIUS: f32 = 3.0;
const BREADCRUMB_MIN_DIST: f32 = 14.0;
const MAX_BREADCRUMBS: usize = 500;
const RETURN_WAYPOINT_RADIUS: f32 = 16.0;
const RETURN_TURN_BLEND: f32 = 0.65;
const RETURN_ROUTE_MIN_DIRECT_DIST: f32 = 400.0;
const RETURN_ROUTE_MIN_EXCESS: f32 = 1.08;
const RETURN_ROUTE_MIN_DEVIATION: f32 = 45.0;
const PICKUP_TURN_OFFSET: f32 = 0.75;
const RETURN_ROUTE_HEADING_WEAVE: f32 = 0.75;
const CARRIER_DIRECT_HOME_GUARD_TICKS: u32 = 180;
const CARRIER_DIRECT_HOME_MIN_DIST: f32 = 80.0;
const CARRIER_DIRECT_HOME_MAX_DIST: f32 = 600.0;
const CARRIER_DIRECT_HOME_DOT: f32 = 0.65;
const CARRIER_FORBIDDEN_HOME_DOT: f32 = 0.92;
const CARRIER_PICKUP_SEARCH_TICKS: u32 = 90;
const CARRIER_PICKUP_SEARCH_TURN: f32 = 0.04;
const CARRIER_DIRECT_HOME_AVOID_BLEND: f32 = 0.88;
const CARRIER_NO_GPS_PICKUP_MIN_DIST: f32 = 300.0;
const CARRIER_NO_GPS_HOME_MIN_DIST: f32 = 90.0;
const CARRIER_NO_GPS_HOME_DOT: f32 = 0.75;
const CARRIER_NO_GPS_RETURN_ANGLE: f32 = 0.85;
const CARRIER_NO_GPS_WEAVE_PERIOD: u32 = 24;
const CARRIER_NO_GPS_BLEND: f32 = 0.90;
// Obstacle maps need tighter trails. Scaling only when walls exist raises
// wall-route quality without changing open-field arc/multi/food-cycle scores.
const WALL_TRAIL_LAY_SCALE: f32 = 0.8;
const NEAR_DEATH_HP_DROP_THRESHOLD: f32 = 0.05;
const NEAR_DEATH_ENERGY_DROP_THRESHOLD: f32 = 0.01;
const NEAR_DEATH_AGE_DROP_TICKS: u32 = 300;

fn blend_angle(from: f32, to: f32, t: f32) -> f32 {
    let mut d = (to - from) % TAU;
    if d > std::f32::consts::PI {
        d -= TAU;
    } else if d < -std::f32::consts::PI {
        d += TAU;
    }
    from + d * t
}

fn rotate_vec(v: Vec2, angle: f32) -> Vec2 {
    let (sin, cos) = angle.sin_cos();
    Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
}

fn dist2_to_seg(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let denom = ab.length_squared();
    if denom <= f32::EPSILON {
        return p.distance_squared(a);
    }
    let t = ((p - a).dot(ab) / denom).clamp(0.0, 1.0);
    (p - (a + ab * t)).length_squared()
}

fn route_is_non_direct(points: &[Vec2], nest_pos: Vec2, end_pos: Vec2) -> bool {
    if points.len() < 3 {
        return false;
    }
    let direct = end_pos.distance(nest_pos).max(1.0);
    if direct < RETURN_ROUTE_MIN_DIRECT_DIST {
        return false;
    }
    let mut length = 0.0;
    let mut max_deviation = 0.0_f32;
    for pair in points.windows(2) {
        length += pair[0].distance(pair[1]);
    }
    for point in points {
        max_deviation = max_deviation.max(dist2_to_seg(*point, nest_pos, end_pos).sqrt());
    }
    length / direct >= RETURN_ROUTE_MIN_EXCESS || max_deviation >= RETURN_ROUTE_MIN_DEVIATION
}

/// Runtime-tweakable sim parameters. The frontend pushes updates via WS
/// commands; nothing here requires a restart.
#[derive(Clone)]
pub struct SimConfig {
    /// How many sim steps to run per server-loop tick. 1 = real-time, 10/100/1000
    /// = fast-forward. Snapshots still ship at the server-loop rate.
    pub speed_mult: u32,
    /// If true, periodically drop a new food pile (up to `max_food_piles`).
    pub food_respawn: bool,
    /// Ticks between respawn attempts.
    pub food_respawn_interval_ticks: u32,
    /// Amount placed in each respawned pile.
    pub food_respawn_amount: f32,
    /// Cap on simultaneous food piles in the world.
    pub max_food_piles: usize,
    /// Ticks between queen spawn attempts. 30 = 1 spawn/sec at 30Hz.
    pub spawn_cooldown_ticks: u32,
    /// Probability that a new spawn is a soldier (vs worker). 0..1.
    pub soldier_ratio: f32,
    /// Soft cap on colony size — queen pauses spawning at or above this.
    pub max_colony_size: u32,
    /// If true, sim steps are skipped this server tick (snapshots still ship).
    pub paused: bool,
    /// Lab-stable mode: no ants die, queen doesn't spawn, energy doesn't
    /// drain. Used by the bench harness so cost functions can converge.
    pub stable_mode: bool,
    /// Per-ant trail lay strength before time-decay.
    pub food_lay_strength: f32,
    /// Food-channel saturation cap — when an ant is on a cell with
    /// Food > this, it stops depositing (prevents over-saturation).
    pub food_sat_cap: f32,
    /// Multiplier on the tiny nest entrance marker. This is not a long-range
    /// beacon; Home trails are ant-laid.
    pub home_emission_mult: f32,
    /// Outbound ants lay Home only on an established Food trail, or near real
    /// FoodSmell. This prevents a map-wide home flood.
    pub outbound_lay_threshold: f32,
    /// Strength of no-entry Repellent from stale trails and depleted piles.
    pub stuck_repel_strength: f32,
    /// true = bilinear splat; false = single-cell deposit.
    pub bilinear_deposit: bool,
    /// johnBuffer time-decay deposit horizon (ticks). Deposit strength
    /// = `food_lay_strength × max(0, 1 - since_state_change / horizon)`.
    /// Should roughly match a one-way trip duration so the ant lays
    /// strong material near its source (food or nest) and tapers off
    /// before reaching the other end.
    pub deposit_decay_horizon: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            speed_mult: 1,
            // Respawn OFF by default — start with a barren world; user drops
            // food manually with the + food tool, or toggles respawn on.
            food_respawn: false,
            food_respawn_interval_ticks: 300, // ~10s of real time at 30Hz
            food_respawn_amount: 60.0,
            max_food_piles: 12,
            // 1 spawn/sec at 1× speed. For a 1000-worker colony with
            // ~9000-tick lifespan, the natural death rate is ~0.11/sec, so
            // the queen has plenty of headroom and pauses at max_colony_size.
            spawn_cooldown_ticks: 30,
            soldier_ratio: 0.05,
            stable_mode: false,
            // Hard ceiling. With cooldown=30 and max_age≈33k, the natural
            // equilibrium is ~1000–1200 — this is just a safety cap so a
            // tweaked cooldown can't blow up to infinity.
            max_colony_size: 2000,
            paused: false,
            food_lay_strength: 1.5,
            food_sat_cap: 50.0,
            home_emission_mult: 1.0, // multiplier for the tiny nest entrance marker; Home trails are ant-laid.
            outbound_lay_threshold: 0.5,
            stuck_repel_strength: 3.0,
            // Tightness levers. Single-cell trail deposits won the topology
            // bench because bilinear splats created disconnected side-branches
            // and visible pheromone fog.
            bilinear_deposit: false,
            // 1500 ticks ≈ 1.25× one-way trip in wall_test (~1200u). Long
            // enough that freshness covers the whole route, short enough
            // that the gradient stays informative (still strong near source,
            // faint near destination).
            deposit_decay_horizon: 1500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_breadcrumb_route_is_not_replayed() {
        let nest = Vec2::new(0.0, 0.0);
        let food = Vec2::new(100.0, 0.0);
        let points = [nest, Vec2::new(50.0, 1.0), food];

        assert!(!route_is_non_direct(&points, nest, food));
    }

    #[test]
    fn curved_breadcrumb_route_can_be_replayed() {
        let nest = Vec2::new(0.0, 0.0);
        let food = Vec2::new(500.0, 0.0);
        let points = [nest, Vec2::new(250.0, 160.0), food];

        assert!(route_is_non_direct(&points, nest, food));
    }

    #[test]
    fn near_death_carrier_drops_food_before_dying() {
        let mut world = World::new(640.0, 480.0);
        world.ants.clear();
        world.brains.clear();
        world.food.clear();
        world.next_id = 1;
        world.config.spawn_cooldown_ticks = 999_999_999;

        let id = world.spawn_ant(Vec2::new(500.0, 400.0), Role::Worker, 0);
        let ant = world
            .ants
            .iter_mut()
            .find(|ant| ant.id == id)
            .expect("spawned worker missing");
        ant.carrying_food = true;
        ant.hp = NEAR_DEATH_HP_DROP_THRESHOLD * 0.5;

        world.step();

        let ant = world
            .ants
            .iter()
            .find(|ant| ant.id == id)
            .expect("near-death worker should still be alive");
        assert!(!ant.carrying_food);
        assert_eq!(world.food.len(), 1);
        assert_eq!(world.food[0].amount, 1.0);
    }
}

pub struct World {
    pub width: f32,
    pub height: f32,
    pub ants: Vec<Ant>,
    pub brains: HashMap<EntityId, Box<dyn Brain>>,
    pub food: Vec<Food>,
    pub corpses: Vec<Corpse>,
    pub nest: Nest,
    pub pheromones: PheromoneField,
    pub tick: u32,
    pub config: SimConfig,
    /// Cumulative count of food units dropped at the nest. Survives the
    /// queen's consumption (which only modifies `food_stored`), so it's the
    /// true "deliveries since spawn" counter used by the bench harness.
    pub food_delivered_total: u32,
    /// Bench/observability counters. Reset on scenario load.
    pub corpse_spawned_total: u32,
    pub corpse_decomposed_total: u32,
    pub corpse_pickup_total: u32,
    pub poison_deaths_total: u32,
    pub poison_kill_ticks_sum: u64,
    /// Break-out (stuck-loop escape) event count.
    pub stuck_escapes_total: u32,
    pub walls: Vec<bool>,
    pub has_walls: bool,
    pub wall_cols: usize,
    pub wall_rows: usize,
    pub wall_cell_size: f32,
    /// Spatial hash for ant neighbor queries. Rebuilt every tick. Stores
    /// indices into `self.ants` per cell. Essential at high N — otherwise
    /// perception is O(N²) and 1000-ant runs choke at 30 Hz.
    spatial: Vec<Vec<usize>>,
    spatial_cols: usize,
    spatial_rows: usize,
    spatial_cell: f32,
    next_id: EntityId,
    rng: SmallRng,
}

impl World {
    pub fn new(width: f32, height: f32) -> Self {
        let wall_cell_size = PHEROMONE_CELL;
        let wall_cols = (width / wall_cell_size).ceil() as usize;
        let wall_rows = (height / wall_cell_size).ceil() as usize;
        // Spatial-grid cell size = perception radius, so each query touches
        // a 3x3 neighborhood at most.
        let spatial_cell = PERCEPTION_RADIUS;
        let spatial_cols = (width / spatial_cell).ceil() as usize;
        let spatial_rows = (height / spatial_cell).ceil() as usize;
        let mut w = Self {
            width,
            height,
            ants: Vec::with_capacity(1200),
            brains: HashMap::with_capacity(1200),
            food: Vec::new(),
            corpses: Vec::new(),
            nest: Nest {
                pos: Vec2::new(width * 0.5, height * 0.5),
                radius: 22.0,
                food_stored: 0.0,
                queen_id: None,
            },
            pheromones: PheromoneField::new(width, height, PHEROMONE_CELL),
            tick: 0,
            config: SimConfig::default(),
            food_delivered_total: 0,
            corpse_spawned_total: 0,
            corpse_decomposed_total: 0,
            corpse_pickup_total: 0,
            poison_deaths_total: 0,
            poison_kill_ticks_sum: 0,
            stuck_escapes_total: 0,
            walls: vec![false; wall_cols * wall_rows],
            has_walls: false,
            wall_cols,
            wall_rows,
            wall_cell_size,
            spatial: vec![Vec::new(); spatial_cols * spatial_rows],
            spatial_cols,
            spatial_rows,
            spatial_cell,
            next_id: 1,
            rng: SmallRng::seed_from_u64(0xA17),
        };

        // Start with ~1000 ants. 970 workers + 30 soldiers.
        let queen_id = w.spawn_ant(w.nest.pos, Role::Queen, 0);
        w.nest.queen_id = Some(queen_id);
        for _ in 0..470 {
            let a = w.rng.gen_range(0.0..TAU);
            let r = w.rng.gen_range(0.0..(w.nest.radius * 0.6));
            w.spawn_ant(
                w.nest.pos + Vec2::new(a.cos(), a.sin()) * r,
                Role::Worker,
                0,
            );
        }
        for _ in 0..30 {
            w.spawn_ant(w.nest.pos, Role::Soldier, 0);
        }

        // No initial food piles — the world starts barren. User adds food
        // with the +food tool, right-click, the spawn-pile button, or by
        // enabling respawn.
        w
    }

    /// Replace the world's content with a named scenario setup. Keeps the
    /// SimConfig (so user-tuned sliders persist).
    pub fn load_scenario(&mut self, name: &str) {
        let prev_config = self.config.clone();
        let (w, h) = (self.width, self.height);
        *self = World::new(w, h);
        self.config = prev_config;
        match name {
            "arc_to_line" => self.setup_arc_to_line(),
            "wall_test" => self.setup_wall_test(),
            "fresh" => {} // already a fresh world
            _ => {}
        }
    }

    /// Wall-test scenario for ACO algorithm evaluation.
    ///
    /// Setup: nest on the left, one huge food pile on the right, a vertical
    /// wall between them with NO gap (ants must detour around top or bottom).
    /// Empty world; respawn off; no pesticide. Clean lab for measuring how
    /// fast a colony forms a stable trail around a barrier.
    ///
    /// SUCCESS CRITERIA:
    ///   1. Distinct yellow corridor forms around top or bottom edge within
    ///      ~3 min sim time at 1×.
    ///   2. Corridor is a stripe (<60 world units wide), not a haze.
    ///   3. food_stored grows monotonically once trail is established.
    ///   4. <30 % of ants pile against the wall surface.
    fn setup_wall_test(&mut self) {
        self.food.clear();
        self.corpses.clear();
        self.clear_walls();

        self.nest.pos = Vec2::new(self.width * 0.18, self.height * 0.5);
        let food_pos = Vec2::new(self.width * 0.82, self.height * 0.5);
        self.food.push(Food {
            pos: food_pos,
            amount: 10_000.0,
        });

        // Vertical wall in the middle. ~600 units tall, runs from y=240
        // down to y=840 in a 1080-tall world. Top gap = 240 px, bottom gap
        // = 240 px — both routes available, ants must discover one.
        let wall_x = self.width * 0.5;
        let wall_top = self.height * 0.22;
        let wall_bot = self.height * 0.78;
        let mut yy = wall_top;
        while yy <= wall_bot {
            self.paint_walls(wall_x, yy, 8.0, true);
            yy += 6.0;
        }

        // Reset ants, brains, pheromones, tick — clean slate.
        self.ants.clear();
        self.brains.clear();
        self.pheromones = PheromoneField::new(self.width, self.height, PHEROMONE_CELL);
        self.next_id = 1;
        self.tick = 0;

        let queen_id = self.spawn_ant(self.nest.pos, Role::Queen, 0);
        self.nest.queen_id = Some(queen_id);

        // Start workers at the nest. The challenge is now route discovery
        // from home, not random workers stumbling onto food from every part
        // of the map.
        for _ in 0..470 {
            let a = self.rng.gen_range(0.0..TAU);
            let r = self.rng.gen_range(0.0..(self.nest.radius * 0.6));
            self.spawn_ant(
                self.nest.pos + Vec2::new(a.cos(), a.sin()) * r,
                Role::Worker,
                0,
            );
        }
        for _ in 0..30 {
            self.spawn_ant(self.nest.pos, Role::Soldier, 0);
        }

        // Initial age + HP diversity. A real colony has workers at every
        // life stage and a small fraction recovering from foraging injuries
        // (failed predator escapes, fights, dehydration). ~10% wounded is
        // typical for an active above-ground colony.
        // Take ownership of rng briefly to avoid double-borrow.
        // Temporarily steal self.rng so we can both mutate ants and roll
        // randoms in the same loop. Replaced back at end of fn.
        let mut rng = std::mem::replace(&mut self.rng, SmallRng::seed_from_u64(0));
        for ant in self.ants.iter_mut() {
            if ant.role == Role::Queen {
                continue;
            }
            // Age: uniform across the first 70% of expected lifespan.
            ant.age = rng.gen_range(0..(ant.max_age * 7 / 10));
            // HP: 90% healthy, 10% wounded from prior foraging.
            if rng.gen::<f32>() < 0.10 {
                ant.hp = rng.gen_range(0.40..0.95);
            }
        }
        self.rng = rng;
    }

    /// Arc-to-line scenario (classic ACO demo): nest on one side, single
    /// food pile on the other, with a pre-painted CURVED Food-pheromone
    /// trail between them. Ants will follow the arc, the natural variation
    /// in their forward sampling creates shortcut routes, and over time the
    /// trail straightens through positive feedback.
    fn setup_arc_to_line(&mut self) {
        // Clear all existing food piles and the entire pheromone field.
        self.food.clear();
        // Move nest to the left side.
        self.nest.pos = Vec2::new(self.width * 0.18, self.height * 0.5);

        // Single food pile on the far right.
        let food_pos = Vec2::new(self.width * 0.82, self.height * 0.5);
        self.food.push(Food {
            pos: food_pos,
            amount: 5000.0,
        });

        // Despawn existing ants and respawn 1000 at the nest.
        let _ids: Vec<_> = self.ants.iter().map(|a| a.id).collect();
        self.ants.clear();
        self.brains.clear();
        self.pheromones = PheromoneField::new(self.width, self.height, PHEROMONE_CELL);
        self.next_id = 1;
        self.tick = 0;

        let queen_id = self.spawn_ant(self.nest.pos, Role::Queen, 0);
        self.nest.queen_id = Some(queen_id);
        for _ in 0..470 {
            self.spawn_ant(self.nest.pos, Role::Worker, 0);
        }
        for _ in 0..30 {
            self.spawn_ant(self.nest.pos, Role::Soldier, 0);
        }

        // Pre-paint an arc with strong Food pheromone (the bootstrap trail).
        // Quadratic Bezier from nest → food with a control point pulled up
        // so the curve bows substantially northward — visibly longer than
        // the straight line.
        let start = self.nest.pos;
        let end = food_pos;
        let mid = (start + end) * 0.5 + Vec2::new(0.0, -self.height * 0.30);
        let n_samples = 900;
        for i in 0..n_samples {
            let t = i as f32 / (n_samples - 1) as f32;
            let one_minus_t = 1.0 - t;
            let p =
                start * (one_minus_t * one_minus_t) + mid * (2.0 * one_minus_t * t) + end * (t * t);
            let tangent = (mid - start) * (2.0 * one_minus_t) + (end - mid) * (2.0 * t);
            let normal = Vec2::new(-tangent.y, tangent.x).normalize_or_zero();
            for offset in [-10.0, 0.0, 10.0] {
                self.pheromones
                    .deposit(PheromoneChannel::Food, p + normal * offset, 10.0);
            }
        }
    }

    fn rebuild_spatial(&mut self) {
        for v in &mut self.spatial {
            v.clear();
        }
        let cs = self.spatial_cell;
        for (idx, ant) in self.ants.iter().enumerate() {
            let c = ((ant.pos.x / cs) as usize).min(self.spatial_cols - 1);
            let r = ((ant.pos.y / cs) as usize).min(self.spatial_rows - 1);
            self.spatial[r * self.spatial_cols + c].push(idx);
        }
    }

    pub fn spawn_ant(&mut self, pos: Vec2, role: Role, colony: u8) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        let heading = self.rng.gen_range(0.0..TAU);
        // Lifespans in sim ticks (30 ticks/sec at 1× speed). Tuned so the
        // steady-state colony at cooldown=30 (1 spawn/sec) settles at ~1000:
        //   births/sec = deaths/sec  →  N = max_age / cooldown_ticks
        //   30000 / 30 = 1000 ants ✓
        // Variance per ant (±10%) so we don't get death waves where the
        // entire initial cohort expires at the same tick.
        let max_age = match role {
            Role::Queen => 5_000_000,
            Role::Worker => 30_000 + self.rng.gen_range(0..6_000),
            Role::Soldier => 60_000 + self.rng.gen_range(0..10_000),
        };
        self.ants.push(Ant {
            id,
            colony,
            role,
            pos,
            heading,
            target_heading: heading,
            energy: 1.0,
            hp: 1.0,
            carrying_food: false,
            pickup_home_dist: 0.0,
            age: 0,
            max_age,
            breadcrumbs: if pos.distance(self.nest.pos) <= self.nest.radius + 4.0 {
                vec![self.nest.pos]
            } else {
                Vec::new()
            },
            return_path: Vec::new(),
            first_poison_tick: None,
            // Fresh ants lay NOTHING until their first pickup/drop. Without
            // this, a colony of N ants all spawning at the nest would dump
            // max-strength Home pheromone simultaneously and flood the field.
            // u32::MAX → freshness clamps to 0 in the deposit formula.
            since_state_change: u32::MAX,
        });
        let brain: Box<dyn Brain> = match role {
            Role::Queen => Box::new(QueenBrain::default()),
            Role::Worker => Box::new(WorkerBrain::default()),
            Role::Soldier => Box::new(SoldierBrain::default()),
        };
        self.brains.insert(id, brain);
        id
    }

    /// Is the given world position inside a wall cell?
    pub fn wall_at(&self, pos: Vec2) -> bool {
        let c = (pos.x / self.wall_cell_size).floor() as i32;
        let r = (pos.y / self.wall_cell_size).floor() as i32;
        if c < 0 || r < 0 || c >= self.wall_cols as i32 || r >= self.wall_rows as i32 {
            return false;
        }
        self.walls[r as usize * self.wall_cols + c as usize]
    }

    /// Combined obstacle check used by ant movement + pheromone deposit.
    pub fn obstacle_at(&self, pos: Vec2) -> bool {
        // Walls block movement. Corpses do NOT — ants walk over their
        // dead nestmates freely (real-ant behavior), and a corpse
        // blocking movement also prevented ants from reaching the food
        // it'd become.
        self.wall_at(pos)
    }

    fn near_wall(&self, pos: Vec2, radius: f32) -> bool {
        const PROBES: [Vec2; 8] = [
            Vec2::new(1.0, 0.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, -1.0),
            Vec2::new(0.707, 0.707),
            Vec2::new(-0.707, 0.707),
            Vec2::new(0.707, -0.707),
            Vec2::new(-0.707, -0.707),
        ];
        PROBES.iter().any(|dir| self.wall_at(pos + *dir * radius))
    }

    fn heading_hits_wall(&self, pos: Vec2, heading: f32, max_dist: f32) -> bool {
        let dir = Vec2::new(heading.cos(), heading.sin());
        let step = self.wall_cell_size;
        let n = (max_dist / step).ceil() as u32;
        for i in 1..=n {
            let d = (i as f32 * step).min(max_dist);
            if self.wall_at(pos + dir * d) {
                return true;
            }
        }
        false
    }

    fn no_gps_carrier_heading(
        &self,
        ant_id: EntityId,
        pickup_home_dist: f32,
        pos: Vec2,
        h: f32,
    ) -> f32 {
        let to_nest_raw = self.nest.pos - pos;
        if pickup_home_dist < CARRIER_NO_GPS_PICKUP_MIN_DIST
            || to_nest_raw.length_squared() <= CARRIER_NO_GPS_HOME_MIN_DIST.powi(2)
        {
            return h;
        }
        let to_nest = to_nest_raw.normalize_or_zero();
        let heading = Vec2::new(h.cos(), h.sin());
        if to_nest.length_squared() <= 0.0 || heading.dot(to_nest) < CARRIER_NO_GPS_HOME_DOT {
            return h;
        }
        let phase = (self.tick / CARRIER_NO_GPS_WEAVE_PERIOD).max(1);
        let side = if (ant_id ^ phase) & 1 == 0 { 1.0 } else { -1.0 };
        let preferred = rotate_vec(to_nest, side * CARRIER_NO_GPS_RETURN_ANGLE);
        let fallback = rotate_vec(to_nest, -side * CARRIER_NO_GPS_RETURN_ANGLE);
        let preferred_h = preferred.y.atan2(preferred.x);
        let fallback_h = fallback.y.atan2(fallback.x);
        let woven_h = if !self.heading_hits_wall(pos, preferred_h, SENSOR_DIST * 4.0) {
            preferred_h
        } else if !self.heading_hits_wall(pos, fallback_h, SENSOR_DIST * 4.0) {
            fallback_h
        } else {
            h
        };
        blend_angle(h, woven_h, CARRIER_NO_GPS_BLEND)
    }

    fn avoid_blocked_home_heading(&self, ant_id: EntityId, pos: Vec2, h: f32) -> f32 {
        let heading = Vec2::new(h.cos(), h.sin());
        let to_nest = (self.nest.pos - pos).normalize_or_zero();
        if to_nest.length_squared() <= 0.0
            || heading.dot(to_nest) < 0.72
            || !self.heading_hits_wall(pos, h, SENSOR_DIST * 6.0)
        {
            return h;
        }
        let side = if (ant_id ^ self.tick) & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        let first = h + side * std::f32::consts::FRAC_PI_2;
        let second = h - side * std::f32::consts::FRAC_PI_2;
        if !self.heading_hits_wall(pos, first, SENSOR_DIST * 3.0) {
            first
        } else if !self.heading_hits_wall(pos, second, SENSOR_DIST * 3.0) {
            second
        } else {
            h
        }
    }

    /// Paint a disc of walls (or remove them) at world (`x`,`y`) with the
    /// given radius. Used by the UI brush.
    pub fn paint_walls(&mut self, x: f32, y: f32, radius: f32, value: bool) {
        let cs = self.wall_cell_size;
        let r2 = radius * radius;
        let c_min = (((x - radius) / cs).floor() as i32).max(0) as usize;
        let c_max = (((x + radius) / cs).ceil() as i32).min(self.wall_cols as i32) as usize;
        let r_min = (((y - radius) / cs).floor() as i32).max(0) as usize;
        let r_max = (((y + radius) / cs).ceil() as i32).min(self.wall_rows as i32) as usize;
        // Don't wall over the nest itself — would trap the queen.
        let nest_safe2 = (self.nest.radius + 4.0) * (self.nest.radius + 4.0);
        let mut changed = false;
        for r in r_min..r_max {
            for c in c_min..c_max {
                let cell_center = Vec2::new(c as f32 * cs + cs * 0.5, r as f32 * cs + cs * 0.5);
                if cell_center.distance_squared(Vec2::new(x, y)) <= r2
                    && cell_center.distance_squared(self.nest.pos) > nest_safe2
                {
                    let i = r * self.wall_cols + c;
                    if self.walls[i] != value {
                        self.walls[i] = value;
                        changed = true;
                    }
                }
            }
        }
        if changed {
            self.has_walls = value || self.walls.iter().any(|wall| *wall);
        }
    }

    pub fn clear_walls(&mut self) {
        for w in &mut self.walls {
            *w = false;
        }
        self.has_walls = false;
    }

    /// Drop a new food pile somewhere away from the nest. Returns true on success.
    pub fn place_food_pile(&mut self, amount: f32) -> bool {
        if self.food.len() >= self.config.max_food_piles {
            return false;
        }
        for _ in 0..30 {
            let x = self.rng.gen_range(20.0..(self.width - 20.0));
            let y = self.rng.gen_range(20.0..(self.height - 20.0));
            let pos = Vec2::new(x, y);
            if pos.distance(self.nest.pos) > 200.0 {
                self.add_food_at(pos, amount);
                return true;
            }
        }
        false
    }

    pub fn add_food_at(&mut self, pos: Vec2, amount: f32) {
        self.pheromones
            .clear_region(PheromoneChannel::FoodSmell, pos, 70.0);
        self.pheromones
            .clear_region(PheromoneChannel::Repellent, pos, 70.0);
        self.food.push(Food { pos, amount });
    }

    /// True while the colony is functionally alive. Two ways to die:
    ///   - queen dead (no replacement possible), OR
    ///   - no workers AND no soldiers remaining (queen alone, no foragers,
    ///     no future food → equivalent to death).
    pub fn is_running(&self) -> bool {
        if self.nest.queen_id.is_none() {
            return false;
        }
        self.ants.iter().any(|a| a.role != Role::Queen)
    }

    pub fn step(&mut self) {
        if !self.is_running() {
            return;
        }
        self.tick += 1;

        // Phase 0: rebuild spatial hash so perception is O(1) per ant.
        self.rebuild_spatial();

        // Phase 1: build perceptions (immutable on world) — parallelized
        // via rayon. Each ant's perception is independent: reads ants /
        // food / nest / pheromones / walls, but mutates nothing. The
        // per-(ant,tick) RNG seed makes forward sampling deterministic
        // without any shared mutable state.
        use rayon::prelude::*;
        let perceptions: Vec<(EntityId, Perception)> = self
            .ants
            .par_iter()
            .map(|a| (a.id, self.perceive(a)))
            .collect();

        // Phase 2: brains decide.
        let mut all_actions: Vec<(EntityId, Vec<Action>)> = Vec::with_capacity(perceptions.len());
        for (id, perception) in perceptions {
            if let Some(brain) = self.brains.get_mut(&id) {
                let actions = brain.decide(&perception, &mut self.rng);
                all_actions.push((id, actions));
            }
        }

        // Phase 3: apply actions, mutate world.
        for (id, actions) in all_actions {
            for action in actions {
                self.apply_action(id, action);
            }
        }

        // Phase 4: world bookkeeping.

        // Food pile FoodSmell — scale rises with the SQUARE of pile size,
        // so a single dead-ant emits a tiny minute smell while a fresh
        // pile creates a local plume. Squared shape (vs linear) lets small
        // piles be barely-perceptible without being entirely silent.
        //   amount=1   → scale 0.0004 → peak ~0.08 (a few cells of range)
        //   amount=10  → scale 0.04   → peak ~8    (moderate range)
        //   amount=50+ → scale 1.0    → strong local plume
        for food in &self.food {
            let scale = (food.amount / 50.0).powi(2).clamp(0.0, 1.0);
            self.pheromones
                .deposit(PheromoneChannel::FoodSmell, food.pos, 12.0 * scale);
            for (dx, dy) in &[
                (8.0, 0.0),
                (-8.0, 0.0),
                (0.0, 8.0),
                (0.0, -8.0),
                (8.0, 8.0),
                (-8.0, 8.0),
                (8.0, -8.0),
                (-8.0, -8.0),
                (16.0, 0.0),
                (-16.0, 0.0),
                (0.0, 16.0),
                (0.0, -16.0),
            ] {
                let pos = glam::Vec2::new(food.pos.x + dx, food.pos.y + dy);
                self.pheromones
                    .deposit(PheromoneChannel::FoodSmell, pos, 4.0 * scale);
            }
        }

        // Home channel is now an ANT-LAID trail (outbound ants deposit
        // with time-decay), not a nest beacon. Real-ant biology: there
        // is no long-range home-pheromone halo — direction comes from
        // the trail itself. Tiny entrance marker only, single-cell, so
        // ants can lock onto "this is the nest cell" on final approach.
        {
            let np = self.nest.pos;
            let m = self.config.home_emission_mult;
            self.pheromones.deposit(PheromoneChannel::Home, np, 8.0 * m);
            for (dx, dy) in &[
                (8.0, 0.0),
                (-8.0, 0.0),
                (0.0, 8.0),
                (0.0, -8.0),
                (8.0, 8.0),
                (-8.0, 8.0),
                (8.0, -8.0),
                (-8.0, -8.0),
                (16.0, 0.0),
                (-16.0, 0.0),
                (0.0, 16.0),
                (0.0, -16.0),
            ] {
                self.pheromones
                    .deposit(PheromoneChannel::Home, np + Vec2::new(*dx, *dy), 2.0 * m);
            }
        }

        self.pheromones.decay_step(&self.walls);
        self.maybe_respawn_food();

        // Energy / age update. In stable_mode (bench), nothing decays —
        // ants live forever, no consumption, no births.
        for ant in &mut self.ants {
            ant.age += 1;
            ant.since_state_change = ant.since_state_change.saturating_add(1);
            if !self.config.stable_mode {
                ant.energy -= if ant.role == Role::Queen {
                    0.00002
                } else {
                    0.00005
                };
            }
        }

        // Trophallaxis: ants inside the nest get fed by nest-mates (in real
        // ants this is mouth-to-mouth food sharing — the colony's social
        // safety net). Free refill — the queen's food consumption is the
        // colony's only real food drain.
        let nest_pos = self.nest.pos;
        let nest_rad = self.nest.radius;
        for ant in &mut self.ants {
            if ant.pos.distance(nest_pos) <= nest_rad {
                ant.energy = 1.0;
                if !ant.carrying_food {
                    ant.breadcrumbs.clear();
                    ant.breadcrumbs.push(nest_pos);
                    ant.return_path.clear();
                }
            }
        }

        // Pesticide-cloud damage: ants in cells with very high Repellent
        // concentration take ongoing HP damage. Wall-collision and
        // stale-trail repellent stay well below this threshold, so only
        // actual pesticide poisons. Queen is immune to pesticide.
        const POISON_THRESHOLD: f32 = 20.0;
        // Dose-dependent poison. Even at full strength the ant takes a
        // LONG time to die — ~500 ticks (~17 sec sim at 30Hz) — so you
        // can watch them stagger, lose orientation, and slowly succumb.
        // Half-dose ~1000 ticks. Below threshold: no damage.
        const POISON_DAMAGE_FULL: f32 = 0.002;
        const MAX_POISON: f32 = 40.0;
        for ant in &mut self.ants {
            if ant.role == Role::Queen {
                continue;
            }
            let r = self.pheromones.sample(PheromoneChannel::Repellent, ant.pos);
            if r > POISON_THRESHOLD {
                if ant.first_poison_tick.is_none() {
                    ant.first_poison_tick = Some(self.tick);
                }
                let dose =
                    ((r - POISON_THRESHOLD) / (MAX_POISON - POISON_THRESHOLD)).clamp(0.0, 1.0);
                ant.hp -= POISON_DAMAGE_FULL * dose;
            }
        }

        // Crowd-Repellent: ants in a dense local cluster deposit a small
        // amount of Repellent.
        // EXCLUDES legitimate clusters near food piles and the nest —
        // those should not light up red. This surfaces genuinely stuck
        // congregations (against a wall, in a dead-end pheromone basin).
        const CROWD_R: f32 = 12.0;
        const CROWD_N: usize = 5;
        const CROWD_DEPOSIT: f32 = 0.10;
        let crowd_r = CROWD_R;
        let crowd_n = CROWD_N;
        let crowd_deposit = CROWD_DEPOSIT;
        let crowd_r2 = crowd_r * crowd_r;
        const LEGIT_CLUSTER_R: f32 = 60.0; // bigger zone around food/nest
        const LEGIT_R2: f32 = LEGIT_CLUSTER_R * LEGIT_CLUSTER_R;
        let cs = self.spatial_cell;
        let n_ants = self.ants.len();
        let nest_pos = self.nest.pos;
        // Snapshot food positions so we don't have to re-borrow inside the loop.
        let food_positions: Vec<Vec2> = self.food.iter().map(|f| f.pos).collect();
        let mut crowd_deposits: Vec<Vec2> = Vec::new();
        for idx in 0..n_ants {
            let p = self.ants[idx].pos;
            // Skip legitimate clusters: at nest or at a food pile.
            if p.distance_squared(nest_pos) < LEGIT_R2 {
                continue;
            }
            if food_positions
                .iter()
                .any(|fp| fp.distance_squared(p) < LEGIT_R2)
            {
                continue;
            }
            // No-conflicting-pheromones rule: skip crowd-Repellent if the
            // ant is anywhere with ANY noticeable path / smell signal —
            // crowding there is legitimate convergence, not a stuck cluster.
            // Lowered thresholds: even a faint smell whiff (0.5) or trail
            // (1.0) is enough to disable the crowd marker.
            let home_v = self.pheromones.sample(PheromoneChannel::Home, p);
            let food_v = self.pheromones.sample(PheromoneChannel::Food, p);
            let smell_v = self.pheromones.sample(PheromoneChannel::FoodSmell, p);
            if home_v > 1.0 || food_v > 1.0 || smell_v > 0.5 {
                continue;
            }
            let c = ((p.x / cs) as i32).max(0).min(self.spatial_cols as i32 - 1);
            let r = ((p.y / cs) as i32).max(0).min(self.spatial_rows as i32 - 1);
            let mut count = 0usize;
            'outer: for dr in -1..=1 {
                let rr = r + dr;
                if rr < 0 || rr >= self.spatial_rows as i32 {
                    continue;
                }
                for dc in -1..=1 {
                    let cc = c + dc;
                    if cc < 0 || cc >= self.spatial_cols as i32 {
                        continue;
                    }
                    let bucket = &self.spatial[rr as usize * self.spatial_cols + cc as usize];
                    for &other in bucket {
                        if other == idx {
                            continue;
                        }
                        if self.ants[other].pos.distance_squared(p) < crowd_r2 {
                            count += 1;
                            if count >= crowd_n {
                                break 'outer;
                            }
                        }
                    }
                }
            }
            if count >= crowd_n {
                crowd_deposits.push(p);
            }
        }
        // Cap crowd-Repellent locally so it can never reach pesticide-
        // poison threshold (POISON_THRESHOLD=20). Crowd ≠ pesticide.
        const CROWD_REPEL_CAP: f32 = 5.0;
        for p in crowd_deposits {
            let cur = self.pheromones.sample(PheromoneChannel::Repellent, p);
            if cur < CROWD_REPEL_CAP {
                self.pheromones
                    .deposit(PheromoneChannel::Repellent, p, crowd_deposit);
            }
        }

        // Queen eats from stored food when hungry. Skipped in stable mode.
        if !self.config.stable_mode {
            if let Some(qid) = self.nest.queen_id {
                if let Some(qidx) = self.ants.iter().position(|a| a.id == qid) {
                    let e = self.ants[qidx].energy;
                    if e < 0.7 && self.nest.food_stored >= 1.0 {
                        self.nest.food_stored -= 1.0;
                        self.ants[qidx].energy = (e + 0.5).min(1.0);
                    }
                }
            }
        }
        // In stable_mode, we still want HP-deaths (so pesticide kills
        // are observable for the bench's pesticide metrics) but suppress
        // age/energy deaths. Reset those each tick.
        if self.config.stable_mode {
            for ant in &mut self.ants {
                ant.energy = 1.0;
                if ant.age >= ant.max_age.saturating_sub(1) {
                    ant.age = ant.max_age / 2; // far from death
                }
            }
        }
        let mut pre_death_drops = Vec::new();
        for ant in &mut self.ants {
            if !ant.carrying_food {
                continue;
            }
            let near_death = ant.hp <= NEAR_DEATH_HP_DROP_THRESHOLD
                || (!self.config.stable_mode
                    && (ant.energy <= NEAR_DEATH_ENERGY_DROP_THRESHOLD
                        || ant.age.saturating_add(NEAR_DEATH_AGE_DROP_TICKS) >= ant.max_age));
            if near_death {
                pre_death_drops.push(ant.pos);
                ant.carrying_food = false;
                ant.pickup_home_dist = 0.0;
                ant.return_path.clear();
                ant.breadcrumbs.clear();
            }
        }
        for pos in pre_death_drops {
            if !self.wall_at(pos) {
                self.food.push(Food { pos, amount: 1.0 });
            }
        }
        // Three death causes (HP only in stable_mode; age/energy too in
        // normal mode):
        //   - combat / damage: hp <= 0
        //   - starvation: energy <= 0 (rare in healthy colonies)
        //   - old age: age >= max_age (the dominant cause, by design)
        // Dead ants drop their cargo and ALSO leave a corpse at the death
        // spot. The corpse acts as a soft obstacle for a while, then
        // decomposes into a small food pile (~3 min sim time later).
        // Corpses take a LONG time to decompose — long enough that you
        // notice piles of dead ants left over from pesticide sprays.
        // 18000 ticks ≈ 10 min sim time at 1× speed.
        // Was 18000 (~10 min sim time) — way too slow. Reduced to 900
        // (~30 s) so corpses convert to food while workers are still
        // foraging in the area. Real ant corpses get moved/eaten within
        // hours; ours just had to keep up with bench/observation pacing.
        const CORPSE_DECOMPOSE_TICKS: u32 = 900;
        let dead: Vec<(EntityId, Vec2, bool, Role, bool)> = self
            .ants
            .iter()
            .filter(|a| a.hp <= 0.0 || a.energy <= 0.0 || a.age >= a.max_age)
            // poisoned = died by HP loss (combat/poison) rather than age/starvation
            .map(|a| (a.id, a.pos, a.carrying_food, a.role, a.hp <= 0.0))
            .collect();
        // Pesticide-kill metric: for ants dying with hp <= 0 who were
        // previously exposed to pesticide, accumulate (now − first_poison)
        // ticks. Average across all such kills = pesticide_kill_time.
        let cur_tick = self.tick;
        for a in &self.ants {
            if a.hp <= 0.0 {
                if let Some(first) = a.first_poison_tick {
                    let elapsed = cur_tick.saturating_sub(first);
                    self.poison_kill_ticks_sum += elapsed as u64;
                    self.poison_deaths_total += 1;
                }
            }
        }
        for (id, pos, was_carrying, role, poisoned) in &dead {
            if Some(*id) == self.nest.queen_id {
                self.nest.queen_id = None;
            }
            self.brains.remove(id);
            // Ants dying inside a wall vanish — no dropped food, no corpse.
            // (Carcass embedded in solid material isn't recoverable.)
            let died_in_wall = self.wall_at(*pos);
            if *was_carrying && !died_in_wall {
                self.food.push(Food {
                    pos: *pos,
                    amount: 1.0,
                });
            }
            // Only ~25 % of dead ants leave a corpse — real ant colonies
            // recycle most carcasses immediately via colony-mate
            // consumption / necrophoresis; only a fraction become external
            // food sources. Queens never leave one (would clog nest); ants
            // dying inside the nest are also cleaned up invisibly; ants
            // dying inside walls simply vanish.
            if *role != Role::Queen
                && !died_in_wall
                && pos.distance(self.nest.pos) > self.nest.radius + 6.0
                && self.rng.gen::<f32>() < 0.25
            {
                self.corpses.push(Corpse {
                    pos: *pos,
                    ticks_remaining: CORPSE_DECOMPOSE_TICKS,
                    poisoned: *poisoned,
                });
                self.corpse_spawned_total += 1;
            }
        }
        self.ants
            .retain(|a| a.hp > 0.0 && a.energy > 0.0 && a.age < a.max_age);

        // Tick corpses. When a corpse's timer hits 0 it decomposes into a
        // small food pile (nutrient recycling). Poisoned corpses keep
        // emitting Repellent so other ants steer around the danger zone.
        let mut corpse_repel: Vec<Vec2> = Vec::new();
        for c in &mut self.corpses {
            if c.ticks_remaining > 0 {
                c.ticks_remaining -= 1;
            }
            if c.poisoned && c.ticks_remaining > 0 {
                corpse_repel.push(c.pos);
            }
        }
        for p in corpse_repel {
            self.pheromones.deposit(PheromoneChannel::Repellent, p, 0.5);
        }
        let mut new_food: Vec<Food> = Vec::new();
        let mut decomposed = 0u32;
        self.corpses.retain(|c| {
            if c.ticks_remaining == 0 {
                decomposed += 1;
                // Poisoned corpses do NOT become food — the toxin is still
                // in the body and real ants avoid pesticide-killed carcasses.
                // Natural-death corpses convert to ~0.5 unit of food
                // (a worker can carry it in one trip).
                if !c.poisoned {
                    new_food.push(Food {
                        pos: c.pos,
                        amount: 0.5,
                    });
                }
                false
            } else {
                true
            }
        });
        self.corpse_decomposed_total += decomposed;
        self.food.extend(new_food);
    }

    fn maybe_respawn_food(&mut self) {
        if !self.config.food_respawn {
            return;
        }
        if self.tick % self.config.food_respawn_interval_ticks.max(1) != 0 {
            return;
        }
        if self.food.len() >= self.config.max_food_piles {
            return;
        }
        self.place_food_pile(self.config.food_respawn_amount);
    }

    fn perceive(&self, ant: &Ant) -> Perception {
        let r2 = PERCEPTION_RADIUS * PERCEPTION_RADIUS;

        let mut nearby_food: Vec<(Vec2, f32)> = self
            .food
            .iter()
            .filter(|f| f.pos.distance_squared(ant.pos) <= r2)
            .map(|f| (f.pos, f.amount))
            .collect();
        nearby_food.sort_by(|a, b| {
            a.0.distance_squared(ant.pos)
                .partial_cmp(&b.0.distance_squared(ant.pos))
                .unwrap()
        });

        // Spatial-grid nearest-neighbor query: visit own cell + 8 neighbors.
        let cs = self.spatial_cell;
        let c0 = (ant.pos.x / cs) as i32;
        let r0 = (ant.pos.y / cs) as i32;
        let mut nearby_ants: Vec<NearbyAnt> = Vec::new();
        for dr in -1..=1i32 {
            let rr = r0 + dr;
            if rr < 0 || rr >= self.spatial_rows as i32 {
                continue;
            }
            for dc in -1..=1i32 {
                let cc = c0 + dc;
                if cc < 0 || cc >= self.spatial_cols as i32 {
                    continue;
                }
                let bucket = &self.spatial[rr as usize * self.spatial_cols + cc as usize];
                for &other_idx in bucket {
                    let other = &self.ants[other_idx];
                    if other.id != ant.id && other.pos.distance_squared(ant.pos) <= r2 {
                        nearby_ants.push(NearbyAnt {
                            id: other.id,
                            pos: other.pos,
                            colony: other.colony,
                            role: other.role,
                        });
                    }
                }
            }
        }

        Perception {
            self_id: ant.id,
            self_pos: ant.pos,
            self_heading: ant.heading,
            self_colony: ant.colony,
            carrying_food: ant.carrying_food,
            pickup_home_dist: ant.pickup_home_dist,
            has_return_route: ant.carrying_food && !ant.return_path.is_empty(),
            at_nest: ant.pos.distance(self.nest.pos) <= self.nest.radius,
            nest_pos: self.nest.pos,
            colony_food: self.nest.food_stored,
            nearby_food,
            nearby_ants,
            gradient_to_food: self.pheromones.gradient(PheromoneChannel::Food, ant.pos),
            gradient_alarm: self.pheromones.gradient(PheromoneChannel::Alarm, ant.pos),
            gradient_food_smell: self
                .pheromones
                .gradient(PheromoneChannel::FoodSmell, ant.pos),
            gradient_repellent: self
                .pheromones
                .gradient(PheromoneChannel::Repellent, ant.pos),
            food_here: self.pheromones.sample(PheromoneChannel::Food, ant.pos),
            food_smell_here: self.pheromones.sample(PheromoneChannel::FoodSmell, ant.pos),
            repellent_here: self.pheromones.sample(PheromoneChannel::Repellent, ant.pos),
            wall_ahead: self.heading_hits_wall(ant.pos, ant.heading, SENSOR_DIST),
            sensor_left: self.sample_sensor(ant, -SENSOR_HALF_ANGLE),
            sensor_center: self.sample_sensor(ant, 0.0),
            sensor_right: self.sample_sensor(ant, SENSOR_HALF_ANGLE),
            tick: self.tick,
            spawn_cooldown_ticks: self.config.spawn_cooldown_ticks,
            soldier_ratio: self.config.soldier_ratio,
            colony_size: self.ants.len() as u32,
            max_colony_size: self.config.max_colony_size,
            food_lay_strength: if self.has_walls {
                self.config.food_lay_strength * WALL_TRAIL_LAY_SCALE
            } else {
                self.config.food_lay_strength
            },
            food_sat_cap: self.config.food_sat_cap,
            outbound_lay_threshold: self.config.outbound_lay_threshold,
            stuck_repel_strength: self.config.stuck_repel_strength,
            since_state_change: ant.since_state_change,
            deposit_decay_horizon: self.config.deposit_decay_horizon,
        }
    }

    fn ant_idx(&self, id: EntityId) -> Option<usize> {
        self.ants.iter().position(|a| a.id == id)
    }

    /// johnBuffer 3-sensor probe. Samples all relevant channels at a
    /// single fixed point offset from the ant by (heading + angle_offset).
    /// Ray-marches outward stopping at walls, so a wall blocks the sensor
    /// (sensor sits just before the wall instead of seeing through it).
    fn sample_sensor(&self, ant: &Ant, angle_offset: f32) -> ForwardSample {
        let a = ant.heading + angle_offset;
        let dir = Vec2::new(a.cos(), a.sin());
        // March in PHEROMONE_CELL-sized steps so a wall is never skipped.
        let step = PHEROMONE_CELL;
        let n = (SENSOR_DIST / step).ceil() as i32;
        let mut hit_dist = SENSOR_DIST;
        for s in 1..=n {
            let d = (s as f32 * step).min(SENSOR_DIST);
            if self.wall_at(ant.pos + dir * d) {
                hit_dist = (d - step).max(0.0);
                break;
            }
            hit_dist = d;
        }
        let pos = ant.pos + dir * hit_dist;
        ForwardSample {
            food: self.pheromones.sample(PheromoneChannel::Food, pos),
            repellent: self.pheromones.sample(PheromoneChannel::Repellent, pos),
            home: self.pheromones.sample(PheromoneChannel::Home, pos),
        }
    }

    fn apply_action(&mut self, id: EntityId, action: Action) {
        let Some(idx) = self.ant_idx(id) else {
            return;
        };
        match action {
            Action::SetHeading(h) => {
                // SetHeading now sets the COMMIT goal (target_heading). The
                // actual heading slews smoothly toward it via the PD-style
                // controller in Forward — gives the johnBuffer-style banked
                // turn look instead of snappy lerps.
                self.ants[idx].target_heading = h;
            }
            Action::Forward(speed) => {
                let s = speed.clamp(0.0, 3.0);
                if self.ants[idx].role == Role::Queen {
                    return;
                }
                let hp = self.ants[idx].hp;
                // PD-style continuous heading slew toward target_heading.
                {
                    let ant = &mut self.ants[idx];
                    let cur = Vec2::new(ant.heading.cos(), ant.heading.sin());
                    let tgt = Vec2::new(ant.target_heading.cos(), ant.target_heading.sin());
                    let perp = Vec2::new(-cur.y, cur.x);
                    let mut sin_align = tgt.dot(perp);
                    let cos_align = tgt.dot(cur);
                    if cos_align < 0.0 && sin_align.abs() < 0.05 {
                        sin_align = 0.3;
                    }
                    const ROT_RATE: f32 = 0.22;
                    ant.heading += (sin_align * 1.5).clamp(-ROT_RATE, ROT_RATE);
                }
                // Randomness scales with LOCAL pesticide concentration —
                // an ant standing in a heavy pesticide cloud lurches almost
                // randomly each tick (panic / disorientation). A whiff
                // produces only a small wobble. No pesticide → no kick.
                // Anti-correlated with HP only as a secondary effect.
                let pesticide_here = self
                    .pheromones
                    .sample(PheromoneChannel::Repellent, self.ants[idx].pos);
                if pesticide_here > 1.0 {
                    let intensity = (pesticide_here / 40.0).clamp(0.0, 1.0);
                    // ±intensity × π → at max pesticide, heading kick up
                    // to ±180° in a single tick (random walk).
                    let kick: f32 = self
                        .rng
                        .gen_range(-std::f32::consts::PI..std::f32::consts::PI)
                        * intensity;
                    self.ants[idx].heading += kick;
                } else if hp < 0.7 {
                    // Residual stagger from prior poisoning even after
                    // moving out of the cloud (toxin still in body).
                    let stagger = (0.7 - hp) * 0.3;
                    let kick: f32 = self.rng.gen_range(-stagger..stagger);
                    self.ants[idx].heading += kick;
                }
                // Speed: base from HP (poisoned ants stagger slower) ×
                // pheromone-path BOOST. When the ant is on a well-defined
                // trail (high local Home OR Food pheromone), it dashes
                // confidently — up to 1.5× normal speed. Off-trail =
                // normal speed (1.0×). Real ants on chemical trails are
                // visibly faster than wandering foragers.
                let hp_speed = (0.4 + 0.6 * hp).clamp(0.4, 1.0);
                let trail_here = {
                    let h_val = self
                        .pheromones
                        .sample(PheromoneChannel::Home, self.ants[idx].pos);
                    let f_val = self
                        .pheromones
                        .sample(PheromoneChannel::Food, self.ants[idx].pos);
                    h_val.max(f_val)
                };
                // Boost factor: 0 trail → 1.0×, peak trail (≥10) → 1.5×.
                let trail_boost = 1.0 + (trail_here / 10.0).clamp(0.0, 0.5);
                let s = s * hp_speed * trail_boost;
                // SOFT REPULSION from nearby ants. Each neighbor within
                // REPEL_R contributes a 1/d² force pushing this ant away.
                // The force is BLENDED with the ant's commanded heading
                // (doesn't override it). Net effect: ants navigate around
                // each other in dense traffic instead of stacking, while
                // still following their pheromone path. No deadlocks
                // because no hard blocking.
                const REPEL_R: f32 = 7.0; // ~2 ant-body-widths
                const REPEL_R2: f32 = REPEL_R * REPEL_R;
                const REPEL_STRENGTH: f32 = 0.35;
                let p_self = self.ants[idx].pos;
                let mut repel = Vec2::ZERO;
                let cs = self.spatial_cell;
                let cx = ((p_self.x / cs) as i32)
                    .max(0)
                    .min(self.spatial_cols as i32 - 1);
                let ry = ((p_self.y / cs) as i32)
                    .max(0)
                    .min(self.spatial_rows as i32 - 1);
                for dr in -1..=1_i32 {
                    let rr = ry + dr;
                    if rr < 0 || rr >= self.spatial_rows as i32 {
                        continue;
                    }
                    for dc in -1..=1_i32 {
                        let cc = cx + dc;
                        if cc < 0 || cc >= self.spatial_cols as i32 {
                            continue;
                        }
                        let bucket = &self.spatial[rr as usize * self.spatial_cols + cc as usize];
                        for &other in bucket {
                            if other == idx {
                                continue;
                            }
                            let delta = p_self - self.ants[other].pos;
                            let d2 = delta.length_squared();
                            if d2 > 0.0001 && d2 < REPEL_R2 {
                                repel += delta / d2; // 1/d² force (magnitude d/d² = 1/d)
                            }
                        }
                    }
                }
                let mut h = self.ants[idx].heading;
                if repel.length_squared() > 0.0 {
                    let move_dir = Vec2::new(h.cos(), h.sin());
                    let blended = (move_dir + repel * REPEL_STRENGTH).normalize_or_zero();
                    if blended.length_squared() > 0.0 {
                        h = blended.y.atan2(blended.x);
                        self.ants[idx].heading = h;
                    }
                }
                let mut used_return_path = false;
                if self.ants[idx].carrying_food {
                    let cur_pos = self.ants[idx].pos;
                    while let Some(target) = self.ants[idx].return_path.last().copied() {
                        if cur_pos.distance_squared(target) <= RETURN_WAYPOINT_RADIUS.powi(2) {
                            self.ants[idx].return_path.pop();
                            continue;
                        }
                        let to_target = (target - cur_pos).normalize_or_zero();
                        let target_dist = cur_pos.distance(target);
                        if self.obstacle_at(target)
                            || self.heading_hits_wall(
                                cur_pos,
                                to_target.y.atan2(to_target.x),
                                target_dist.min(SENSOR_DIST * 8.0),
                            )
                        {
                            self.ants[idx].return_path.pop();
                            continue;
                        }
                        if to_target.length_squared() > 0.0 {
                            let weave_side = if (self.ants[idx].id ^ (self.tick / 10)) & 1 == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            let route_weave = if self.has_walls {
                                weave_side * RETURN_ROUTE_HEADING_WEAVE
                            } else {
                                0.0
                            };
                            let target_heading = to_target.y.atan2(to_target.x) + route_weave;
                            h = blend_angle(h, target_heading, RETURN_TURN_BLEND);
                            if self.heading_hits_wall(cur_pos, h, SENSOR_DIST * 3.0) {
                                h = target_heading;
                            }
                            self.ants[idx].heading = h;
                            used_return_path = true;
                        }
                        break;
                    }
                }
                let blocked_home_aim = if self.ants[idx].carrying_food {
                    let heading = Vec2::new(h.cos(), h.sin());
                    let to_nest = (self.nest.pos - self.ants[idx].pos).normalize_or_zero();
                    to_nest.length_squared() > 0.0
                        && heading.dot(to_nest) > 0.75
                        && self.heading_hits_wall(self.ants[idx].pos, h, 260.0)
                } else {
                    false
                };
                let imminent_wall_without_route = self.ants[idx].carrying_food
                    && !used_return_path
                    && self.heading_hits_wall(self.ants[idx].pos, h, SENSOR_DIST * 3.0);
                if blocked_home_aim || imminent_wall_without_route {
                    let preferred_side = if (self.ants[idx].pos.y - self.nest.pos.y).abs() > 24.0 {
                        if self.ants[idx].pos.y < self.nest.pos.y {
                            -1.0
                        } else {
                            1.0
                        }
                    } else if (self.ants[idx].id ^ self.tick) & 1 == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let first = h + preferred_side * std::f32::consts::FRAC_PI_2;
                    let second = h - preferred_side * std::f32::consts::FRAC_PI_2;
                    h = if !self.heading_hits_wall(self.ants[idx].pos, first, SENSOR_DIST * 2.0) {
                        first
                    } else {
                        second
                    };
                    self.ants[idx].heading = h;
                    self.ants[idx].target_heading = h;
                }
                if self.ants[idx].carrying_food
                    && !used_return_path
                    && self.ants[idx].since_state_change <= CARRIER_DIRECT_HOME_GUARD_TICKS
                    && self.ants[idx].pos.distance_squared(self.nest.pos)
                        > CARRIER_DIRECT_HOME_MIN_DIST.powi(2)
                    && self.ants[idx].pos.distance_squared(self.nest.pos)
                        <= CARRIER_DIRECT_HOME_MAX_DIST.powi(2)
                {
                    // Fresh short-range pickups must search/curve before they
                    // settle onto a return trail. This block only turns away
                    // from too-homebound headings; it never adds a vector
                    // toward the nest.
                    let to_nest = (self.nest.pos - self.ants[idx].pos).normalize_or_zero();
                    let heading = Vec2::new(h.cos(), h.sin());
                    let side = if self.ants[idx].id & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let search_turn =
                        if self.ants[idx].since_state_change <= CARRIER_PICKUP_SEARCH_TICKS {
                            CARRIER_PICKUP_SEARCH_TURN
                        } else {
                            0.0
                        };
                    let scale = 1.0
                        - self.ants[idx]
                            .since_state_change
                            .min(CARRIER_DIRECT_HOME_GUARD_TICKS) as f32
                            / CARRIER_DIRECT_HOME_GUARD_TICKS as f32
                            * 0.5;
                    if to_nest.length_squared() > 0.0
                        && heading.dot(to_nest) >= CARRIER_DIRECT_HOME_DOT
                    {
                        let tangent = Vec2::new(-to_nest.y, to_nest.x) * side;
                        let lateral_search = (tangent + heading * 0.15).normalize_or_zero();
                        if lateral_search.length_squared() > 0.0 {
                            h = blend_angle(
                                h,
                                lateral_search.y.atan2(lateral_search.x),
                                CARRIER_DIRECT_HOME_AVOID_BLEND,
                            );
                        }
                    }
                    let turn = search_turn * side * scale;
                    if turn.abs() > 0.0 || h != self.ants[idx].heading {
                        h += turn;
                        let final_heading = Vec2::new(h.cos(), h.sin());
                        if to_nest.length_squared() > 0.0
                            && final_heading.dot(to_nest) >= CARRIER_FORBIDDEN_HOME_DOT
                        {
                            let tangent = Vec2::new(-to_nest.y, to_nest.x) * side;
                            h = tangent.y.atan2(tangent.x);
                        }
                        self.ants[idx].heading = h;
                        self.ants[idx].target_heading = h;
                    }
                }
                if self.ants[idx].carrying_food {
                    let pos = self.ants[idx].pos;
                    let adjusted_h = self.no_gps_carrier_heading(
                        self.ants[idx].id,
                        self.ants[idx].pickup_home_dist,
                        pos,
                        h,
                    );
                    if adjusted_h != h {
                        h = adjusted_h;
                        self.ants[idx].heading = h;
                    }
                    let wall_safe_h = self.avoid_blocked_home_heading(self.ants[idx].id, pos, h);
                    if wall_safe_h != h {
                        h = wall_safe_h;
                        self.ants[idx].heading = h;
                        self.ants[idx].target_heading = h;
                    }
                }
                // Now actual movement, with per-axis wall/bounds reflection.
                let cur = self.ants[idx].pos;
                let nx = cur.x + h.cos() * s;
                let ny = cur.y + h.sin() * s;
                let blocked_x =
                    nx <= 0.0 || nx >= self.width || self.obstacle_at(Vec2::new(nx, cur.y));
                let blocked_y =
                    ny <= 0.0 || ny >= self.height || self.obstacle_at(Vec2::new(cur.x, ny));
                // Dead-end repellent on wall/edge collision. Capped at a
                // local concentration of 4 so the buildup never reaches the
                // pesticide-poison threshold (20). Pesticide deposits ignore
                // the cap and can still saturate cells to 50.
                if blocked_x || blocked_y {
                    let pos = self.ants[idx].pos;
                    if !self.obstacle_at(pos) {
                        let current = self.pheromones.sample(PheromoneChannel::Repellent, pos);
                        if current < 4.0 {
                            self.pheromones
                                .deposit(PheromoneChannel::Repellent, pos, 0.15);
                        }
                    }
                }
                if blocked_x {
                    self.ants[idx].heading = std::f32::consts::PI - self.ants[idx].heading;
                    self.ants[idx].target_heading = self.ants[idx].heading;
                }
                if blocked_y {
                    self.ants[idx].heading = -self.ants[idx].heading;
                    self.ants[idx].target_heading = self.ants[idx].heading;
                }
                if !blocked_x {
                    self.ants[idx].pos.x = nx.clamp(0.5, self.width - 0.5);
                }
                if !blocked_y {
                    self.ants[idx].pos.y = ny.clamp(0.5, self.height - 0.5);
                }
                if self.ants[idx].carrying_food {
                    let mut final_h = self.no_gps_carrier_heading(
                        self.ants[idx].id,
                        self.ants[idx].pickup_home_dist,
                        self.ants[idx].pos,
                        self.ants[idx].heading,
                    );
                    final_h = self.avoid_blocked_home_heading(
                        self.ants[idx].id,
                        self.ants[idx].pos,
                        final_h,
                    );
                    self.ants[idx].heading = final_h;
                    self.ants[idx].target_heading = final_h;
                }
                if !self.ants[idx].carrying_food
                    && !self.near_wall(self.ants[idx].pos, self.wall_cell_size * 3.0)
                {
                    let should_push = self.ants[idx].breadcrumbs.last().map_or(true, |p| {
                        self.ants[idx].pos.distance_squared(*p) >= BREADCRUMB_MIN_DIST.powi(2)
                    });
                    if should_push {
                        if self.ants[idx].breadcrumbs.len() >= MAX_BREADCRUMBS {
                            self.ants[idx].breadcrumbs.remove(0);
                        }
                        let pos = self.ants[idx].pos;
                        self.ants[idx].breadcrumbs.push(pos);
                    }
                }
            }
            Action::PickupFood => {
                if self.ants[idx].carrying_food {
                    return;
                }
                let pos = self.ants[idx].pos;
                if let Some(fi) = self
                    .food
                    .iter()
                    .position(|f| f.pos.distance(pos) < PICKUP_RADIUS && f.amount > 0.0)
                {
                    // A "corpse-food" pile has amount < 1.0 (decomposed
                    // corpses produce 0.5-unit piles). Detect & count.
                    if self.food[fi].amount < 1.0 {
                        self.corpse_pickup_total += 1;
                    }
                    self.food[fi].amount -= 1.0;
                    self.ants[idx].carrying_food = true;
                    self.ants[idx].pickup_home_dist = pos.distance(self.nest.pos);
                    // johnBuffer-style: successful pickup fully refills the
                    // ant's "autonomy". As long as it keeps finding food,
                    // it never starves.
                    self.ants[idx].energy = 1.0;
                    // Reset deposit-strength budget. Carrier now lays
                    // strong pheromone for ~decay_horizon ticks, fading
                    // as it walks back to nest.
                    self.ants[idx].since_state_change = 0;
                    let filtered_route = self.ants[idx]
                        .breadcrumbs
                        .iter()
                        .copied()
                        .filter(|p| !self.obstacle_at(*p))
                        .collect::<Vec<_>>();
                    let has_nest_anchor = filtered_route.first().map_or(false, |p| {
                        p.distance_squared(self.nest.pos) <= 40.0_f32.powi(2)
                    });
                    let non_direct_route = route_is_non_direct(&filtered_route, self.nest.pos, pos);
                    if has_nest_anchor && non_direct_route && filtered_route.len() >= 4 {
                        if self.has_walls {
                            self.ants[idx].return_path = filtered_route;
                        } else {
                            self.ants[idx].return_path =
                                filtered_route.iter().rev().copied().collect();
                        }
                    } else {
                        self.ants[idx].return_path.clear();
                    }
                    self.ants[idx].breadcrumbs.clear();
                    // SNAP a 180° flip — physically the ant was heading INTO
                    // the food pile; now it leaves in the opposite direction.
                    // NOT "snap toward nest" (that would be GPS through walls).
                    // Wall-respecting trail/wander then guides them home.
                    let side = if (self.ants[idx].id ^ self.tick) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let mut h =
                        self.ants[idx].heading + std::f32::consts::PI + side * PICKUP_TURN_OFFSET;
                    let to_nest = (self.nest.pos - pos).normalize_or_zero();
                    if to_nest.length_squared() > 0.0
                        && pos.distance_squared(self.nest.pos)
                            > CARRIER_DIRECT_HOME_MIN_DIST.powi(2)
                        && pos.distance_squared(self.nest.pos)
                            <= CARRIER_DIRECT_HOME_MAX_DIST.powi(2)
                    {
                        let heading = Vec2::new(h.cos(), h.sin());
                        if heading.dot(to_nest) >= CARRIER_DIRECT_HOME_DOT {
                            let tangent = Vec2::new(-to_nest.y, to_nest.x) * side;
                            let lateral_search = (tangent + heading * 0.15).normalize_or_zero();
                            if lateral_search.length_squared() > 0.0 {
                                h = blend_angle(
                                    h,
                                    lateral_search.y.atan2(lateral_search.x),
                                    CARRIER_DIRECT_HOME_AVOID_BLEND,
                                );
                            }
                        }
                        let final_heading = Vec2::new(h.cos(), h.sin());
                        if final_heading.dot(to_nest) >= CARRIER_FORBIDDEN_HOME_DOT {
                            let tangent = Vec2::new(-to_nest.y, to_nest.x) * side;
                            h = tangent.y.atan2(tangent.x);
                        }
                    }
                    self.ants[idx].heading = h;
                    self.ants[idx].target_heading = h;
                    if self.food[fi].amount <= 0.0 {
                        let depleted_pos = self.food[fi].pos;
                        self.food.swap_remove(fi);
                        let nearby_food_remaining = self
                            .food
                            .iter()
                            .any(|f| f.pos.distance_squared(depleted_pos) < 80.0_f32.powi(2));
                        if !nearby_food_remaining {
                            self.pheromones.clear_region(
                                PheromoneChannel::FoodSmell,
                                depleted_pos,
                                360.0,
                            );
                            self.pheromones.deposit(
                                PheromoneChannel::Repellent,
                                depleted_pos,
                                self.config.stuck_repel_strength,
                            );
                        }
                    }
                }
            }
            Action::DropFood => {
                if !self.ants[idx].carrying_food {
                    return;
                }
                let pos = self.ants[idx].pos;
                if pos.distance(self.nest.pos) <= self.nest.radius {
                    self.nest.food_stored += 1.0;
                    self.food_delivered_total += 1;
                    self.ants[idx].carrying_food = false;
                    self.ants[idx].pickup_home_dist = 0.0;
                    self.ants[idx].energy = 1.0;
                    self.ants[idx].return_path.clear();
                    self.ants[idx].breadcrumbs.clear();
                    self.ants[idx].breadcrumbs.push(self.nest.pos);
                    // Reset budget — outbound ant now lays strong
                    // pheromone for ~decay_horizon ticks heading toward food.
                    self.ants[idx].since_state_change = 0;
                    // SNAP 180° flip — ant arrived at nest heading inward,
                    // now leaves heading outward. NOT "snap away from nest"
                    // (that would also be GPS — just slightly less obvious).
                    let h = self.ants[idx].heading + std::f32::consts::PI;
                    self.ants[idx].heading = h;
                    self.ants[idx].target_heading = h;
                }
            }
            Action::LayPheromone { channel, strength } => {
                let pos = self.ants[idx].pos;
                let trail_channel =
                    matches!(channel, PheromoneChannel::Food | PheromoneChannel::Home);
                if !self.obstacle_at(pos)
                    && (!trail_channel || !self.near_wall(pos, self.wall_cell_size * 1.5))
                {
                    if self.config.bilinear_deposit {
                        self.pheromones.deposit_bilinear(channel, pos, strength);
                    } else {
                        self.pheromones.deposit(channel, pos, strength);
                    }
                }
                // Detect stuck-escape Repellent deposit signature.
                if matches!(channel, PheromoneChannel::Repellent)
                    && (strength - self.config.stuck_repel_strength).abs() < 0.001
                {
                    self.stuck_escapes_total += 1;
                }
            }
            Action::Attack { target_id } => {
                let attacker_pos = self.ants[idx].pos;
                if let Some(tidx) = self.ant_idx(target_id) {
                    if self.ants[tidx].pos.distance(attacker_pos) < ATTACK_RADIUS {
                        self.ants[tidx].hp -= 0.35;
                    }
                }
            }
            Action::Spawn { role } => {
                if Some(id) == self.nest.queen_id && self.nest.food_stored >= 1.0 {
                    self.nest.food_stored -= 1.0;
                    self.spawn_ant(self.nest.pos, role, 0);
                }
            }
            Action::Idle => {}
        }
    }
}
