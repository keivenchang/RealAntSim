//! Brain abstraction: every ant has a `Box<dyn Brain>`.
//!
//! The sim builds a `Perception` for the ant, the brain returns a `Vec<Action>`,
//! the sim applies the actions. Brains hold their own per-ant state.
//!
//! Adding a new behavior = implement `Brain` and register it in
//! `World::spawn_ant`. To swap rule-based for a neural net later: write
//! `NnBrain { net: SmallNet }` that flattens `Perception` to a fixed-size float
//! input vector and decodes the output back to `Action`s. No sim changes needed.

use crate::entities::{EntityId, Role};
use glam::Vec2;
use rand::rngs::SmallRng;

#[derive(Clone, Copy, Debug)]
pub enum PheromoneChannel {
    /// "I came from the nest" — laid by outbound ants, followed by ants
    /// trying to get back home.
    Home,
    /// "Food this way" — laid by ants returning with food. A reinforcement
    /// trail (red).
    Food,
    /// "Danger here" — laid by soldiers / dying ants, recruits soldiers.
    Alarm,
    /// "I am food" — emitted by food piles themselves (not by ants). The
    /// gradient from this channel unambiguously points TO a pile, since
    /// piles are its only source. Rendered green.
    FoodSmell,
    /// "Don't bother this way" — laid by ants who got stuck or found a
    /// depleted pile. Acts as a *negative* attractant; outbound workers
    /// bias away from it. Idea borrowed from johnBuffer/AntSimulator.
    Repellent,
}

#[derive(Clone, Copy)]
pub struct NearbyAnt {
    pub id: EntityId,
    pub pos: Vec2,
    pub colony: u8,
    pub role: Role,
}

/// One ray cast into the ant's forward cone for stochastic sampling
/// (replaces deterministic gradient descent). The brain picks the highest-
/// scoring direction.
#[derive(Clone, Copy)]
pub struct ForwardSample {
    pub food: f32,
    pub repellent: f32,
    pub home: f32,
}

pub struct Perception {
    pub self_id: EntityId,
    pub self_pos: Vec2,
    pub self_heading: f32,
    pub self_colony: u8,
    pub carrying_food: bool,
    pub pickup_home_dist: f32,
    /// True when the ant is carrying food and has a breadcrumb route to replay.
    /// This is route memory, not a nest vector.
    pub has_return_route: bool,
    pub at_nest: bool,
    pub nest_pos: Vec2,
    pub colony_food: f32,
    pub nearby_food: Vec<(Vec2, f32)>, // (pos, amount), within perception radius
    pub nearby_ants: Vec<NearbyAnt>,   // within perception radius, excludes self
    pub gradient_to_food: Vec2,        // unit vector toward higher Food pheromone (or zero)
    pub gradient_alarm: Vec2,
    /// Unit vector toward stronger FoodSmell. Always points roughly toward
    /// the nearest food pile (modulo diffusion / multiple piles).
    pub gradient_food_smell: Vec2,
    /// Unit vector toward stronger Repellent. We use the *opposite* of this
    /// direction as a "stay away from here" bias.
    pub gradient_repellent: Vec2,
    /// Concentration of each pheromone channel at the ant's current cell.
    /// 0..PheromoneField::max_value (50). When this is high, the ant is in
    /// a saturated patch and the gradient direction is unreliable.
    pub food_here: f32,
    pub food_smell_here: f32,
    pub repellent_here: f32,
    /// True if there's a wall in the cell immediately ahead of the ant.
    /// Brains can use this to avoid heading straight into static obstacles.
    pub wall_ahead: bool,
    /// johnBuffer-style 3-sensor probe: left/center/right at a fixed
    /// distance ahead of the ant. Brain picks the highest-pheromone
    /// sensor and turns toward it. Simpler and tighter than the 16-ray
    /// stochastic cone — produces single-file trail-following.
    pub sensor_left: ForwardSample,
    pub sensor_center: ForwardSample,
    pub sensor_right: ForwardSample,
    pub tick: u32,
    /// Live queen-spawn params (so the QueenBrain reads them each tick and
    /// the user can tweak via the UI without restarting).
    pub spawn_cooldown_ticks: u32,
    pub soldier_ratio: f32,
    pub colony_size: u32,
    pub max_colony_size: u32,
    /// Per-ant Food trail lay strength (from SimConfig).
    pub food_lay_strength: f32,
    /// Food-channel saturation cap (from SimConfig).
    pub food_sat_cap: f32,
    /// Outbound Home-trail reinforcement gate.
    pub outbound_lay_threshold: f32,
    /// Strength of the Repellent laid when an outbound ant gives up on a
    /// stale Food trail.
    pub stuck_repel_strength: f32,
    /// johnBuffer-style time-decayed deposit. Ticks since the ant's
    /// most recent state change (pickup/drop). Deposit strength fades
    /// as this increases: `strength × max(0, 1 - since/horizon)`.
    pub since_state_change: u32,
    /// Decay horizon in ticks. After this many ticks, deposit strength
    /// hits zero — past this the ant is too far from its source and
    /// shouldn't pollute the field with weak deposits.
    pub deposit_decay_horizon: u32,
}

#[derive(Clone, Copy)]
pub enum Action {
    /// Set absolute heading (radians). Sim clamps angular wrap.
    SetHeading(f32),
    /// Move forward at this speed (units/tick), clamped by sim.
    Forward(f32),
    PickupFood,
    DropFood,
    LayPheromone {
        channel: PheromoneChannel,
        strength: f32,
    },
    Attack {
        target_id: EntityId,
    },
    /// Queen-only. Spawns a new ant of `role` at the nest, costs 1 stored food.
    Spawn {
        role: Role,
    },
    Idle,
}

// Sync is required so &World can cross thread boundaries during the
// parallel perception phase (rayon par_iter). All current brain impls
// hold only POD fields so this is satisfied trivially.
pub trait Brain: Send + Sync {
    fn decide(&mut self, p: &Perception, rng: &mut SmallRng) -> Vec<Action>;
}
