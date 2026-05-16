//! Rule-based brains for queen / worker / soldier.
//!
//! Each is intentionally short (~30 lines) and side-effect free apart from its
//! own internal state. Rewriting any one of these does not affect the others
//! or the sim. To add a neural-net brain, mirror this file: same trait, same
//! `decide` signature.

use crate::brain::{Action, Brain, Perception, PheromoneChannel};
use crate::entities::Role;
use glam::Vec2;
use rand::rngs::SmallRng;
use rand::Rng;
use std::f32::consts::{PI, TAU};

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

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker state is intentionally small. No stored nest vector: the worker only
/// remembers stale Food-trail exposure and, while carrying, a local search
/// heading so losing Home signal does not become a random walk back to food.
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
}

impl Default for WorkerBrain {
    fn default() -> Self {
        Self {
            stale_trail_ticks: 0,
            carrier_search_heading: None,
            carrier_wall_side: 0.0,
            carrier_wall_follow_ticks: 0,
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
const STALE_TRAIL_LIMIT: u32 = 60;

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
const JB_FOOD_SMELL_SEARCH_WEIGHT: f32 = 1.0;
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
const CARRIER_WALL_FOLLOW_TICKS: u32 = 160;

impl Brain for WorkerBrain {
    fn decide(&mut self, p: &Perception, _rng: &mut SmallRng) -> Vec<Action> {
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
            if self.stale_trail_ticks >= STALE_TRAIL_LIMIT {
                self.stale_trail_ticks = 0;
                let jitter_seed =
                    (p.self_id.wrapping_mul(1103515245) ^ p.tick.wrapping_mul(12345)) as i32;
                let r01 = ((jitter_seed & 0xffff) as f32 / 65535.0) - 0.5;
                let turn = PI + r01 * 0.8;
                return vec![
                    Action::LayPheromone {
                        channel: PheromoneChannel::Repellent,
                        strength: p.stuck_repel_strength,
                    },
                    Action::SetHeading(p.self_heading + turn),
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

        let new_heading = if max >= JB_SENSOR_FLOOR {
            if p.carrying_food {
                self.carrier_search_heading = None;
                self.carrier_wall_follow_ticks = 0;
            }
            // Trail detected — turn toward the strongest sensor.
            let trail_heading = if c >= l && c >= r && !center_blocked {
                p.self_heading
            } else if l > r {
                p.self_heading - JB_TURN_PER_TICK
            } else {
                p.self_heading + JB_TURN_PER_TICK
            };
            if !p.carrying_food && p.gradient_food_smell.length_squared() > 0.0 {
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
                        self.carrier_wall_follow_ticks = CARRIER_WALL_FOLLOW_TICKS;
                        search_heading += self.carrier_wall_side * JB_TURN_PER_TICK;
                    } else if self.carrier_wall_follow_ticks > 0 {
                        self.carrier_wall_follow_ticks -= 1;
                        search_heading += jitter * CARRIER_SEARCH_JITTER_SCALE;
                    } else if p.gradient_food_smell.length_squared() > 0.0 {
                        search_heading = heading_of(-p.gradient_food_smell);
                    } else {
                        let search_dir = Vec2::new(search_heading.cos(), search_heading.sin())
                            * CARRIER_SEARCH_MOMENTUM_WEIGHT;
                        let repellent_avoid = if p.repellent_here > 0.2 {
                            -p.gradient_repellent * CARRIER_SEARCH_REPELLENT_WEIGHT
                        } else {
                            Vec2::ZERO
                        };
                        let search = (search_dir + repellent_avoid).normalize_or_zero();
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
            } else {
                let attraction = if p.wall_ahead {
                    let turn = if (p.self_id ^ p.tick) & 1 == 0 {
                        -JB_TURN_PER_TICK
                    } else {
                        JB_TURN_PER_TICK
                    };
                    return vec![
                        Action::SetHeading(p.self_heading + turn),
                        Action::Forward(JB_FORWARD_SPEED),
                    ];
                } else {
                    p.gradient_to_food * JB_TRAIL_GRADIENT_WEIGHT
                        + p.gradient_food_smell * JB_FOOD_SMELL_SEARCH_WEIGHT
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
                    p.self_heading + step + jitter * 0.5
                } else {
                    p.self_heading + jitter
                }
            }
        };

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
        }
    }

    fn base_perception(carrying_food: bool) -> Perception {
        Perception {
            self_id: 7,
            self_pos: Vec2::new(100.0, 100.0),
            self_heading: PI * 0.5,
            self_colony: 0,
            carrying_food,
            pickup_home_dist: 0.0,
            has_return_route: false,
            at_nest: false,
            nest_pos: Vec2::new(0.0, 0.0),
            colony_food: 0.0,
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
                } if (*strength - p.stuck_repel_strength).abs() < 0.001
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
