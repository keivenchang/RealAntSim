//! Commands from the frontend. JSON over the same WebSocket.
//!
//! Example payloads (sent by the browser):
//!   {"op":"set_speed","value":100}
//!   {"op":"set_respawn","value":true}
//!   {"op":"spawn_food"}
//!   {"op":"reset"}

use crate::brain::WorkerBrainKind;
use crate::world::World;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    SetSpeed {
        value: u32,
    },
    SetRespawn {
        value: bool,
    },
    SetRespawnInterval {
        value: u32,
    },
    SetSpawnCooldown {
        value: u32,
    },
    SetSoldierRatio {
        value: f32,
    },
    SpawnFood,
    Reset,
    /// Debug-only: instantly kill the queen, triggering the sim halt path.
    /// Useful before v2 introduces real combat.
    KillQueen,
    /// Paint a wall disc at world (x, y) with the given brush radius.
    AddWall {
        x: f32,
        y: f32,
        radius: f32,
    },
    RemoveWall {
        x: f32,
        y: f32,
        radius: f32,
    },
    ClearWalls,
    LoadScenario {
        name: String,
    },
    /// Kill every (non-queen) ant within `radius` of (`x`, `y`). Dead ants
    /// are removed immediately; they do not become corpses or food.
    SprayPesticide {
        x: f32,
        y: f32,
        radius: f32,
    },
    /// Drop a fresh food pile at the cursor. `amount` is optional so older
    /// click-only clients still use the current configured pile size.
    PlaceFoodAt {
        x: f32,
        y: f32,
        amount: Option<f32>,
    },
    /// Toggle the simulation pause flag. Snapshots keep flowing while paused;
    /// only the sim step is skipped.
    TogglePause,
    SetMaxColonySize {
        value: u32,
    },
    SetWorkerBrain {
        value: WorkerBrainKind,
    },
}

pub fn apply(world: &mut World, cmd: Command) {
    match cmd {
        Command::SetSpeed { value } => {
            world.config.speed_mult = value.clamp(1, 1000);
        }
        Command::SetRespawn { value } => {
            world.config.food_respawn = value;
        }
        Command::SetRespawnInterval { value } => {
            world.config.food_respawn_interval_ticks = value.max(1);
        }
        Command::SetSpawnCooldown { value } => {
            world.config.spawn_cooldown_ticks = value.clamp(1, 1000);
        }
        Command::SetSoldierRatio { value } => {
            world.config.soldier_ratio = value.clamp(0.0, 1.0);
        }
        Command::SpawnFood => {
            world.place_food_pile(world.config.food_respawn_amount);
        }
        Command::Reset => {
            let (w, h) = (world.width, world.height);
            let prev_config = world.config.clone();
            *world = World::new(w, h);
            world.config = prev_config;
            world.rebuild_worker_brains();
        }
        Command::KillQueen => {
            if let Some(qid) = world.nest.queen_id {
                if let Some(idx) = world.ants.iter().position(|a| a.id == qid) {
                    world.ants[idx].hp = 0.0;
                }
            }
        }
        Command::AddWall { x, y, radius } => {
            world.paint_walls(x, y, radius.clamp(2.0, 50.0), true);
        }
        Command::RemoveWall { x, y, radius } => {
            world.paint_walls(x, y, radius.clamp(2.0, 50.0), false);
        }
        Command::ClearWalls => {
            world.clear_walls();
        }
        Command::LoadScenario { name } => {
            world.load_scenario(&name);
        }
        Command::SprayPesticide { x, y, radius } => {
            use crate::brain::PheromoneChannel;
            let center = glam::Vec2::new(x, y);
            let cloud_radius = radius * 1.5;
            // 1) Pesticide DOMINATES — wipe all other pheromone channels
            //    in the cloud area. The toxin overpowers any existing
            //    trails / smells. Ants in the cloud lose orientation.
            for ch in [
                PheromoneChannel::Home,
                PheromoneChannel::Food,
                PheromoneChannel::FoodSmell,
                PheromoneChannel::Alarm,
            ] {
                world.pheromones.clear_region(ch, center, cloud_radius);
            }
            // 2) Deposit a STRONG, persistent Repellent cloud. HP damage
            //    is dose-dependent (handled in world bookkeeping), so a
            //    full-strength cell kills in a few hundred ticks; weaker
            //    exposure takes proportionally longer. No instant death
            //    from the spray itself — ants stagger and slowly die.
            let step = 6.0_f32;
            let n = (cloud_radius / step).ceil() as i32;
            for dy in -n..=n {
                for dx in -n..=n {
                    let offset = glam::Vec2::new(dx as f32 * step, dy as f32 * step);
                    let dist = offset.length();
                    if dist >= cloud_radius {
                        continue;
                    }
                    // Strong, smooth falloff. Peak at center, 0 at edge.
                    let strength = (1.0 - dist / cloud_radius) * 40.0;
                    world.pheromones.deposit(
                        PheromoneChannel::Repellent,
                        center + offset,
                        strength,
                    );
                }
            }
        }
        Command::PlaceFoodAt { x, y, amount } => {
            let pos = glam::Vec2::new(x, y);
            if !world.wall_at(pos) && pos.distance(world.nest.pos) > 30.0 {
                let amount = amount
                    .unwrap_or(world.config.food_respawn_amount)
                    .clamp(1.0, 5_000.0);
                world.add_food_at(pos, amount);
            }
        }
        Command::TogglePause => {
            world.config.paused = !world.config.paused;
        }
        Command::SetMaxColonySize { value } => {
            world.config.max_colony_size = value.clamp(10, 5000);
        }
        Command::SetWorkerBrain { value } => {
            world.set_worker_brain_kind(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_food_clicks_bypass_automatic_respawn_cap() {
        let mut world = World::new(300.0, 300.0);
        world.config.max_food_piles = 0;

        apply(
            &mut world,
            Command::PlaceFoodAt {
                x: 240.0,
                y: 150.0,
                amount: None,
            },
        );
        apply(
            &mut world,
            Command::PlaceFoodAt {
                x: 240.0,
                y: 180.0,
                amount: None,
            },
        );

        assert_eq!(world.food.len(), 2);
    }

    #[test]
    fn manual_food_drag_can_create_large_pile() {
        let mut world = World::new(300.0, 300.0);

        apply(
            &mut world,
            Command::PlaceFoodAt {
                x: 240.0,
                y: 150.0,
                amount: Some(900.0),
            },
        );

        assert_eq!(world.food.len(), 1);
        assert_eq!(world.food[0].amount, 900.0);
    }

    #[test]
    fn worker_brain_command_updates_config() {
        let mut world = World::new(300.0, 300.0);

        apply(
            &mut world,
            Command::SetWorkerBrain {
                value: WorkerBrainKind::Weighted,
            },
        );

        assert_eq!(world.config.worker_brain_kind, WorkerBrainKind::Weighted);
    }

    #[test]
    fn worker_brain_command_accepts_snake_case_json() {
        let cmd: Command =
            serde_json::from_str(r#"{"op":"set_worker_brain","value":"weighted"}"#).unwrap();

        assert!(matches!(
            cmd,
            Command::SetWorkerBrain {
                value: WorkerBrainKind::Weighted
            }
        ));
    }

    #[test]
    fn worker_brain_command_accepts_neural_json() {
        let cmd: Command =
            serde_json::from_str(r#"{"op":"set_worker_brain","value":"neural"}"#).unwrap();

        assert!(matches!(
            cmd,
            Command::SetWorkerBrain {
                value: WorkerBrainKind::Neural
            }
        ));
    }

    #[test]
    fn random_spawn_food_still_respects_respawn_cap() {
        let mut world = World::new(300.0, 300.0);
        world.config.max_food_piles = 0;

        apply(&mut world, Command::SpawnFood);

        assert_eq!(world.food.len(), 0);
    }
}
