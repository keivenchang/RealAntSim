//! JSON snapshot shipped to the frontend each tick. v1 keeps it human-readable
//! so it's easy to debug in DevTools. Switch to a packed binary in v2 if
//! bandwidth becomes a concern.

use crate::brain::PheromoneChannel;
use crate::entities::Role;
use crate::world::World;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Serialize;

// Pheromone snapshot resolution. Chosen to match the source field exactly
// (world 800x600 / cell_size 4.0 = 200x150 cells), so there is NO downsample
// and no MAX-over-overlapping-windows smear. Each target pixel is exactly
// one source cell, and renders at exactly that cell's world footprint.
const PHER_COLS: usize = 200;
const PHER_ROWS: usize = 150;
/// Lower scale = more sensitive (faint trails show up brighter).
const PHER_SCALE: f32 = 4.0;
// Repellent renders over a much wider dynamic range — a pesticide spray
// peaks at ~40 (max_value=50), 8× higher than typical trails. Without a
// separate scale, the whole cloud saturates byte=255 and the gradient
// you wanted to see flattens to one solid color.
const PHER_SCALE_REPEL: f32 = 30.0;
/// Higher than ant trails because the food-pile source saturates at ~35,
/// and we want to actually SEE the gradient falling off around piles
/// rather than the whole plume reading "bright" at the top of the gamma curve.
const PHER_SCALE_SMELL: f32 = 18.0;

// Packed binary layout — 18 bytes/ant, little-endian:
//   offset  size  field
//   0       4     id (u32)
//   4       4     x  (f32)
//   8       4     y  (f32)
//   12      1     heading mapped to 0..=255 over 0..2π
//   13      1     role: 0=Queen, 1=Worker, 2=Soldier
//   14      1     flags: bit0 = carrying_food
//   15      1     hp × 255
//   16      1     colony
//   17      1     reserved (pad to even byte alignment)
//
// 1000 ants ≈ 18 KB raw → 24 KB base64. Beats the old ~80 KB JSON array.
pub const ANT_STRIDE: usize = 18;

#[derive(Serialize)]
pub struct FoodDto {
    pub x: f32,
    pub y: f32,
    pub amount: f32,
}

#[derive(Serialize)]
pub struct CorpseDto {
    pub x: f32,
    pub y: f32,
    /// 0..1 — 1 = fresh, 0 = about to decompose.
    pub fresh: f32,
    /// True if this ant died from pesticide. Rendered black in the GUI.
    pub poisoned: bool,
}

#[derive(Serialize)]
pub struct NestDto {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub food_stored: f32,
    pub queen_alive: bool,
    /// 0..1. The queen drains slowly and refills from `food_stored`; when
    /// this hits 0 she dies and the sim halts.
    pub queen_energy: f32,
    /// 0..1. Queen's HP. Decreased by combat (v2). Drops to 0 = death.
    pub queen_hp: f32,
}

#[derive(Serialize)]
pub struct Stats {
    pub n_workers: u32,
    pub n_soldiers: u32,
    pub n_queens: u32,
    pub food_stored: f32,
    pub food_in_world: f32,
}

#[derive(Serialize)]
pub struct ConfigDto {
    pub speed_mult: u32,
    pub food_respawn: bool,
    pub food_respawn_interval_ticks: u32,
    pub spawn_cooldown_ticks: u32,
    pub soldier_ratio: f32,
    pub max_colony_size: u32,
    pub paused: bool,
}

#[derive(Serialize)]
pub struct Snapshot {
    pub tick: u32,
    pub width: f32,
    pub height: f32,
    pub running: bool,
    /// Base64-encoded packed binary ant array. See ANT_STRIDE comment for
    /// the layout. Replaces the per-ant JSON object array — saves ~5× bytes.
    pub ants_packed: String,
    pub n_ants: u32,
    pub food: Vec<FoodDto>,
    pub corpses: Vec<CorpseDto>,
    pub nest: NestDto,
    pub pher_cols: usize,
    pub pher_rows: usize,
    /// Base64-encoded `PHER_COLS * PHER_ROWS` bytes (0..=255 per cell).
    pub pher_food: String,
    pub pher_home: String,
    pub pher_smell: String,
    pub pher_repel: String,
    /// Wall grid (rows × cols of u8, 0 = clear, 1 = wall), base64-encoded.
    pub walls: String,
    pub wall_cols: usize,
    pub wall_rows: usize,
    pub wall_cell_size: f32,
    pub stats: Stats,
    pub config: ConfigDto,
}

pub fn snapshot(w: &World) -> Snapshot {
    use std::f32::consts::TAU;
    // Pack ants into a flat byte buffer.
    let mut ant_bytes = Vec::with_capacity(w.ants.len() * ANT_STRIDE);
    let mut n_workers: u32 = 0;
    let mut n_soldiers: u32 = 0;
    let mut n_queens: u32 = 0;
    for a in &w.ants {
        ant_bytes.extend_from_slice(&a.id.to_le_bytes());
        ant_bytes.extend_from_slice(&a.pos.x.to_le_bytes());
        ant_bytes.extend_from_slice(&a.pos.y.to_le_bytes());
        let h_norm = (a.heading.rem_euclid(TAU) / TAU).clamp(0.0, 0.999);
        ant_bytes.push((h_norm * 256.0) as u8);
        let (role_u, role_count) = match a.role {
            Role::Queen => (0u8, &mut n_queens),
            Role::Worker => (1u8, &mut n_workers),
            Role::Soldier => (2u8, &mut n_soldiers),
        };
        *role_count += 1;
        ant_bytes.push(role_u);
        ant_bytes.push(if a.carrying_food { 1 } else { 0 });
        ant_bytes.push((a.hp.clamp(0.0, 1.0) * 255.0) as u8);
        ant_bytes.push(a.colony);
        ant_bytes.push(0); // pad
    }
    let ants_packed = STANDARD.encode(&ant_bytes);
    let n_ants = w.ants.len() as u32;
    let food: Vec<FoodDto> = w
        .food
        .iter()
        .map(|f| FoodDto {
            x: f.pos.x,
            y: f.pos.y,
            amount: f.amount,
        })
        .collect();
    let corpses: Vec<CorpseDto> = w
        .corpses
        .iter()
        .map(|c| CorpseDto {
            x: c.pos.x,
            y: c.pos.y,
            fresh: (c.ticks_remaining as f32 / 18000.0).clamp(0.0, 1.0),
            poisoned: c.poisoned,
        })
        .collect();
    let food_in_world: f32 = w.food.iter().map(|f| f.amount).sum();
    let (queen_energy, queen_hp) = w
        .nest
        .queen_id
        .and_then(|qid| w.ants.iter().find(|a| a.id == qid))
        .map(|a| (a.energy, a.hp))
        .unwrap_or((0.0, 0.0));
    let nest = NestDto {
        x: w.nest.pos.x,
        y: w.nest.pos.y,
        radius: w.nest.radius,
        food_stored: w.nest.food_stored,
        queen_alive: w.nest.queen_id.is_some(),
        queen_energy,
        queen_hp,
    };
    Snapshot {
        tick: w.tick,
        width: w.width,
        height: w.height,
        running: w.is_running(),
        ants_packed,
        n_ants,
        food,
        corpses,
        nest,
        pher_cols: PHER_COLS,
        pher_rows: PHER_ROWS,
        pher_food: STANDARD.encode(w.pheromones.snapshot_downsampled(
            PheromoneChannel::Food,
            PHER_COLS,
            PHER_ROWS,
            PHER_SCALE,
        )),
        pher_home: STANDARD.encode(w.pheromones.snapshot_downsampled(
            PheromoneChannel::Home,
            PHER_COLS,
            PHER_ROWS,
            PHER_SCALE,
        )),
        pher_smell: STANDARD.encode(w.pheromones.snapshot_downsampled(
            PheromoneChannel::FoodSmell,
            PHER_COLS,
            PHER_ROWS,
            PHER_SCALE_SMELL,
        )),
        pher_repel: STANDARD.encode(w.pheromones.snapshot_downsampled(
            PheromoneChannel::Repellent,
            PHER_COLS,
            PHER_ROWS,
            PHER_SCALE_REPEL,
        )),
        walls: STANDARD.encode(
            w.walls
                .iter()
                .map(|&b| if b { 1u8 } else { 0u8 })
                .collect::<Vec<u8>>(),
        ),
        wall_cols: w.wall_cols,
        wall_rows: w.wall_rows,
        wall_cell_size: w.wall_cell_size,
        stats: Stats {
            n_workers,
            n_soldiers,
            n_queens,
            food_stored: w.nest.food_stored,
            food_in_world,
        },
        config: ConfigDto {
            speed_mult: w.config.speed_mult,
            food_respawn: w.config.food_respawn,
            food_respawn_interval_ticks: w.config.food_respawn_interval_ticks,
            spawn_cooldown_ticks: w.config.spawn_cooldown_ticks,
            soldier_ratio: w.config.soldier_ratio,
            max_colony_size: w.config.max_colony_size,
            paused: w.config.paused,
        },
    }
}
