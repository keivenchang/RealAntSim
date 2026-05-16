use crate::brain::PheromoneChannel;
use glam::Vec2;

const N_CHANNELS: usize = 5;
// In obstacle maps, high Home diffusion blurs the route around wall corners
// and creates extra branch cells. Cap only Home diffusion, only with walls.
const WALL_HOME_DIFFUSION_CAP: f32 = 0.02;
// Obstacle maps accumulate side branches because ants repeatedly probe wall
// edges before a detour wins. Evaporate ant-laid trails slightly faster only
// when walls exist; open-field convergence benches keep the default decay.
const WALL_TRAIL_DECAY_CAP: f32 = 0.9975;

/// Coarse 2D scalar field per channel. Cheap deposit/sample/decay.
/// Diffusion is intentionally skipped for v1 — pure decay produces recognizable
/// fading trails without ping-pong buffers.
pub struct PheromoneField {
    pub cell_size: f32,
    pub cols: usize,
    pub rows: usize,
    pub width: f32,
    pub height: f32,
    grids: [Vec<f32>; N_CHANNELS],
    /// One scratch buffer PER channel (was previously shared). Letting each
    /// channel own its own scratch unlocks parallel decay+diffuse — rayon
    /// can hand a (grid, scratch) pair to each worker thread independently.
    scratches: [Vec<f32>; N_CHANNELS],
    pub decay: [f32; N_CHANNELS],
    pub diffusion: [f32; N_CHANNELS],
    pub max_value: f32,
}

impl PheromoneField {
    pub fn new(width: f32, height: f32, cell_size: f32) -> Self {
        let cols = (width / cell_size).ceil() as usize;
        let rows = (height / cell_size).ceil() as usize;
        let n = cols * rows;
        Self {
            cell_size,
            cols,
            rows,
            width,
            height,
            grids: [
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
            ],
            scratches: [
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
            ],
            // Channels: Home, Food, Alarm, FoodSmell, Repellent.
            // Home/Food are ant-laid trails, so they should evaporate in
            // place instead of blooming into a map-wide halo. FoodSmell is
            // the only deliberately diffusive signal, and even that stays
            // short enough that a pile creates a local plume rather than a
            // full-screen field.
            decay: [0.998, 0.998, 0.988, 0.9994, 0.999],
            diffusion: [0.03, 0.0, 0.10, 0.06, 0.02],
            // Restored to 50 since deposits are now LIGHT (ACO-style, ~0.3
            // per ant per tick) — the saturation problem is avoided by
            // the explicit suppress-deposit-on-saturated-cell rule in
            // brains.rs, not by capping max_value low.
            max_value: 50.0,
        }
    }

    fn ch_idx(ch: PheromoneChannel) -> usize {
        match ch {
            PheromoneChannel::Home => 0,
            PheromoneChannel::Food => 1,
            PheromoneChannel::Alarm => 2,
            PheromoneChannel::FoodSmell => 3,
            PheromoneChannel::Repellent => 4,
        }
    }

    fn cell_of(&self, pos: Vec2) -> (usize, usize) {
        let c = ((pos.x / self.cell_size) as i32).clamp(0, self.cols as i32 - 1) as usize;
        let r = ((pos.y / self.cell_size) as i32).clamp(0, self.rows as i32 - 1) as usize;
        (c, r)
    }

    /// Single-cell deposit. Real ants drop pheromone at a point (tarsus
    /// contact). Tightest trails, but no immediate halo so distant ants
    /// can't detect the trail.
    pub fn deposit(&mut self, ch: PheromoneChannel, pos: Vec2, strength: f32) {
        let (c, r) = self.cell_of(pos);
        let i = r * self.cols + c;
        let ch_index = Self::ch_idx(ch);
        let g = &mut self.grids[ch_index];
        g[i] = (g[i] + strength).min(self.max_value);
    }

    /// Bilinear deposit. Spreads across the 4 cells bracketing `pos`, so
    /// every deposit creates a small 2×2 halo. Combined with strong
    /// gradient-climb in the brain, this gives "wide halo for detection,
    /// sharp climb to the core" — biological chemotaxis behavior.
    pub fn deposit_bilinear(&mut self, ch: PheromoneChannel, pos: Vec2, strength: f32) {
        let s = strength * 4.0;
        let fx = pos.x / self.cell_size - 0.5;
        let fy = pos.y / self.cell_size - 0.5;
        let c0 = fx.floor() as i32;
        let r0 = fy.floor() as i32;
        let dx = fx - c0 as f32;
        let dy = fy - r0 as f32;
        let weights = [
            (c0, r0, (1.0 - dx) * (1.0 - dy)),
            (c0 + 1, r0, dx * (1.0 - dy)),
            (c0, r0 + 1, (1.0 - dx) * dy),
            (c0 + 1, r0 + 1, dx * dy),
        ];
        let cols = self.cols as i32;
        let rows = self.rows as i32;
        let ch_index = Self::ch_idx(ch);
        let max_value = self.max_value;
        let g = &mut self.grids[ch_index];
        for (c, r, w) in weights {
            if c < 0 || r < 0 || c >= cols || r >= rows {
                continue;
            }
            let i = (r as usize) * self.cols + (c as usize);
            g[i] = (g[i] + s * w).min(max_value);
        }
    }

    /// Zero out a channel within a circular region. Used by pesticide
    /// spray to override existing trails — the toxin dominates.
    pub fn clear_region(&mut self, ch: PheromoneChannel, center: Vec2, radius: f32) {
        let g = &mut self.grids[Self::ch_idx(ch)];
        let r2 = radius * radius;
        let cs = self.cell_size;
        let c0 = ((center.x - radius) / cs).floor().max(0.0) as usize;
        let c1 = (((center.x + radius) / cs).ceil() as usize).min(self.cols);
        let r0 = ((center.y - radius) / cs).floor().max(0.0) as usize;
        let r1 = (((center.y + radius) / cs).ceil() as usize).min(self.rows);
        for r in r0..r1 {
            for c in c0..c1 {
                let cx = (c as f32 + 0.5) * cs;
                let cy = (r as f32 + 0.5) * cs;
                let dx = cx - center.x;
                let dy = cy - center.y;
                if dx * dx + dy * dy < r2 {
                    g[r * self.cols + c] = 0.0;
                }
            }
        }
    }

    /// Concentration at the cell containing `pos`. Used by brains to detect
    /// when they're sitting inside a saturated pheromone region (and so
    /// should stop trusting the gradient direction).
    pub fn sample(&self, ch: PheromoneChannel, pos: Vec2) -> f32 {
        let (c, r) = self.cell_of(pos);
        self.grids[Self::ch_idx(ch)][r * self.cols + c]
    }

    pub fn gradient(&self, ch: PheromoneChannel, pos: Vec2) -> Vec2 {
        let g = &self.grids[Self::ch_idx(ch)];
        let (c, r) = self.cell_of(pos);
        let here = g[r * self.cols + c];
        let mut acc = Vec2::ZERO;
        const NEIGHBORS: [(i32, i32); 8] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ];
        for &(dc, dr) in &NEIGHBORS {
            let nc = c as i32 + dc;
            let nr = r as i32 + dr;
            if nc < 0 || nr < 0 || nc >= self.cols as i32 || nr >= self.rows as i32 {
                continue;
            }
            let v = g[nr as usize * self.cols + nc as usize];
            let dir = Vec2::new(dc as f32, dr as f32).normalize_or_zero();
            acc += dir * (v - here);
        }
        // Very low threshold — we only filter out true noise (round-off /
        // empty regions). Faint distant gradients (a food pile's smell
        // reaching us from far across the map) need to make it through so
        // ants can still detect food from outside their vision radius.
        if acc.length_squared() < 1e-10 {
            Vec2::ZERO
        } else {
            acc.normalize()
        }
    }

    /// Per-tick update: diffuse to neighbors, then decay. Each channel is
    /// independent → process all five in parallel via rayon.
    pub fn decay_step(&mut self, walls: &[bool]) {
        use rayon::prelude::*;
        let cols = self.cols;
        let rows = self.rows;
        let decay = self.decay;
        let diffusion = self.diffusion;
        let has_walls = walls.iter().any(|wall| *wall);
        self.grids
            .par_iter_mut()
            .zip(self.scratches.par_iter_mut())
            .enumerate()
            .for_each(|(ch_idx, (grid, scratch))| {
                let d = if ch_idx == 0 && has_walls {
                    diffusion[ch_idx].min(WALL_HOME_DIFFUSION_CAP)
                } else {
                    diffusion[ch_idx]
                };
                let keep = 1.0 - d;
                let dec = if has_walls && ch_idx <= 1 {
                    decay[ch_idx].min(WALL_TRAIL_DECAY_CAP)
                } else {
                    decay[ch_idx]
                };
                // Walls block all pheromone and smell channels. Signals can
                // only diffuse around corners.
                let block_walls = true;
                std::mem::swap(grid, scratch);
                for r in 0..rows {
                    let row_base = r * cols;
                    for c in 0..cols {
                        let i = row_base + c;
                        if block_walls && walls[i] {
                            grid[i] = 0.0;
                            continue;
                        }
                        let here = scratch[i];
                        // Edges DRAIN (treat off-grid as 0) so pheromone
                        // hitting the world border evaporates instead of
                        // accumulating into a visible "wall" of color.
                        // Walls (interior obstacles) still use here (no
                        // flow into the wall, but nothing reflects back).
                        let up = if r > 0 {
                            if block_walls && walls[i - cols] {
                                here
                            } else {
                                scratch[i - cols]
                            }
                        } else {
                            0.0
                        };
                        let dn = if r < rows - 1 {
                            if block_walls && walls[i + cols] {
                                here
                            } else {
                                scratch[i + cols]
                            }
                        } else {
                            0.0
                        };
                        let lf = if c > 0 {
                            if block_walls && walls[i - 1] {
                                here
                            } else {
                                scratch[i - 1]
                            }
                        } else {
                            0.0
                        };
                        let rt = if c < cols - 1 {
                            if block_walls && walls[i + 1] {
                                here
                            } else {
                                scratch[i + 1]
                            }
                        } else {
                            0.0
                        };
                        let neighbor_avg = (up + dn + lf + rt) * 0.25;
                        let new_val = (here * keep + neighbor_avg * d) * dec;
                        grid[i] = if new_val < 0.01 { 0.0 } else { new_val };
                    }
                }
            });
    }

    /// Downsample one channel to a `target_cols × target_rows` byte image.
    /// Uses **max** over the source window (thin trails survive downsample)
    /// and a **sqrt gamma** ramp so low concentrations are still visible.
    pub fn snapshot_downsampled(
        &self,
        ch: PheromoneChannel,
        target_cols: usize,
        target_rows: usize,
        scale: f32,
    ) -> Vec<u8> {
        let g = &self.grids[Self::ch_idx(ch)];
        let mut out = vec![0u8; target_cols * target_rows];
        let sx = self.cols as f32 / target_cols as f32;
        let sy = self.rows as f32 / target_rows as f32;
        for ty in 0..target_rows {
            let r0 = (ty as f32 * sy) as usize;
            let r1 = (((ty + 1) as f32 * sy).ceil() as usize)
                .max(r0 + 1)
                .min(self.rows);
            for tx in 0..target_cols {
                let c0 = (tx as f32 * sx) as usize;
                let c1 = (((tx + 1) as f32 * sx).ceil() as usize)
                    .max(c0 + 1)
                    .min(self.cols);
                let mut peak = 0.0f32;
                for r in r0..r1 {
                    let row_base = r * self.cols;
                    for c in c0..c1 {
                        let v = g[row_base + c];
                        if v > peak {
                            peak = v;
                        }
                    }
                }
                let norm = (peak / scale).clamp(0.0, 1.0).sqrt();
                out[ty * target_cols + tx] = (norm * 255.0) as u8;
            }
        }
        out
    }
}
