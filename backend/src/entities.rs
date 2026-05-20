use glam::Vec2;

pub type EntityId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Queen,
    Worker,
    Soldier,
}

pub struct Ant {
    pub id: EntityId,
    pub colony: u8,
    pub role: Role,
    pub pos: Vec2,
    pub heading: f32,        // radians, smoothly slewed each tick
    pub target_heading: f32, // radians, the brain's commit goal
    pub energy: f32,         // 0..1, "feeding reserve", dies at 0 (starvation)
    pub hp: f32,             // 0..1, dies at 0 (combat)
    pub carrying_food: bool,
    /// Distance from the nest when this ant last picked up food. Used to
    /// distinguish long visible returns from short local food-cycle pickups.
    pub pickup_home_dist: f32,
    pub age: u32,     // ticks lived
    pub max_age: u32, // ticks before old-age death
    /// Sparse outbound breadcrumbs since the last nest visit. When the ant
    /// finds food, it can retrace these waypoints instead of using a hidden
    /// nest vector.
    pub breadcrumbs: Vec<Vec2>,
    pub return_path: Vec<Vec2>,
    /// World-tick at which this ant first entered a cell with Repellent
    /// above POISON_THRESHOLD. None until first exposure. Used to
    /// compute "how long pesticide took to kill" for the bench.
    pub first_poison_tick: Option<u32>,
    /// Ticks since the ant's most recent state change (picked up food or
    /// dropped food at nest). johnBuffer-style time-decayed deposit:
    /// `lay_strength × max(0, 1 - since_state/decay_horizon)`. A freshly-fed
    /// carrier lays full strength; the same carrier near nest lays almost
    /// zero. Same for outbound after leaving nest. Encodes direction
    /// intrinsically — no separate Home channel needed.
    pub since_state_change: u32,
}

pub struct Food {
    pub pos: Vec2,
    pub amount: f32,
}

/// A corpse marker kept for protocol compatibility. Normal ant deaths remove
/// the ant immediately and do not create one.
pub struct Corpse {
    pub pos: Vec2,
    pub ticks_remaining: u32,
    /// True if this ant died from pesticide poisoning. Poisoned corpses
    /// emit Repellent (the chemical signal of the toxin is in the body)
    /// for a while before decomposing — so other ants avoid them.
    pub poisoned: bool,
}

pub struct Nest {
    pub pos: Vec2,
    pub radius: f32,
    pub food_stored: f32,
    pub queen_id: Option<EntityId>,
}
