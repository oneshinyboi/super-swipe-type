mod swipe_trajectory_processor;
mod keyboard_grid;

use std::collections::HashMap;
use std::ops::Sub;
use std::time::{Duration, Instant};
use vector2::Vector2;

// todo: encoder, decoder, beamsearch, wordlist
// done: feature extracter

#[derive(Clone)]
pub struct SwipePoint {
    point: Vector2,
    timestamp: Duration,
}
#[derive(Clone)]
struct FeaturePoint {
    point: Vector2,
    velocity: Vector2,
    acceleration: Vector2,
    nearest_key: Option<char>,
}
struct QwertyKeyboardGrid {
    key_positions: HashMap<char, Vector2>
}
struct SwipeTrajectoryProcessor {
    keyboard_grid: QwertyKeyboardGrid
}
impl FeaturePoint {
    pub fn zero() -> Self {
        Self {point: Vector2::ZERO, velocity: Vector2::ZERO, acceleration: Vector2::ZERO, nearest_key: None}
    }
}


