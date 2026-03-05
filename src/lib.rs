mod swipe_trajectory_processor;

use std::ops::Sub;
use std::time::{Duration, Instant};
use vector2::Vector2;

// todo: encoder, decoder, beamsearch, wordlist
// done: feature extracter

#[derive(Clone)]
struct SwipePoint {
    point: Vector2,
    timestamp: Duration,
}
#[derive(Clone)]
struct FeaturePoint {
    point: Vector2,
    velocity: Vector2,
    acceleration: Vector2,
}
impl FeaturePoint {
    pub fn zero() -> Self {
        Self {point: Vector2::ZERO, velocity: Vector2::ZERO, acceleration: Vector2::ZERO}
    }
}


