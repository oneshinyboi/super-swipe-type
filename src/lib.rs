mod swipe_trajectory_processor;
mod keyboard_grid;
mod tensor_factory;
mod encoder;
mod swipe_orchestrator;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Sub;
use std::rc::Rc;
use std::time::{Duration, Instant};
use ort::memory::{AllocatedBlock, Allocator};
use ort::session::Session;
use vector2::Vector2;

// todo: encoder, decoder, beamsearch, wordlist
// done: feature extracter
pub struct SwipeOrchestrator {
    ort_environment: RefCell<OrtEnvironment>
}
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
    nearest_key: char,
}
struct QwertyKeyboardGrid {
    key_positions: HashMap<char, Vector2>
}
struct SwipeTrajectoryProcessor {
    max_sequence_length: usize,
    keyboard_grid: QwertyKeyboardGrid
}
struct OrtEnvironment {
    session: Session,
    allocator: Allocator
}
struct TensorFactory {
    max_sequence_length: usize
}
struct Encoder {
    ort_environment: RefCell<OrtEnvironment>,
    tensor_factory: TensorFactory
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_orchestrator() {
        let _swipe_orchestrator = SwipeOrchestrator::new();
    }

    #[test]
    fn test_encoder() {
        let swipe_orchestrator = SwipeOrchestrator::new();
        let keyboard_manager = QwertyKeyboardGrid::new();
        let encoder = Encoder {
            ort_environment: swipe_orchestrator.ort_environment,
            tensor_factory: TensorFactory { max_sequence_length: 250}
        };
        let features = vec![
            FeaturePoint{
                point: Vector2 {x: 0.2, y: 0.4},
                velocity: Default::default(),
                acceleration: Default::default(),
                nearest_key: keyboard_manager.get_nearest_key(&Vector2 { x: 0.2, y: 0.4 }),
            },
            FeaturePoint {
                point: Vector2 {x: 0.7, y: 0.3},
                velocity: Default::default(),
                acceleration: Default::default(),
                nearest_key: keyboard_manager.get_nearest_key(&Vector2 { x: 0.7, y: 0.3 }),
        }];
        encoder.encode(features);
    }

}

