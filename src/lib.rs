mod swipe_trajectory_processor;
mod keyboard_grid;
mod encoder;
mod swipe_orchestrator;
mod decoder;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Sub;
use std::rc::Rc;
use std::time::{Duration, Instant};
use ort::memory::{AllocatedBlock, Allocator};
use ort::session::Session;
use ort::value::Tensor;
use vector2::Vector2;
use crate::encoder::EncodeResult;

// todo: beam search, wordlist, orchestrator
// done: feature extractor, decoder, encoder
pub struct SwipeOrchestrator {
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

struct Encoder {
    session: Session,
    max_sequence_length: usize
}
struct Decoder {
    session: Session,
    encode_result: EncodeResult,
    max_sequence_length: usize
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_encoder() {
        let session = Session::builder().unwrap().commit_from_file("assets/swipe_encoder_android.onnx").unwrap();
        let keyboard_manager = QwertyKeyboardGrid::new();
        let encoder = Encoder {
            session,
            max_sequence_length: 250
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

        assert!(encoder.encode(features).is_ok());
    }

    #[test]
    fn test_decoder() {
        let encoder_session = Session::builder().unwrap().commit_from_file("assets/swipe_encoder_android.onnx").unwrap();
        let decoder_session = Session::builder().unwrap().commit_from_file("assets/swipe_decoder_android.onnx").unwrap();
        let keyboard_manager = QwertyKeyboardGrid::new();

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
        let encoder = Encoder {
            session: encoder_session,
            max_sequence_length: 250
        };
        let result = encoder.encode(features);
        assert!(result.is_ok());

        let decoder = Decoder {
            session: decoder_session,
            encode_result: result.unwrap(),
            max_sequence_length: 250,
        };

        let decode_result = decoder.decode_sequential(&vec![decoder::SOS_IDX].to_vec());
        assert!(decode_result.is_ok());
    }

}

