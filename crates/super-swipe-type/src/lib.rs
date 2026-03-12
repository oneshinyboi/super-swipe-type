mod swipe_trajectory_processor;
pub mod keyboard_manager;
mod encoder;
pub mod swipe_orchestrator;
mod decoder;
mod beam_search;
#[cfg(test)]
mod tests;
mod dictionary;

use std::cmp::Ordering;
use std::time::Duration;
use vector2::Vector2;

// todo: wordlist, orchestrator, vocab trie
// done: feature extractor, decoder, encoder, beam search

const PAD_IDX: u8 = 0;
pub const SOS_IDX: u8 = 2;
pub const EOS_IDX: u8 = 3;
const DECODER_SEQ_LEN: u8 = 20; // Must match model export
#[derive(Clone, Debug)]
pub struct SwipePoint {
    point: Vector2,
    timestamp: Duration,
}
impl SwipePoint {
    pub fn new(x_pos: f64, y_pos: f64, timestamp: Duration) -> Self {
        Self {
            point: Vector2 {x: x_pos, y: y_pos},
            timestamp,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SwipeCandidate {
    pub word: String,
    pub confidence: f32,
}
impl Eq for SwipeCandidate {}

impl PartialEq<Self> for SwipeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}

impl PartialOrd<Self> for SwipeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.confidence.partial_cmp(&other.confidence)
    }
}

impl Ord for SwipeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

