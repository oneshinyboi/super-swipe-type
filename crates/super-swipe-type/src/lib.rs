mod swipe_trajectory_processor;
mod keyboard_manager;
mod encoder;
mod swipe_orchestrator;
mod decoder;
mod beam_search;
#[cfg(test)]
mod tests;
mod dictionary;

use std::cmp::Ordering;
use crate::encoder::EncodeResult;
use std::ops::Sub;
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
#[derive(Clone)]
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

