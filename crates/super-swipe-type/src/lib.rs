mod beam_search;
mod decoder;
mod dictionary;
mod encoder;
pub mod keyboard_manager;
pub mod swipe_orchestrator;
mod swipe_trajectory_processor;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::time::Duration;
use vector2::Vector2;

const PAD_IDX: u8 = 0;

const SOS_IDX: u8 = 2;

const EOS_IDX: u8 = 3;

const DECODER_SEQ_LEN: u8 = 20; // Must match model export

/// A single raw point captured during a swipe gesture.
///
/// Coordinates are **normalised** over the QWERTY keyboard area:
/// - `(0.0, 0.0)` is the top-left corner of the **Q** key.
/// - `(1.0, 0.0)` is the top-right corner of the **P** key.
/// - `(0.0, 1.0)` / `(1.0, 1.0)` are the corresponding bottom corners.
///
/// Construct with [`SwipePoint::new`] and collect a sequence of these to
/// pass to [`swipe_orchestrator::SwipeOrchestrator::predict`].
#[derive(Clone, Debug)]
pub struct SwipePoint {
    point: Vector2,
    timestamp: Duration,
}
impl SwipePoint {
    /// Creates a new [`SwipePoint`].
    ///
    /// # Parameters
    /// - `x_pos` – Normalised horizontal position in `[0.0, 1.0]`.
    /// - `y_pos` – Normalised vertical position in `[0.0, 1.0]`.
    /// - `timestamp` – Time elapsed since the start of the swipe gesture.
    pub fn new(x_pos: f64, y_pos: f64, timestamp: Duration) -> Self {
        Self {
            point: Vector2 { x: x_pos, y: y_pos },
            timestamp,
        }
    }
}

/// A predicted word candidate returned by [`swipe_orchestrator::SwipeOrchestrator::predict`].
///
/// The results vector is sorted in descending order of `confidence`, so
/// `results[0]` is always the most likely prediction.
///
/// # Ordering
/// [`SwipeCandidate`] implements [`Ord`] / [`PartialOrd`] by `confidence`,
/// and [`Eq`] / [`PartialEq`] by `word`.
#[derive(Clone, Debug)]
pub struct SwipeCandidate {
    /// The predicted word string.
    pub word: String,
    /// Log-probability–based confidence score.  Higher values indicate a
    /// more likely prediction.
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
