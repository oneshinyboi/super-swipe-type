use std::collections::HashMap;
use vector2::Vector2;

// use same keyboard specs as used in training models
const KEY_WIDTH: f64 = 0.1;
const ROW_HEIGHT: f64 = 1.0 / 3.0;
const ROW_0: &str = "qwertyuiop";
const ROW_1: &str = "asdfghjkl";
const ROW_2: &str = "zxcvbnm";
const ROW_0_OFFSET: f64 = 0.0;
const ROW_1_OFFSET: f64 = 0.05;
const ROW_2_OFFSET: f64 = 0.15;

/// A QWERTY keyboard layout expressed as a map from key characters to their
/// 2-D centre positions in the same normalised coordinate space used by
/// [`SwipePoint`](crate::SwipePoint).
///
/// The grid matches the keyboard specification used when training the
/// underlying neural models, so it must be used when interpreting swipe
/// coordinates.
///
/// Coordinate conventions:
/// - `x ∈ [0.0, 1.0]` spans the full keyboard width (Q–P row, left to right).
/// - `y ∈ [0.0, 1.0]` spans the full keyboard height (top row to bottom row).
///
/// Key width is `0.1` (10 keys across the Q–P row).  
/// Row height is `1/3` (three rows of keys).
#[derive(Debug)]
pub struct QwertyKeyboardGrid {
    /// Maps every lowercase letter (`'a'`–`'z'`) to its key-centre position.
    pub key_positions: HashMap<char, Vector2>,
}
impl QwertyKeyboardGrid {
    /// Constructs a [`QwertyKeyboardGrid`] pre-populated with the standard
    /// QWERTY key positions used during model training.
    pub fn new() -> Self {
        Self {
            key_positions: Self::build_keyboard_positions(),
        }
    }

    /// Returns the height of a single key row in normalised coordinates
    /// (`1.0 / 3.0 ≈ 0.333`).
    pub fn get_key_height() -> f64 {
        ROW_HEIGHT
    }

    /// Returns the width of a single key in normalised coordinates (`0.1`).
    pub fn get_key_width() -> f64 {
        KEY_WIDTH
    }

    fn build_keyboard_positions() -> HashMap<char, Vector2> {
        let mut out = HashMap::new();
        for (i, char) in ROW_0.chars().enumerate() {
            out.insert(
                char,
                Vector2 {
                    x: ROW_0_OFFSET + i as f64 * KEY_WIDTH + KEY_WIDTH / 2.0,
                    y: ROW_HEIGHT * 0.0 + ROW_HEIGHT / 2.0,
                },
            );
        }
        for (i, char) in ROW_1.chars().enumerate() {
            out.insert(
                char,
                Vector2 {
                    x: ROW_1_OFFSET + i as f64 * KEY_WIDTH + KEY_WIDTH / 2.0,
                    y: ROW_HEIGHT * 1.0 + ROW_HEIGHT / 2.0,
                },
            );
        }
        for (i, char) in ROW_2.chars().enumerate() {
            out.insert(
                char,
                Vector2 {
                    x: ROW_2_OFFSET + i as f64 * KEY_WIDTH + KEY_WIDTH / 2.0,
                    y: ROW_HEIGHT * 2.0 + ROW_HEIGHT / 2.0,
                },
            );
        }
        out
    }
    /// Returns the character whose key centre is closest to `point`.
    ///
    /// The input is clamped to `[0.0, 1.0]` on both axes before the search,
    /// so out-of-bounds coordinates map to the nearest edge key rather than
    /// producing an error.
    ///
    /// Distance is measured as squared Euclidean distance between `point`
    /// and each key's centre position.
    pub fn get_nearest_key(&self, point: &Vector2) -> char {
        let clamped_point = Vector2 {
            x: point.x.clamp(0.0, 1.0),
            y: point.y.clamp(0.0, 1.0),
        };
        let mut closest_char = 'a';
        let mut shortest_dist = f64::MAX;

        for (key, pos) in &self.key_positions {
            let dist = (*pos - clamped_point).sqr_magnitude();
            if dist < shortest_dist {
                shortest_dist = dist;
                closest_char = *key;
            }
        }
        closest_char
    }
}

pub(crate) struct KeyTokenizer {}

impl KeyTokenizer {
    pub fn char_to_index(char: char) -> u8 {
        if char >= 'a' && char <= 'z' {
            char as u8 - 'a' as u8 + 4
        } else {
            0 // padding index
        }
    }
    pub fn index_to_char(i: u8) -> Option<char> {
        if 4 <= i && i <= 29 {
            Some((i - 4 + 'a' as u8) as char)
        } else {
            None
        }
    }
    pub fn indices_to_string(indices: &Vec<u8>) -> String {
        let mut out = String::new();
        indices
            .iter()
            .for_each(|i| out.push(Self::index_to_char(*i).unwrap_or_default()));
        out.chars().filter(|c| *c != char::default()).collect()
    }
}
