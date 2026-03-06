use std::collections::HashMap;
use vector2::Vector2;
use crate::QwertyKeyboardGrid;

// use same keyboard specs as used in training models
const KEY_WIDTH: f64 = 0.1;
const ROW_HEIGHT: f64 = 1.0/3.0;
const ROW_0: &str = "qwertyuiop";
const ROW_1: &str = "asdfghjkl";
const ROW_2: &str = "zxcvbnm";
const ROW_0_OFFSET: f64 = 0.0;
const ROW_1_OFFSET: f64 = 0.05;
const ROW_2_OFFSET: f64 = 0.15;


impl QwertyKeyboardGrid {
    pub fn new() -> Self {
        Self {
            key_positions: Self::build_keyboard_positions()
        }
    }
    fn build_keyboard_positions() -> HashMap<char, Vector2> {
        let mut out = HashMap::new();
        for (i, char) in ROW_0.chars().enumerate() {
            out.insert(char, Vector2 {
                x: ROW_0_OFFSET + i as f64 * KEY_WIDTH,
                y: ROW_HEIGHT * 0.0 + ROW_HEIGHT / 2.0
            });
        }
        for (i, char ) in ROW_1.chars().enumerate() {
            out.insert(char, Vector2 {
                x: ROW_1_OFFSET + i as f64 * KEY_WIDTH,
                y: ROW_HEIGHT * 1.0 + ROW_HEIGHT / 2.0
            });
        }
        for (i, char) in ROW_2.chars().enumerate() {
            out.insert(char, Vector2 {
                x: ROW_2_OFFSET + i as f64 * KEY_WIDTH,
                y: ROW_HEIGHT * 1.0 + ROW_HEIGHT / 2.0
            });
        }
        out

    }
    pub fn get_nearest_key(&self, point: &Vector2) -> char {
        let clamped_point = Vector2 {
            x: point.x.clamp(0.0, 1.0),
            y: point.y.clamp(0.0, 1.0)
        };
        let mut closest_char = 'a';
        let mut shortest_dist = f64::MAX;

        for (key, pos) in &self.key_positions {
            let dist = (*pos-clamped_point).sqr_magnitude();
            if dist < shortest_dist {
                shortest_dist = dist;
                closest_char = *key;
            }
        }
        closest_char
    }
    // convert chars to token index for models
    pub fn get_char_token_index(char: char) -> i32 {
        if char >= 'a' && char <= 'z' {
            char as i32 - 'a' as i32 + 4
        } else {
            0 // padding index
        }
    }
}