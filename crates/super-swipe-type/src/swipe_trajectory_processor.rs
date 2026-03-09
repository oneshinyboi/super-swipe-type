use vector2::Vector2;
use crate::SwipePoint;
use crate::keyboard_manager::QwertyKeyboardGrid;


#[derive(Clone)]
pub(crate) struct FeaturePoint {
    pub(crate) point: Vector2,
    pub(crate) velocity: Vector2,
    pub(crate) acceleration: Vector2,
    pub(crate) nearest_key: char,
}
pub struct SwipeTrajectoryProcessor {
    max_sequence_length: usize,
    keyboard_grid: QwertyKeyboardGrid
}
/// turns swipe points into feature points
impl SwipeTrajectoryProcessor {
    pub fn new(max_sequence_length: usize) -> Self {
        Self {
            max_sequence_length,
            keyboard_grid: QwertyKeyboardGrid::new()
        }
    }
    pub fn extract_features(&self, normalized_swipe_input: Vec<SwipePoint>) -> Vec<FeaturePoint> {
        // resample input to fit max length
        let mut normalized_swipe_input = normalized_swipe_input;
        if normalized_swipe_input.len() > self.max_sequence_length {
            normalized_swipe_input = Self::resample_points(normalized_swipe_input, self.max_sequence_length);
        }
        self.calculate_features(&normalized_swipe_input)
    }
    fn calculate_features(&self, swipe_points: &Vec<SwipePoint>) -> Vec<FeaturePoint> {
        let n = swipe_points.len();
        let mut out = Vec::new();
        let mut dt = Vec::new();

        dt.push(0);
        for i in 1..n {
            dt.push(swipe_points[i].timestamp.as_millis()-swipe_points[i-1].timestamp.as_millis())
        }

        out.push(FeaturePoint {
            point: swipe_points[0].point.clone(),
            velocity: Vector2::ZERO,
            acceleration: Vector2::ZERO,
            nearest_key: self.keyboard_grid.get_nearest_key(&swipe_points[0].point)
        });

        // calculate velocities
        for i in 1..n {
            out.push(FeaturePoint {
                point: swipe_points[i].point.clone(),
                velocity: (swipe_points[i].point - swipe_points[i-1].point) / dt[i] as f64,
                acceleration: Vector2::ZERO,
                nearest_key: self.keyboard_grid.get_nearest_key(&swipe_points[i].point)
            })
        }

        // calculate acceleration
        for i in 1..n {
            out[i].acceleration = (out[i].velocity - out[i-1].velocity) / dt[i] as f64
        }
        out
    }
    fn resample_points(swipe_points: Vec<SwipePoint>, target_length: usize) -> Vec<SwipePoint> {
        if swipe_points.len() <= 2 {
            return swipe_points;
        }

        let mut out = Vec::with_capacity(target_length);

        // Always include the first point
        out.push(swipe_points[0].clone());

        let num_middle = target_length.saturating_sub(2);
        let original_length = swipe_points.len();
        let available_range = (original_length as f32 - 2.0).max(0.0);

        if num_middle == 0 {
            // If target_length is 2 or less, just add the last point
            out.push(swipe_points[original_length - 1].clone());
            return out;
        }

        // Split into 3 zones: start (35%), middle (30%), end (35%)
        let start_zone_end = 1 + (available_range * 0.35) as usize;
        let end_zone_start = (original_length - 1).saturating_sub((available_range * 0.35) as usize);

        let points_in_start = ((num_middle as f32) * 0.35) as usize;
        let points_in_end = ((num_middle as f32) * 0.35) as usize;
        let points_in_middle = num_middle.saturating_sub(points_in_start + points_in_end);

        // Sample from start zone
        for i in 0..points_in_start {
            let idx = if points_in_start > 1 {
                1 + (i * (start_zone_end - 1)) / (points_in_start - 1)
            } else {
                1
            };
            if idx < swipe_points.len() {
                out.push(swipe_points[idx].clone());
            }
        }

        // Sample from middle zone
        let middle_zone_size = end_zone_start.saturating_sub(start_zone_end);
        if points_in_middle > 0 && middle_zone_size > 0 {
            for i in 0..points_in_middle {
                let idx = if points_in_middle > 1 {
                    start_zone_end + (i * middle_zone_size) / (points_in_middle - 1)
                } else {
                    start_zone_end
                };
                if idx < swipe_points.len() {
                    out.push(swipe_points[idx].clone());
                }
            }
        }

        // Sample from end zone
        let end_zone_size = (original_length - 1).saturating_sub(end_zone_start);
        if points_in_end > 0 && end_zone_size > 0 {
            for i in 0..points_in_end {
                let idx = if points_in_end > 1 {
                    end_zone_start + (i * end_zone_size) / (points_in_end - 1)
                } else {
                    end_zone_start
                };
                if idx < swipe_points.len() {
                    out.push(swipe_points[idx].clone());
                }
            }
        }

        // Always include the last point
        out.push(swipe_points[original_length - 1].clone());

        out
    }
}
