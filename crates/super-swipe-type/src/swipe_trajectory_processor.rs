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
        for window in swipe_points.windows(2) {
            out.push(FeaturePoint {
                point: window[1].point.clone(),
                velocity: (window[1].point - swipe_points[0].point) / dt[1] as f64,
                acceleration: Vector2::ZERO,
                nearest_key: self.keyboard_grid.get_nearest_key(&window[1].point)
            })
        }

        // calculate acceleration
        for i in 1..n {
            out[i].acceleration = (out[i].velocity - out[i-1].velocity) / dt[i] as f64
        }
        out
    }
    fn resample_points(swipe_points: Vec<SwipePoint>, target_length: usize) -> Vec<SwipePoint> {
        let mut out = Vec::new();

        let num_middle = target_length - 2;
        let original_length = swipe_points.len();
        let available_range = original_length as f32 - 2f32;

        // Use weighted selection: more points at start/end
        // Split into 3 zones: start (30%), middle (40%), end (30%)
        let start_zone_end = 1 + (available_range * 0.3) as usize;
        let end_zone_start = original_length - 1 - (available_range * 0.3) as usize;

        let points_in_start = num_middle*0.35 as usize;
        let points_in_end = num_middle*0.35 as usize;
        let points_in_middle = num_middle - points_in_start - points_in_end;

        for i in 0..points_in_start {
            let idx = 1 + (i* (start_zone_end - 1)) / points_in_start;
            out.push(swipe_points[idx].clone());

        }
        let middle_zone_size = end_zone_start - start_zone_end;
        for i in 0..points_in_start {
            let idx = start_zone_end + (i * middle_zone_size) / points_in_middle;
            out.push(swipe_points[idx].clone());
        }
        let end_zone_size = (original_length - 1) - end_zone_start;
        for i in 0..points_in_start {
            let idx = end_zone_start + (i * end_zone_size) / points_in_end;
            out.push(swipe_points[idx].clone())
        }
        out
    }
}
