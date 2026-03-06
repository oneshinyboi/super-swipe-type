use ort::value::{Tensor};
use crate::{TensorFactory, FeaturePoint, QwertyKeyboardGrid};

impl TensorFactory {
    pub fn create_trajectory_tensor(&self, features: &Vec<FeaturePoint>) -> ort::Result<Tensor<f32>> {
        let mut feature_array= Vec::new();
        for feature_point in features {
            feature_array.push(feature_point.point.x as f32);
            feature_array.push(feature_point.point.y as f32);
            feature_array.push(feature_point.velocity.x as f32);
            feature_array.push(feature_point.velocity.y as f32);
            feature_array.push(feature_point.acceleration.x as f32);
            feature_array.push(feature_point.acceleration.y as f32);
        }
        for i in features.len()..self.max_sequence_length {
            for i in 0..6 {feature_array.push(0.0)}
        }
        Tensor::from_array(([1, self.max_sequence_length, 6], feature_array))
    }
    pub fn create_nearest_keys_tensor(&self, features: &Vec<FeaturePoint>) -> ort::Result<Tensor<i32>> {
        let mut feature_array = Vec::new();
        for feature_point in features {
            feature_array.push(QwertyKeyboardGrid::get_char_token_index(feature_point.nearest_key));
        }
        for i in features.len()..self.max_sequence_length {
            feature_array.push(0);
        }
        Tensor::from_array(([1, self.max_sequence_length], feature_array))
    }

}