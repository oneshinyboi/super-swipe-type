use ort::value::{DynTensor, Tensor};
use std::collections::HashMap;
use ort::Error;
use ort::session::Session;
use crate::keyboard_manager::KeyTokenizer;
use crate::swipe_trajectory_processor::FeaturePoint;

const INPUT_TRAJECTORY_FEATURES: &str = "trajectory_features";
const INPUT_NEAREST_KEYS: &str = "nearest_keys";
const INPUT_ACTUAL_LENGTH: &str = "actual_length";
#[derive(Debug)]
pub(crate) struct EncodeResult {
    pub memory_tensor: Tensor<f32>,
    pub actual_length_tensor: Tensor<i32>,
}
#[derive(Debug)]
pub(crate) struct Encoder {
    pub(crate) session: Session,
    pub(crate) max_sequence_length: usize
}
impl Encoder {
    // encodes swipe features and returns memory tensor of encoder
    pub fn encode(&mut self, features: Vec<FeaturePoint>) -> Result<EncodeResult, Error> {
        let trajectory_tensor = self.create_trajectory_tensor(&features)?;
        let nearest_keys_tensor = self.create_nearest_keys_tensor(&features)?;
        let actual_length_tensor = Tensor::from_array(([1], vec![features.len() as i32].to_vec()))?;

        let mut encoder_input: HashMap<&str, DynTensor> = HashMap::new();
        encoder_input.insert(INPUT_TRAJECTORY_FEATURES, trajectory_tensor.upcast());
        encoder_input.insert(INPUT_NEAREST_KEYS, nearest_keys_tensor.upcast());
        encoder_input.insert(INPUT_ACTUAL_LENGTH, actual_length_tensor.clone().upcast());

        let outputs = self.session.run(encoder_input)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(EncodeResult {
            memory_tensor: Tensor::from_array((shape.clone(), data.to_vec()))?,
            actual_length_tensor
        })
    }

    fn create_trajectory_tensor(&self, features: &Vec<FeaturePoint>) -> ort::Result<Tensor<f32>> {
        let mut feature_array= Vec::new();
        for feature_point in features {
            feature_array.push(feature_point.point.x as f32);
            feature_array.push(feature_point.point.y as f32);
            feature_array.push(feature_point.velocity.x as f32);
            feature_array.push(feature_point.velocity.y as f32);
            feature_array.push(feature_point.acceleration.x as f32);
            feature_array.push(feature_point.acceleration.y as f32);
        }
        feature_array.resize(self.max_sequence_length*6, 0.0);
        Tensor::from_array(([1, self.max_sequence_length, 6], feature_array))
    }

    fn create_nearest_keys_tensor(&self, features: &Vec<FeaturePoint>) -> ort::Result<Tensor<i32>> {
        let mut feature_array: Vec<i32> = Vec::new();
        for feature_point in features {
            feature_array.push(KeyTokenizer::char_to_index(feature_point.nearest_key) as i32);
        }
        feature_array.resize(self.max_sequence_length, 0);
        Tensor::from_array(([1, self.max_sequence_length], feature_array))
    }
}