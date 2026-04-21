use crate::keyboard_manager::KeyTokenizer;
use crate::swipe_trajectory_processor::FeaturePoint;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Arc;
use tract_onnx::prelude::{IntoRunnable, IntoTensor, Runnable, RunnableModel, TVec, Tensor, TractResult, TypedFact, TypedModel, TypedOp};

const INPUT_TRAJECTORY_FEATURES: &str = "trajectory_features";
const INPUT_NEAREST_KEYS: &str = "nearest_keys";
const INPUT_ACTUAL_LENGTH: &str = "actual_length";
#[derive(Debug)]
pub(crate) struct EncodeResult {
    pub memory_tensor: Tensor,
    pub actual_length_tensor: Tensor,
}
#[derive(Debug)]
pub(crate) struct Encoder {
    pub(crate) model: Arc<RunnableModel<TypedFact,Box<dyn TypedOp>>>,
    pub(crate) max_sequence_length: usize,
}
impl Encoder {
    // encodes swipe features and returns memory tensor of encoder
    pub fn encode(&mut self, features: Vec<FeaturePoint>) -> TractResult<EncodeResult> {
        let trajectory_tensor = self.create_trajectory_tensor(&features)?;
        let nearest_keys_tensor = self.create_nearest_keys_tensor(&features)?;
        let actual_length_tensor = Tensor::from_shape([1].as_slice(), vec![features.len() as i32].as_slice())?;

        let mut input_features= Vec::new();
        input_features.push(trajectory_tensor.into());
        input_features.push(nearest_keys_tensor.into());
        input_features.push(actual_length_tensor.clone().into());

        let output = self.model.run(input_features.into())?;

        let mut memory_tensor = output.into_iter().next().unwrap().into_tensor();
        //memory_tensor.set_shape(&[1, self.max_sequence_length, 256])?;

        Ok(EncodeResult {
            memory_tensor,
            actual_length_tensor,
        })
    }

    fn create_trajectory_tensor(&self, features: &Vec<FeaturePoint>) -> TractResult<Tensor> {
        let mut feature_array = Vec::new();
        for feature_point in features {
            feature_array.push(feature_point.point.x as f32);
            feature_array.push(feature_point.point.y as f32);
            feature_array.push(feature_point.velocity.x as f32);
            feature_array.push(feature_point.velocity.y as f32);
            feature_array.push(feature_point.acceleration.x as f32);
            feature_array.push(feature_point.acceleration.y as f32);
        }
        feature_array.resize(self.max_sequence_length * 6, 0.0);
        Tensor::from_shape([1, self.max_sequence_length, 6].as_slice(), feature_array.as_slice())
    }

    fn create_nearest_keys_tensor(&self, features: &Vec<FeaturePoint>) -> TractResult<Tensor> {
        let mut feature_array: Vec<i32> = Vec::new();
        for feature_point in features {
            feature_array.push(KeyTokenizer::char_to_index(feature_point.nearest_key) as i32);
        }
        feature_array.resize(self.max_sequence_length, 0);
        Tensor::from_shape([1, self.max_sequence_length].as_slice(), feature_array.as_slice())
    }
}
