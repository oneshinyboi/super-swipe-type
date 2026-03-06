use crate::{Encoder, FeaturePoint};
use ort::value::{DynTensor, Tensor};
use std::collections::HashMap;
const INPUT_TRAJECTORY_FEATURES: &str = "trajectory_features";
const INPUT_NEAREST_KEYS: &str = "nearest_keys";
const INPUT_ACTUAL_LENGTH: &str = "actual_length";
impl Encoder {
    pub fn encode(&self, features: Vec<FeaturePoint>) {
        let trajectory_tensor = self.tensor_factory.create_trajectory_tensor(&features);
        let nearest_keys_tensor = self.tensor_factory.create_nearest_keys_tensor(&features);
        let actual_length_tensor = Tensor::from_array(([1], vec![features.len() as i32].to_vec()));

        if let (Ok(trajectory_tensor),Ok(nearest_keys_tensor), Ok(actual_length_tensor)) = (trajectory_tensor, nearest_keys_tensor, actual_length_tensor) {
            let mut ort_env = self.ort_environment.borrow_mut();

            let mut encoder_input: HashMap<&str, DynTensor> = HashMap::new();
            encoder_input.insert(INPUT_TRAJECTORY_FEATURES, trajectory_tensor.upcast());
            encoder_input.insert(INPUT_NEAREST_KEYS, nearest_keys_tensor.upcast());
            encoder_input.insert(INPUT_ACTUAL_LENGTH, actual_length_tensor.upcast());

            let outputs = ort_env.session.run(encoder_input).expect("TODO: panic message");
            let mem = &outputs[0].try_extract_tensor::<f32>();
        }
    }
}