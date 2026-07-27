use crate::keyboard_manager::KeyTokenizer;
use crate::swipe_trajectory_processor::FeaturePoint;
use anyhow::Result;
use rten::{Model, NodeId, Value};

#[derive(Debug)]
pub(crate) struct EncodeResult {
    pub memory_tensor: Value,
    pub actual_length_tensor: Value,
}
#[derive(Debug)]
pub(crate) struct Encoder {
    pub(crate) model: Model,
    pub(crate) max_sequence_length: usize,
    input_trajectory: NodeId,
    input_keys: NodeId,
    input_length: NodeId,
    output_node: NodeId,
}
impl Encoder {
    pub fn new(model: Model, max_sequence_length: usize) -> Result<Self> {
        Ok(Self {
            input_trajectory: model.node_id("trajectory_features")?,
            input_keys: model.node_id("nearest_keys")?,
            input_length: model.node_id("actual_length")?,
            output_node: model.node_id("encoder_output")?,
            model,
            max_sequence_length,
        })
    }

    pub fn encode(&mut self, features: Vec<FeaturePoint>) -> Result<EncodeResult> {
        let trajectory_tensor = self.create_trajectory_tensor(&features)?;
        let nearest_keys_tensor = self.create_nearest_keys_tensor(&features)?;
        let actual_length = features.len() as i32;

        let actual_length_value = Value::from_shape([1], vec![actual_length])?;

        let inputs = vec![
            (self.input_trajectory, trajectory_tensor.into()),
            (self.input_keys, nearest_keys_tensor.into()),
            (self.input_length, actual_length_value.clone().into()),
        ];
        let outputs = [self.output_node];
        let [memory_tensor] = self
            .model
            .run_n(inputs, outputs, None)
            .map_err(|e| anyhow::anyhow!("Encoder inference failed: {}", e))?;

        Ok(EncodeResult {
            memory_tensor,
            actual_length_tensor: actual_length_value,
        })
    }

    fn create_trajectory_tensor(&self, features: &Vec<FeaturePoint>) -> Result<Value> {
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
        Ok(Value::from_shape(
            [1, self.max_sequence_length, 6],
            feature_array,
        )?)
    }

    fn create_nearest_keys_tensor(&self, features: &Vec<FeaturePoint>) -> Result<Value> {
        let mut feature_array: Vec<i32> = Vec::new();
        for feature_point in features {
            feature_array.push(KeyTokenizer::char_to_index(feature_point.nearest_key) as i32);
        }
        feature_array.resize(self.max_sequence_length, 0);
        Ok(Value::from_shape(
            [1, self.max_sequence_length],
            feature_array,
        )?)
    }
}