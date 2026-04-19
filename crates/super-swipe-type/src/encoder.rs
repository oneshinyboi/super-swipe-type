use crate::keyboard_manager::KeyTokenizer;
use crate::swipe_trajectory_processor::FeaturePoint;
use anyhow::Result;
use tract_onnx::prelude::*;

pub(crate) type OnnxSession = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug)]
pub(crate) struct EncodeResult {
    pub memory_data: Vec<f32>,
    pub memory_shape: TVec<usize>,
    pub actual_length: i32,
}

#[derive(Debug)]
pub(crate) struct Encoder {
    pub(crate) session: OnnxSession,
    pub(crate) max_sequence_length: usize,
}

impl Encoder {
    /// Encodes swipe features and returns the encoder memory tensor.
    pub fn encode(&mut self, features: Vec<FeaturePoint>) -> Result<EncodeResult> {
        let actual_length = features.len() as i32;

        let trajectory = self.create_trajectory_array(&features);
        let nearest_keys = self.create_nearest_keys_array(&features);
        let length_array = tract_ndarray::arr1(&[actual_length]).into_shape_with_order((1,))?;

        let trajectory_tensor: Tensor = trajectory.into();
        let nearest_keys_tensor: Tensor = nearest_keys.into();
        let length_tensor: Tensor = length_array.into();

        // The model's inputs are ordered: trajectory_features, nearest_keys, actual_length
        let outputs = self.session.run(tvec![
            trajectory_tensor.into(),
            nearest_keys_tensor.into(),
            length_tensor.into(),
        ])?;

        let memory_view = outputs[0].to_array_view::<f32>()?;
        let memory_shape: TVec<usize> = memory_view.shape().iter().copied().collect();
        let memory_data: Vec<f32> = memory_view.iter().copied().collect();

        Ok(EncodeResult {
            memory_data,
            memory_shape,
            actual_length,
        })
    }

    fn create_trajectory_array(
        &self,
        features: &[FeaturePoint],
    ) -> tract_ndarray::Array3<f32> {
        let mut data = vec![0.0f32; self.max_sequence_length * 6];
        for (i, fp) in features.iter().enumerate() {
            let base = i * 6;
            data[base]     = fp.point.x as f32;
            data[base + 1] = fp.point.y as f32;
            data[base + 2] = fp.velocity.x as f32;
            data[base + 3] = fp.velocity.y as f32;
            data[base + 4] = fp.acceleration.x as f32;
            data[base + 5] = fp.acceleration.y as f32;
        }
        tract_ndarray::Array3::from_shape_vec((1, self.max_sequence_length, 6), data)
            .expect("trajectory array shape is always valid")
    }

    fn create_nearest_keys_array(
        &self,
        features: &[FeaturePoint],
    ) -> tract_ndarray::Array2<i32> {
        let mut data = vec![0i32; self.max_sequence_length];
        for (i, fp) in features.iter().enumerate() {
            data[i] = KeyTokenizer::char_to_index(fp.nearest_key) as i32;
        }
        tract_ndarray::Array2::from_shape_vec((1, self.max_sequence_length), data)
            .expect("nearest_keys array shape is always valid")
    }
}
