use std::cmp::{max, min};
use std::collections::HashMap;
use std::iter::repeat;
use ort::Error;
use ort::value::Tensor;
use crate::Decoder;
const PAD_IDX: i32 = 0;
const SOS_IDX: i32= 2;
const EOS_IDX: i32 = 3;
const DECODER_SEQ_LEN: usize = 20; // Must match model export

impl Decoder {
    pub fn decode_sequential(self, tokens: &Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let mut target_tokens = tokens.clone();
        target_tokens.resize(DECODER_SEQ_LEN, PAD_IDX);

        Ok(self.decode(1, target_tokens)?)
    }
    pub fn decode_batched(self, batched_tokens: &Vec<Vec<i32>>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let mut batched_target_tokens: Vec<i32> = Vec::new();

        // flatten and resize to correct sequence length
        batched_target_tokens.extend(batched_tokens
            .iter()
            .flat_map(|token| {
                let mut new_token = token.clone();
                new_token.resize(DECODER_SEQ_LEN, PAD_IDX);
                new_token
            })
        );
        Ok(self.decode(batched_tokens.len(), batched_target_tokens)?)
    }
    fn decode(mut self, num_beams: usize, batched_target_tokens: Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let target_tokens_tensor = Tensor::from_array(([num_beams, DECODER_SEQ_LEN], batched_target_tokens))?;

        let mut decoder_inputs = HashMap::new();
        decoder_inputs.insert("memory", self.encode_result.memory_tensor.clone().upcast());
        decoder_inputs.insert("actual_src_length", self.encode_result.actual_length_tensor.clone().upcast());
        decoder_inputs.insert("target_tokens", target_tokens_tensor.upcast());

        let output = self.session.run(decoder_inputs)?;
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;

        // un-flatten data output tensor
        Ok(data
            .chunks_exact(DECODER_SEQ_LEN * 30) // todo: explain why 30 is 30
            .map(|beam_slice| {
                beam_slice
                    .chunks_exact(30)
                    .map(|step_slice| step_slice.to_vec())
                    .collect()
            }).collect()
        )

    }
}