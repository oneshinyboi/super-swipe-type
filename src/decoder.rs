use crate::encoder::EncodeResult;
use crate::{DECODER_SEQ_LEN, PAD_IDX};
use ort::session::Session;
use ort::value::Tensor;
use ort::Error;
use std::collections::HashMap;


pub(crate) struct Decoder {
    pub(crate) session: Session,
    pub(crate) encode_result: EncodeResult,
}
impl Decoder {
    pub fn decode(&mut self, tokens: &Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let mut target_tokens = tokens.clone();
        target_tokens.resize(DECODER_SEQ_LEN.into(), PAD_IDX.into());

        Ok(self.run_inference(1, target_tokens)?)
    }
    pub fn decode_batched(&mut self, batched_tokens: &Vec<Vec<i32>>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let mut batched_target_tokens: Vec<i32> = Vec::new();

        // flatten and resize to correct sequence length
        batched_target_tokens.extend(batched_tokens
            .iter()
            .flat_map(|token| {
                let mut new_token = token.clone();
                new_token.resize(DECODER_SEQ_LEN.into(), PAD_IDX.into());
                new_token
            })
        );
        Ok(self.run_inference(batched_tokens.len(), batched_target_tokens)?)
    }
    fn run_inference(&mut self, num_beams: usize, batched_target_tokens: Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>, Error> {
        let target_tokens_tensor = Tensor::from_array(([num_beams, DECODER_SEQ_LEN.into()], batched_target_tokens))?;

        let mut decoder_inputs = HashMap::new();
        decoder_inputs.insert("memory", self.encode_result.memory_tensor.clone().upcast());
        decoder_inputs.insert("actual_src_length", self.encode_result.actual_length_tensor.clone().upcast());
        decoder_inputs.insert("target_tokens", target_tokens_tensor.upcast());

        let output = self.session.run(decoder_inputs)?;
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;

        // un-flatten data output tensor
        Ok(data
            .chunks_exact(DECODER_SEQ_LEN as usize * 30) // todo: explain why 30 is 30
            .map(|beam_slice| {
                beam_slice
                    .chunks_exact(30)
                    .map(|step_slice| step_slice.to_vec())
                    .collect()
            }).collect()
        )

    }
}