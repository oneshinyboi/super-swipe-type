use crate::encoder::EncodeResult;
use crate::{DECODER_SEQ_LEN, PAD_IDX};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tract_onnx::prelude::{IntoTensor, RunnableModel, Tensor, TypedFact, TypedOp};
use tract_onnx::prelude::tract_itertools::Itertools;

#[derive(Debug)]
pub(crate) struct Decoder {
    pub(crate) model: Arc<RunnableModel<TypedFact,Box<dyn TypedOp>>>,
    pub(crate) encode_result: Option<EncodeResult>,
}
impl Decoder {
    pub fn decode(&mut self, tokens: &Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut target_tokens = tokens.clone();
        target_tokens.resize(DECODER_SEQ_LEN.into(), PAD_IDX.into());

        Ok(self.run_inference(1, target_tokens)?)
    }
    pub fn decode_sequentially(
        &mut self,
        batched_tokens: &Vec<Vec<i32>>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut out = Vec::new();
        for tokens in batched_tokens {
            out.append(self.decode(tokens)?.as_mut())
        }
        Ok(out)
    }
    pub fn decode_batched(&mut self, batched_tokens: &Vec<Vec<i32>>) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut batched_target_tokens: Vec<i32> = Vec::new();

        // flatten and resize to correct sequence length
        batched_target_tokens.extend(batched_tokens.iter().flat_map(|token| {
            let mut new_token = token.clone();
            new_token.resize(DECODER_SEQ_LEN.into(), PAD_IDX.into());
            new_token
        }));
        Ok(self.run_inference(batched_tokens.len(), batched_target_tokens)?)
    }
    fn run_inference(
        &mut self,
        num_beams: usize,
        batched_target_tokens: Vec<i32>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let encode_result = self.encode_result.as_ref().ok_or(anyhow!(
            "use set_encode_result to provide the required tensors before running inference"
        ))?;

        let target_tokens_tensor =
            Tensor::from_shape([num_beams, DECODER_SEQ_LEN.into()].as_slice(), batched_target_tokens.as_slice())?;

        let mut decoder_inputs = Vec::new();
        decoder_inputs.push(encode_result.memory_tensor.clone().into());
        decoder_inputs.push(target_tokens_tensor.into());
        decoder_inputs.push(encode_result.actual_length_tensor.clone().into());

        let output = self.model.run(decoder_inputs.into())?;
        let output= output.into_iter().next().unwrap().into_tensor();
        let (data, _) = output.into_plain_array::<f32>()?.into_raw_vec_and_offset();

        // un-flatten data output tensor
        Ok(data
            .chunks_exact(DECODER_SEQ_LEN as usize * 30) // todo: explain why 30 is 30
            .map(|beam_slice| {
                beam_slice
                    .chunks_exact(30)
                    .map(|step_slice| step_slice.to_vec())
                    .collect()
            })
            .collect())
    }
    pub fn set_encode_result(&mut self, encode_result: EncodeResult) {
        self.encode_result = Some(encode_result);
    }
}
