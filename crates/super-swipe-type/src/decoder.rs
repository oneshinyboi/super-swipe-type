use crate::encoder::{EncodeResult, OnnxSession};
use crate::{DECODER_SEQ_LEN, PAD_IDX};
use anyhow::{anyhow, Result};
use tract_onnx::prelude::*;

#[derive(Debug)]
pub(crate) struct Decoder {
    pub(crate) session: OnnxSession,
    pub(crate) encode_result: Option<EncodeResult>,
}

impl Decoder {
    pub fn decode(&mut self, tokens: &Vec<i32>) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut target_tokens = tokens.clone();
        target_tokens.resize(DECODER_SEQ_LEN.into(), PAD_IDX.into());
        self.run_inference(1, target_tokens)
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
        self.run_inference(batched_tokens.len(), batched_target_tokens)
    }

    fn run_inference(
        &mut self,
        num_beams: usize,
        batched_target_tokens: Vec<i32>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let encode_result = self.encode_result.as_ref().ok_or(anyhow!(
            "use set_encode_result to provide the required tensors before running inference"
        ))?;

        // Reconstruct the memory tensor from the stored data/shape
        let memory_tensor: Tensor = tract_ndarray::ArrayD::from_shape_vec(
            encode_result.memory_shape.as_slice(),
            encode_result.memory_data.clone(),
        )?
        .into();

        // actual_src_length: shape [num_beams] — repeat the length once per beam so the
        // cross-attention mask broadcasts to [num_beams, enc_seq].
        let length_tensor: Tensor =
            tract_ndarray::Array1::from_elem(num_beams, encode_result.actual_length).into();

        // target_tokens: shape [num_beams, DECODER_SEQ_LEN]
        let seq_len: usize = DECODER_SEQ_LEN.into();
        let target_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (num_beams, seq_len),
            batched_target_tokens,
        )?
        .into();

        // Model inputs are ordered: memory, target_tokens, actual_src_length
        let output = self.session.run(tvec![
            memory_tensor.into(),
            target_tensor.into(),
            length_tensor.into(),
        ])?;

        let data_view = output[0].to_array_view::<f32>()?;
        let data: Vec<f32> = data_view.iter().copied().collect();

        // un-flatten: [num_beams, seq_len=20, vocab=30]
        Ok(data
            .chunks_exact(DECODER_SEQ_LEN as usize * 30) // 30 = vocab size
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
