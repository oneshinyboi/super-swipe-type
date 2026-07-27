use crate::encoder::EncodeResult;
use crate::{DECODER_SEQ_LEN, PAD_IDX};
use anyhow::{anyhow, Result};
use rten::{Model, NodeId, Value};

#[derive(Debug)]
pub(crate) struct Decoder {
    pub(crate) model: Model,
    pub(crate) encode_result: Option<EncodeResult>,
    input_memory: NodeId,
    input_target_tokens: NodeId,
    input_actual_src_length: NodeId,
    output_node: NodeId,
}
impl Decoder {
    pub fn new(model: Model) -> Result<Self> {
        Ok(Self {
            input_memory: model.node_id("memory")?,
            input_target_tokens: model.node_id("target_tokens")?,
            input_actual_src_length: model.node_id("actual_src_length")?,
            output_node: model.node_id("log_probs")?,
            model,
            encode_result: None,
        })
    }

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

        let target_tokens_value =
            Value::from_shape([num_beams, DECODER_SEQ_LEN.into()], batched_target_tokens)?;

        let inputs = vec![
            (self.input_memory, encode_result.memory_tensor.clone().into()),
            (self.input_target_tokens, target_tokens_value.into()),
            (self.input_actual_src_length, encode_result.actual_length_tensor.clone().into()),
        ];

        let outputs = [self.output_node];
        let [output] = self
            .model
            .run_n(inputs, outputs, None)
            .map_err(|e| anyhow!("Decoder inference failed: {}", e))?;

        let (_, data) = output
            .into_shape_vec::<f32, 3>()
            .map_err(|e| anyhow!("Failed to extract decoder output: {}", e))?;

        Ok(data
            .chunks_exact(DECODER_SEQ_LEN as usize * 30)
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