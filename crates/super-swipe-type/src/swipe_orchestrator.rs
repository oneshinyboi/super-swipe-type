use crate::beam_search::BeamSearchEngine;
use crate::decoder::Decoder;
use crate::dictionary::Dictionary;
use crate::encoder::Encoder;
use crate::swipe_trajectory_processor::SwipeTrajectoryProcessor;
use crate::{SwipeCandidate, SwipePoint, DECODER_SEQ_LEN};
use anyhow::Result;
use cached_path::cached_path;
use std::fs;
use tract_onnx::prelude::*;

const ASSET_COMPAT_VER: &str = "v0.1.2";
const MAX_SEQUENCE_LENGTH: usize = 250; // max length of swipe points that can be processed by the model at once

/// Top-level entry point for the swipe-to-type prediction pipeline.
///
/// `SwipeOrchestrator` owns and coordinates all internal components:
/// the trajectory pre-processor, the ONNX encoder, the ONNX decoder, and
/// the beam-search engine.  Model weights and FST dictionaries are
/// downloaded and cached automatically on first use.
///
/// # Example
/// ```rust,no_run
/// use super_swipe_type::swipe_orchestrator::SwipeOrchestrator;
/// use super_swipe_type::SwipePoint;
/// use std::time::Duration;
///
/// let mut orchestrator = SwipeOrchestrator::new()
///     .expect("Failed to create SwipeOrchestrator");
///
/// let swipe_points = vec![
///     SwipePoint::new(0.2, 0.4, Duration::from_millis(0)),
///     SwipePoint::new(0.7, 0.3, Duration::from_millis(100)),
/// ];
///
/// let candidates = orchestrator.predict(swipe_points, &None).unwrap();
/// let best_word = &candidates[0].word;
/// ```
#[derive(Debug)]
pub struct SwipeOrchestrator {
    swipe_trajectory_processor: SwipeTrajectoryProcessor,
    encoder: Encoder,
    decoder: Decoder,
    beam_search_engine: BeamSearchEngine,
}
impl SwipeOrchestrator {
    /// Creates a new [`SwipeOrchestrator`], downloading and caching all
    /// required model assets if they are not already present.
    ///
    /// On first call this fetches the ONNX encoder, ONNX decoder, and two
    /// FST dictionary files from the GitHub release tagged `v0.1.2` and
    /// stores them in the platform's default cache directory via
    /// `cached_path`.  Subsequent calls reuse the cached files and do not
    /// require a network connection.
    ///
    /// # Errors
    /// Returns an [`anyhow::Error`] if the network request fails, if the
    /// ONNX sessions cannot be built, or if the dictionary files are
    /// malformed.
    pub fn new() -> Result<Self> {
        let base_url = format!(
            "https://github.com/oneshinyboi/super-swipe-type/raw/refs/tags/{}",
            ASSET_COMPAT_VER
        );

        let unigram_path = cached_path(&format!(
            "{}/crates/super-swipe-type/assets/dictionaries/en_wordlist.fst",
            base_url
        ))?;
        let bigram_path = cached_path(&format!(
            "{}/crates/super-swipe-type/assets/dictionaries/en_bigrams.fst",
            base_url
        ))?;
        let encoder_path = cached_path(&format!(
            "{}/crates/super-swipe-type/assets/models/swipe_encoder_android.onnx",
            base_url
        ))?;
        let decoder_path = cached_path(&format!(
            "{}/crates/super-swipe-type/assets/models/swipe_decoder_android.onnx",
            base_url
        ))?;

        let encoder_bytes = fs::read(&encoder_path)?;
        let decoder_bytes = fs::read(decoder_path)?;

        // Encoder: batch=1 always.
        // The ONNX loader pre-populates intermediate node facts (from value_info) with the
        // symbolic 'batch' dim. These conflict with our concrete batch=1 input facts during
        // analysis. Fix: clear all non-input node output facts so analysis starts from
        // only our concrete input facts and propagates forward without contradictions.
        let mut enc_model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&encoder_bytes))?;
        for node in enc_model.nodes_mut() {
            for outlet in &mut node.outputs {
                outlet.fact = InferenceFact::default();
            }
        }
        let encoder_session = enc_model
            .with_input_fact(0, f32::fact([1usize, MAX_SEQUENCE_LENGTH, 6]).into())?
            .with_input_fact(1, i32::fact([1usize, MAX_SEQUENCE_LENGTH]).into())?
            .with_input_fact(2, i32::fact([1usize]).into())?
            .into_optimized()?
            .into_runnable()?;

        // Decoder: dec_seq is always DECODER_SEQ_LEN (20); num_beams is dynamic.
        // Same issue: clear intermediate facts from value_info, then re-set input facts
        // with dec_seq pinned to 20 so the self-attn Reshape([-1,1,1,20]) resolves.
        let mut dec_model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&decoder_bytes))?;
        for node in dec_model.nodes_mut() {
            for outlet in &mut node.outputs {
                outlet.fact = InferenceFact::default();
            }
        }
        // memory is always batch=1 from the encoder; num_beams only applies to target_tokens
        // and actual_src_length (which must be [num_beams] for the cross-attention mask to
        // broadcast to [num_beams, enc_seq] correctly).
        let nb: TDim = dec_model.symbols.sym("n").to_dim();
        let dec_seq: TDim = (DECODER_SEQ_LEN as i64).into();
        let decoder_session = dec_model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    vec![1.to_dim(), 250.to_dim(), 256.to_dim()],
                ),
            )?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(
                    i32::datum_type(),
                    vec![nb.clone(), dec_seq],
                ),
            )?
            .with_input_fact(
                2,
                InferenceFact::dt_shape(i32::datum_type(), vec![nb.clone()]),
            )?
            .into_optimized()?
            .into_runnable()?;

        let encoder = Encoder {
            session: encoder_session,
            max_sequence_length: 250,
        };
        let decoder = Decoder {
            session: decoder_session,
            encode_result: None,
        };

        let dictionary = Dictionary::create_from_file(&unigram_path, &bigram_path)?;

        Ok(Self {
            swipe_trajectory_processor: SwipeTrajectoryProcessor::new(MAX_SEQUENCE_LENGTH),
            encoder,
            decoder,
            beam_search_engine: BeamSearchEngine::new(dictionary, 5, 8, 20, 1.0),
        })
    }
    /// Runs the full swipe-to-type pipeline and returns word predictions.
    ///
    /// The pipeline performs three steps:
    /// 1. **Feature extraction** – converts raw [`SwipePoint`]s into
    ///    position/velocity/acceleration feature vectors sampled at up to
    ///    `MAX_SEQUENCE_LENGTH` (250) points.
    /// 2. **Encoding** – feeds the feature sequence through the ONNX encoder
    ///    transformer to produce a memory tensor.
    /// 3. **Beam-search decoding** – autoregressively decodes the memory
    ///    tensor into word candidates, consulting the FST dictionary to
    ///    prune invalid prefixes and scoring with optional bigram context.
    ///
    /// # Parameters
    /// - `swipe_points` – Ordered sequence of normalised touch points
    ///   captured during the gesture (see [`SwipePoint`] for the coordinate
    ///   system).  Must contain at least two points.
    /// - `prev_word` – The word that was typed immediately before this
    ///   swipe, used to boost candidates via bigram log-probabilities.
    ///   Pass `&None` if there is no preceding word.
    ///
    /// # Returns
    /// A `Vec<SwipeCandidate>` sorted by `confidence` in **descending**
    /// order; the first element is the most likely prediction.
    ///
    /// # Errors
    /// Returns an [`anyhow::Error`] if the encoder or decoder ONNX session
    /// fails, or if beam search encounters an unrecoverable error.
    pub fn predict(
        &mut self,
        swipe_points: Vec<SwipePoint>,
        prev_word: &Option<String>,
    ) -> Result<Vec<SwipeCandidate>> {
        let feature_points = self
            .swipe_trajectory_processor
            .extract_features(swipe_points);
        let encode_result = self.encoder.encode(feature_points)?;
        self.decoder.set_encode_result(encode_result);

        Ok(self
            .beam_search_engine
            .search(prev_word, &mut self.decoder)?)
    }
}
