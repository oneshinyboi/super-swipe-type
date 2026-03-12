use std::fs;
use std::fs::File;
use ort::session::Session;
use crate::beam_search::BeamSearchEngine;
use crate::decoder::Decoder;
use crate::dictionary::Dictionary;
use crate::encoder::Encoder;
use crate::{SwipeCandidate, SwipePoint};
use crate::swipe_trajectory_processor::SwipeTrajectoryProcessor;
use anyhow::Result;
use cached_path::cached_path;

const ASSET_COMPAT_VER: &str = "v0.1.2";
const MAX_SEQUENCE_LENGTH: usize = 250; // max length of swipe points that can be processed by the model at once
#[derive(Debug)]
pub struct SwipeOrchestrator {
    swipe_trajectory_processor: SwipeTrajectoryProcessor,
    encoder: Encoder,
    decoder: Decoder,
    beam_search_engine: BeamSearchEngine

}
impl SwipeOrchestrator {
    pub fn new() -> Result<Self> {
        let base_url = format!("https://github.com/oneshinyboi/super-swipe-type/raw/refs/tags/{}", ASSET_COMPAT_VER);

        let unigram_path = cached_path(&format!("{}/crates/super-swipe-type/assets/dictionaries/en_wordlist.fst", base_url))?;
        let bigram_path = cached_path(&format!("{}/crates/super-swipe-type/assets/dictionaries/en_bigrams.fst", base_url))?;
        let encoder_path = cached_path(&format!("{}/crates/super-swipe-type/assets/models/swipe_encoder_android.onnx", base_url))?;
        let decoder_path = cached_path(&format!("{}/crates/super-swipe-type/assets/models/swipe_decoder_android.onnx", base_url))?;

        let encoder_bytes = fs::read(encoder_path)?;
        let decoder_bytes = fs::read(decoder_path)?;

        let encoder_session = Session::builder()?.commit_from_memory(&*encoder_bytes)?;
        let decoder_session = Session::builder()?.commit_from_memory(&*decoder_bytes)?;

        let encoder = Encoder {
            session: encoder_session,
            max_sequence_length: 250,
        };
        let decoder = Decoder {
            session: decoder_session,
            encode_result: None
        };

        let dictionary = Dictionary::create_from_file(&unigram_path,  &bigram_path)?;

        Ok(Self {
            swipe_trajectory_processor: SwipeTrajectoryProcessor::new(MAX_SEQUENCE_LENGTH),
            encoder,
            decoder,
            beam_search_engine: BeamSearchEngine::new(
                dictionary,
                5,
                8,
                20,
                1.0
            )
        })
    }
    pub fn predict(&mut self, swipe_points: Vec<SwipePoint>, prev_word: &Option<String>) -> Result<Vec<SwipeCandidate>> {
        let feature_points = self.swipe_trajectory_processor.extract_features(swipe_points);
        let encode_result = self.encoder.encode(feature_points)?;
        self.decoder.set_encode_result(encode_result);

        Ok(self.beam_search_engine.search(prev_word, &mut self.decoder)?)
    }

}