use crate::beam_search::BeamSearchEngine;
use crate::decoder::Decoder;
use crate::dictionary::Dictionary;
use crate::encoder::Encoder;
use crate::swipe_trajectory_processor::SwipeTrajectoryProcessor;
use crate::{SwipeCandidate, SwipePoint};
use anyhow::{Context, Result};
use cached_path::cached_path;
use rten::Model;
use std::fs;
use std::path::Path;

const ASSET_COMPAT_VER: &str = "v0.1.2";
const MAX_SEQUENCE_LENGTH: usize = 250; // max length of swipe points that can be processed by the model at once

const ENCODER_PATH: &str = "models/swipe_encoder_android.onnx";
const DECODER_PATH: &str = "models/swipe_decoder_android.onnx";
const UNIGRAM_PATH: &str = "dictionaries/en_wordlist.fst";
const BIGRAM_PATH: &str = "dictionaries/en_bigrams.fst";

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
    /// On first call this fetches the `en.tar` archive (containing the ONNX
    /// encoder, ONNX decoder, and both FST dictionary files) from the GitHub
    /// release tagged `v0.1.2` and caches it via `cached_path`.  The relevant
    /// assets are located within the archive and loaded directly from memory
    /// into the model sessions; nothing is unpacked to disk.  Subsequent calls
    /// reuse the cached archive and do not require a network connection.
    ///
    /// # Errors
    /// Returns an [`anyhow::Error`] if the network request fails, if the
    /// archive is malformed, if the ONNX sessions cannot be built, or if the
    /// dictionary files are malformed.
    pub fn new() -> Result<Self> {
        let base_url = format!(
            "https://github.com/oneshinyboi/super-swipe-type/raw/refs/tags/{}",
            ASSET_COMPAT_VER
        );

        let archive_path = cached_path(&format!(
            "{}/crates/super-swipe-type/assets/en.tar",
            base_url
        ))?;

        Self::from_archive(&archive_path)
    }

    /// Creates a [`SwipeOrchestrator`] from a local `en.tar` archive.
    ///
    /// Assets are located within the archive by offset and loaded directly
    /// from memory, so the archive does not need to remain available after
    /// this call returns.
    ///
    /// # Errors
    /// Returns an [`anyhow::Error`] if the archive cannot be opened, if it is
    /// malformed, or if any of the embedded assets are missing or malformed.
    pub fn new_from_path<P: AsRef<Path>>(archive_path: P) -> Result<Self> {
        Self::from_archive(archive_path.as_ref())
    }

    /// Reads the archive and builds an orchestrator by locating each asset's
    /// bytes and loading them.
    fn from_archive(archive_path: &Path) -> Result<Self> {
        let tar_bytes = fs::read(archive_path)
            .with_context(|| format!("failed to read archive {}", archive_path.display()))?;

        let mut encoder_bytes = None;
        let mut decoder_bytes = None;
        let mut unigram_bytes = None;
        let mut bigram_bytes = None;

        let mut archive = tar::Archive::new(&tar_bytes[..]);
        for entry in archive.entries()? {
            let entry = entry?;
            let path = entry.path()?;

            let Some(name) = path.to_str() else {
                continue;
            };
            let start = entry.raw_file_position() as usize;
            let end = start
                .checked_add(entry.size() as usize)
                .context("tar entry overflows archive bounds")?;
            let bytes = tar_bytes
                .get(start..end)
                .context("tar entry is truncated")?
                .to_vec();

            let target = match name {
                x if x == ENCODER_PATH => &mut encoder_bytes,
                x if x == DECODER_PATH => &mut decoder_bytes,
                x if x == UNIGRAM_PATH => &mut unigram_bytes,
                x if x == BIGRAM_PATH => &mut bigram_bytes,
                _ => continue,
            };
            *target = Some(bytes);
        }

        Self::from_bytes(
            required(encoder_bytes, ENCODER_PATH)?,
            required(decoder_bytes, DECODER_PATH)?,
            required(unigram_bytes, UNIGRAM_PATH)?,
            required(bigram_bytes, BIGRAM_PATH)?,
        )
    }

    /// Creates a [`SwipeOrchestrator`] from explicit paths to the ONNX
    /// encoder, ONNX decoder, and both FST dictionaries.
    ///
    /// # Errors
    /// Returns an [`anyhow::Error`] if any of the files cannot be read or
    /// is malformed.
    pub fn new_from_paths(
        encoder_path: &Path,
        decoder_path: &Path,
        unigram_path: &Path,
        bigram_path: &Path,
    ) -> Result<Self> {
        let encoder_bytes = std::fs::read(encoder_path)
            .with_context(|| format!("failed to read encoder {}", encoder_path.display()))?;
        let decoder_bytes = std::fs::read(decoder_path)
            .with_context(|| format!("failed to read decoder {}", decoder_path.display()))?;
        let unigram_bytes = std::fs::read(unigram_path)
            .with_context(|| format!("failed to read unigram dictionary {}", unigram_path.display()))?;
        let bigram_bytes = std::fs::read(bigram_path)
            .with_context(|| format!("failed to read bigram dictionary {}", bigram_path.display()))?;

        Self::from_bytes(encoder_bytes, decoder_bytes, unigram_bytes, bigram_bytes)
    }

    fn from_bytes(
        encoder_bytes: Vec<u8>,
        decoder_bytes: Vec<u8>,
        unigram_bytes: Vec<u8>,
        bigram_bytes: Vec<u8>,
    ) -> Result<Self> {
        let encoder_model = Model::load(encoder_bytes)
            .context("failed to load encoder model")?;
        let decoder_model = Model::load(decoder_bytes)
            .context("failed to load decoder model")?;

        let encoder = Encoder::new(encoder_model, MAX_SEQUENCE_LENGTH)?;
        let decoder = Decoder::new(decoder_model)?;

        let dictionary = Dictionary::create_from_bytes(unigram_bytes, bigram_bytes)
            .map_err(|e| anyhow::anyhow!("failed to create dictionary: {}", e))?;

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

    pub(crate) fn encoder_mut(&mut self) -> &mut Encoder {
        &mut self.encoder
    }
    pub(crate) fn decoder_mut(&mut self) -> &mut Decoder {
        &mut self.decoder
    }
}

/// Returns the asset bytes, or a descriptive error if it was missing from the
/// archive.
fn required(bytes: Option<Vec<u8>>, name: &str) -> Result<Vec<u8>> {
    bytes.with_context(|| format!("asset '{}' not found in archive", name))
}
