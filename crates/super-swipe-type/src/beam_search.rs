use crate::decoder::Decoder;
use crate::keyboard_manager::KeyTokenizer;
use crate::{SwipeCandidate, EOS_IDX, PAD_IDX, SOS_IDX};
use ort::Error;
use std::cmp::Ordering;
use std::sync::OnceLock;


impl SwipeCandidate {
    fn from_beam(beam: &Beam) -> Self {
        let mut word = String::new();
        for token in &beam.tokens {
            if let Some(char) = KeyTokenizer::index_to_char(*token) {
                word.push(char)
            }
        }
        Self {
            word,
            confidence: (beam.normalized_score() * -1.0).exp()
        }
    }
}
pub(crate) struct BeamSearchEngine {
    decoder: Decoder,
    beam_width: u32,
    branching_factor: u32,
    max_levels: u32,
    temperature: f32,
    active_beams: Vec<Beam>,
    finished_beams: Vec<Beam>
}

#[derive(Clone)]
struct Beam {
    tokens: Vec<u8>,
    score: f32, // accumulated NLLs for each token from the model
    finished: bool,
    normalized_score_cache: OnceLock<f32>,
}
impl Beam {
    fn normalized_score(&self) -> f32 {
        *self.normalized_score_cache.get_or_init(|| {
            let len = self.tokens.len() as f32;
            self.score / ((5.0 + len) / 6.0).powf(1.2)
        })
    }
}
struct TokenProb {
    log_prob: f32,
    index: u8
}

impl Eq for Beam {}

impl PartialEq<Self> for Beam {
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens
    }
}

impl PartialOrd<Self> for Beam {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.normalized_score().partial_cmp(&self.normalized_score())
    }
}

impl Ord for Beam {
    fn cmp(&self, other: &Self) -> Ordering {
        other.partial_cmp(&self)
            .unwrap_or(Ordering::Less)
    }
}


impl BeamSearchEngine {
    pub fn new(decoder: Decoder, beam_width: u32, branching_factor: u32, max_levels: u32)  -> Self {
         Self {
            decoder,
            beam_width,
            branching_factor,
            max_levels,
            temperature: 1.0,
            active_beams: Vec::new(),
            finished_beams: Vec::new(),
        }
    }
    pub fn search(&mut self) -> Result<Vec<SwipeCandidate>, Error> {
        let mut level = 0;

        // initialize beams
        self.active_beams.push(Beam {
            tokens: vec![SOS_IDX],
            score: 0.0,
            finished: false,
            normalized_score_cache: OnceLock::new(),
        });

        while level < self.max_levels && !self.active_beams.is_empty() {
            self.active_beams.sort();
            self.expand_beams()?;


            level += 1;
        }

        let mut candidates: Vec<SwipeCandidate> = self.finished_beams.iter().map(|b|SwipeCandidate::from_beam(b)).collect();
        candidates.sort();
        candidates.reverse();
        Ok(candidates)
    }
    /// selects the top branching_factor continuation beams for each beam in beams
    /// and adds them to active_beams or finished_beams
    fn expand_beams(&mut self) -> Result<(), Error> {
        let beam_width = self.beam_width as usize;
        let beams_to_expand: Vec<Beam> = self.active_beams.drain(..beam_width.min(self.active_beams.len())).collect();

        let tokens: Vec<Vec<i32>> = beams_to_expand.iter()
            .map(|b| b.tokens.iter().map(|t| *t as i32).collect())
            .collect();

        let mut result = self.decoder.decode_batched(&tokens)?;

        for (i, beam_continuation) in result.iter_mut().enumerate() {
            let current_pos = beams_to_expand[i].tokens.len() - 1;
            let logits = &mut beam_continuation[current_pos];

            let token_continuations = self.process_logits(&beams_to_expand[i], logits);

            for continuation in token_continuations {

                let mut new_beam = beams_to_expand[i].clone();
                new_beam.tokens.push(continuation.index);
                new_beam.score -= continuation.log_prob;
                new_beam.normalized_score_cache = OnceLock::new(); // Reset cache for modified beam

                if continuation.index == EOS_IDX {
                    new_beam.finished = true;
                    self.finished_beams.push(new_beam);
                } else if let Some(_) = KeyTokenizer::index_to_char(continuation.index) {
                    new_beam.finished = false;
                    self.active_beams.push(new_beam);
                }
            }

        }
        Ok(())
    }
    /// Return top TokenProbs for given branch and logits
    fn process_logits(&self, beam: &Beam, logits: &mut Vec<f32>) -> Vec<TokenProb> {
        self.apply_trie_masking(beam, logits);
        let log_probs: Vec<f32> = self.log_soft_max(logits);

        let mut log_probs = log_probs
            .iter().enumerate()
            .collect::<Vec<(usize, &f32)>>();

        log_probs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Less));
        log_probs.iter().rev().take(self.branching_factor as usize)
            .map(|(i, p)| TokenProb {log_prob: **p, index: *i as u8 })
            .collect()
    }

    /// convert logits to log probabilities
    fn log_soft_max(&self, logits: &Vec<f32>) -> Vec<f32> {
        let scaled_logits = if self.temperature == 1.0 {logits} else {
            &logits.iter().map(|l| l / self.temperature).collect()
        };
        let mut max_logit = f32::NEG_INFINITY;

        for logit in scaled_logits {
            if logit > &max_logit {max_logit = *logit}
        }
        let mut sum_exp = 0.0;
        for logit in scaled_logits {
            sum_exp += (logit - max_logit).exp();
        }
        let log_sum_exp = max_logit + sum_exp.ln();

        let log_probs = scaled_logits.iter().map(|l| l - log_sum_exp).collect();
        log_probs
    }

    fn apply_trie_masking(&self, beam: &Beam, logits: &mut Vec<f32>) {

        // todo: actually use a vocab trie to mask properly
        for i in 0..logits.len() {
            if i == SOS_IDX as usize || i == PAD_IDX as usize {
                logits[i] = f32::NEG_INFINITY;
            }
        }

    }
}