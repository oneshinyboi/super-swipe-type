use crate::decoder::Decoder;
use crate::dictionary::Dictionary;
use crate::keyboard_manager::KeyTokenizer;
use crate::{SwipeCandidate, EOS_IDX, PAD_IDX, SOS_IDX};
use anyhow::Result;
use std::cmp::Ordering;
use std::mem;
use std::sync::OnceLock;

impl SwipeCandidate {
    fn from_beam(beam: &Beam, unigram_log_prob: f32, bigram_log_prob: f32) -> Self {
        let mut word = String::new();
        for token in &beam.tokens {
            if let Some(char) = KeyTokenizer::index_to_char(*token) {
                word.push(char)
            }
        }
        Self {
            word,
            confidence: (beam.normalized_score() + 0.3 * unigram_log_prob + 0.3 * bigram_log_prob)
                .exp(),
        }
    }
}
#[derive(Debug)]
pub(crate) struct BeamSearchEngine {
    dictionary: Dictionary,
    beam_width: u32,
    branching_factor: u32,
    max_levels: u32,
    temperature: f32,
    active_beams: Vec<Beam>,
    candidates: Vec<SwipeCandidate>,
}

#[derive(Clone, Debug)]
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
            self.score / ((5.0 + len) / 6.0).powf(0.8)
        })
    }
}
#[derive(Debug)]
struct TokenProb {
    log_prob: f32,
    index: u8,
}

impl Eq for TokenProb {}

impl PartialEq<Self> for TokenProb {
    fn eq(&self, other: &Self) -> bool {
        self.log_prob.eq(&other.log_prob)
    }
}

impl PartialOrd<Self> for TokenProb {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.log_prob.partial_cmp(&other.log_prob)
    }
}

impl Ord for TokenProb {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Less)
    }
}

impl Eq for Beam {}

impl PartialEq<Self> for Beam {
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens
    }
}

impl PartialOrd<Self> for Beam {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.normalized_score()
            .partial_cmp(&other.normalized_score())
    }
}

impl Ord for Beam {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Less)
    }
}

impl BeamSearchEngine {
    pub fn new(
        dictionary: Dictionary,
        beam_width: u32,
        branching_factor: u32,
        max_levels: u32,
        temperature: f32,
    ) -> Self {
        Self {
            dictionary,
            beam_width,
            branching_factor,
            max_levels,
            temperature,
            active_beams: Vec::new(),
            candidates: Vec::new(),
        }
    }
    pub fn search(
        &mut self,
        prev_word: &Option<String>,
        decoder: &mut Decoder,
    ) -> Result<Vec<SwipeCandidate>> {
        let mut level = 0;

        // initialize beams
        self.active_beams = Vec::new();
        self.candidates = Vec::new();

        self.active_beams.push(Beam {
            tokens: vec![SOS_IDX],
            score: 0.0,
            finished: false,
            normalized_score_cache: OnceLock::new(),
        });

        while level < self.max_levels && !self.active_beams.is_empty() {
            // sort in descending order to put high scorers first
            self.active_beams.sort();
            self.active_beams.reverse();
            self.expand_beams(prev_word, decoder)?;

            level += 1;
        }

        self.candidates.sort();
        self.candidates.reverse();
        Ok(mem::take(&mut self.candidates))
    }
    /// selects the top branching_factor continuation beams for each beam in beams
    /// and adds them to active_beams or finished_beams
    fn expand_beams(&mut self, prev_word: &Option<String>, decoder: &mut Decoder) -> Result<()> {
        let beam_width = self.beam_width as usize;
        let beams_to_expand: Vec<Beam> = self
            .active_beams
            .drain(..beam_width.min(self.active_beams.len()))
            .collect();

        let tokens: Vec<Vec<i32>> = beams_to_expand
            .iter()
            .map(|b| b.tokens.iter().map(|t| *t as i32).collect())
            .collect();

        let mut result = decoder.decode_batched(&tokens)?;

        for (i, beam_continuation) in result.iter_mut().enumerate() {
            let current_pos = beams_to_expand[i].tokens.len() - 1;
            let logits = &mut beam_continuation[current_pos];

            let token_continuations = self.process_logits(&beams_to_expand[i], logits);

            for continuation in token_continuations {
                let mut new_beam = beams_to_expand[i].clone();
                new_beam.tokens.push(continuation.index);
                new_beam.score += continuation.log_prob;
                new_beam.normalized_score_cache = OnceLock::new(); // Reset cache for modified beam

                if continuation.index == EOS_IDX {
                    new_beam.finished = true;

                    let beam_word = &*KeyTokenizer::indices_to_string(&new_beam.tokens);
                    // if finished it should be a valid word
                    if let Some(word) = self.dictionary.get_word(beam_word) {
                        let log_prob = self.dictionary.get_unigram_log_probability(&word);

                        let mut bigram_prob = 0.0;
                        if let Some(prev_word) = &prev_word {
                            bigram_prob = self.dictionary.get_bigram_log_prob(prev_word, &word);
                        }

                        let mut candidate =
                            SwipeCandidate::from_beam(&new_beam, log_prob, bigram_prob);
                        candidate.word = word;
                        self.candidates.push(candidate);
                    }
                } else if let Some(_) = KeyTokenizer::index_to_char(continuation.index) {
                    new_beam.finished = false;
                    self.active_beams.push(new_beam);
                }
            }
        }
        Ok(())
    }
    /// Return top TokenProbs for given branch and logits
    fn process_logits(&mut self, beam: &Beam, logits: &mut Vec<f32>) -> Vec<TokenProb> {
        self.apply_masking(beam, logits);

        let log_probs: Vec<f32> = self.log_soft_max(logits);

        let mut log_token_probs: Vec<TokenProb> = log_probs
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_nan())
            .map(|(i, p)| TokenProb {
                log_prob: *p,
                index: i as u8,
            })
            .collect();

        log_token_probs.sort();
        log_token_probs.reverse();
        log_token_probs
            .drain(..(self.branching_factor as usize).min(log_token_probs.len()))
            .collect()
    }

    /// convert logits to log probabilities
    fn log_soft_max(&self, logits: &Vec<f32>) -> Vec<f32> {
        let scaled_logits = if self.temperature == 1.0 {
            logits
        } else {
            &logits.iter().map(|l| l / self.temperature).collect()
        };
        let mut max_logit = f32::NEG_INFINITY;

        for logit in scaled_logits {
            if logit > &max_logit {
                max_logit = *logit
            }
        }
        let mut sum_exp = 0.0;
        for logit in scaled_logits {
            sum_exp += (logit - max_logit).exp();
        }
        let log_sum_exp = max_logit + sum_exp.ln();

        let log_probs = scaled_logits.iter().map(|l| l - log_sum_exp).collect();
        log_probs
    }

    fn apply_masking(&mut self, beam: &Beam, logits: &mut Vec<f32>) {
        let partial_word = KeyTokenizer::indices_to_string(&beam.tokens);

        let allowed_next_chars = self.dictionary.get_allowed_next_chars(&partial_word);
        let is_word = self.dictionary.does_word_exist(&partial_word);

        for i in 0..logits.len() as u8 {
            if i == SOS_IDX || i == PAD_IDX {
                logits[i as usize] = f32::NEG_INFINITY;
            } else if i == EOS_IDX && !is_word {
                logits[i as usize] = f32::NEG_INFINITY;
            } else {
                match KeyTokenizer::index_to_char(i) {
                    Some(char) => {
                        if !allowed_next_chars.contains(&char) {
                            logits[i as usize] = f32::NEG_INFINITY;
                        }
                    }
                    None => {}
                }
            }
        }
    }
}
