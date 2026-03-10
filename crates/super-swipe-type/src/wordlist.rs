use std::fs::File;
use std::io;
use std::path::Path;
use fst::{IntoStreamer, Map};
use memmap::Mmap;
use regex_automata::{dense, DenseDFA};

const TOTAL_UNIGRAM_COUNT: u64 = 588124220187;
#[derive(Debug)]
pub enum WordListCreationError {
    Io(io::Error),
    Fst(fst::Error),
}

impl std::fmt::Display for WordListCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WordListCreationError::Io(e) => write!(f, "IO error: {}", e),
            WordListCreationError::Fst(e) => write!(f, "FST error: {}", e),
        }
    }
}

impl std::error::Error for WordListCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WordListCreationError::Io(e) => Some(e),
            WordListCreationError::Fst(e) => Some(e),
        }
    }
}

impl From<io::Error> for WordListCreationError {
    fn from(err: io::Error) -> Self {
        WordListCreationError::Io(err)
    }
}

impl From<fst::Error> for WordListCreationError {
    fn from(err: fst::Error) -> Self {
        WordListCreationError::Fst(err)
    }
}

pub(crate) struct WordList {
    set: Map<Mmap>,
    pattern_manager: PatternManager
}
#[derive(Default)]
struct PatternManager {
    word: String,
    search_pattern: String,
    dfa: Option<DenseDFA<Vec<usize>, usize>>
}
impl PatternManager {
    pub fn optionally_create_pattern_manager(&mut self, word: &str) {
        if word != self.word || word == "" {
            let search_pattern = Self::build_pattern(word);
            let dfa = dense::Builder::new().anchored(true).build(&search_pattern).unwrap();

            self.word = String::from(word);
            self.search_pattern = search_pattern;
            self.dfa = Some(dfa)
        }

    }
    fn build_pattern(word: &str) -> String {
        // match case insensitive
        let mut pattern = String::from("(?i)");

        for ch in word.chars() {
            let escaped = regex::escape(&ch.to_string());
            pattern.push_str(&escaped);
            // After each character, optionally match apostrophes
            pattern.push_str("'*");
        }

        pattern
    }
}

impl WordList {

    /// gets the next lowercase letter that follows the partial_word for all possible words it could create
    /// ignores apostrophes
    pub fn get_allowed_next_chars(&mut self, partial_word: &str) -> Vec<char> {
        self.pattern_manager.optionally_create_pattern_manager(partial_word);
        // Match zero or more characters after
        let mut search_pattern = self.pattern_manager.search_pattern.clone();
        search_pattern.push_str(".*");

        let dfa = dense::Builder::new().anchored(true).build(&search_pattern).unwrap();
        let keys = self.set.search(dfa).into_stream().into_str_keys().unwrap();

        //println!("{:?}", keys);
        let mut chars: Vec<char> = keys.iter()
            .map(|w|
                w.chars()
                    .filter(|c| *c != '\'')
                    .map(|c| c.to_ascii_lowercase())
                    .nth(partial_word.len())
                    .unwrap_or_default())
            .filter(|c| *c != char::default())
            .collect();
        chars.sort();
        chars.dedup();
        chars
    }
    pub fn get_unigram_count(&mut self, word: &str) -> u64 {
        self.pattern_manager.optionally_create_pattern_manager(word);
        let dfa = self.pattern_manager.dfa.as_ref().unwrap();

        let vals = self.set.search(dfa).into_stream().into_values();

        match vals.first() {
            Some(n) => *n,
            None => 0
        }
    }
    pub fn get_unigram_log_probability(&mut self, word: &str) -> f64 {
        let count = self.get_unigram_count(word);
        (count as f64 / TOTAL_UNIGRAM_COUNT as f64).ln()
    }

    pub fn does_word_exist(&mut self, string: &str) -> bool {
        match self.get_word(string) {
            None => false,
            Some(_) => true
        }
    }
    pub fn get_word(&mut self, word: &str) -> Option<String> {
        self.pattern_manager.optionally_create_pattern_manager(word);
        let dfa = self.pattern_manager.dfa.as_ref().unwrap();

        let keys = self.set.search(dfa).into_stream().into_str_keys().unwrap();

        match keys.is_empty() {
            true => None,
            false => Some(keys.first().unwrap().clone())
        }
    }
    pub fn create_from_file(fst_file: &Path) -> Result<Self, WordListCreationError> {
        let mmap = unsafe { Mmap::map(&File::open(fst_file)?)? };
        let set = Map::new(mmap)?;
        Ok(Self {
            set,
            pattern_manager: PatternManager::default()
        })
    }
}