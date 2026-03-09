use std::fs::File;
use std::io;
use std::path::Path;
use fst::{IntoStreamer, Set};
use memmap::Mmap;
use regex_automata::dense;

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
    set: Set<Mmap>,
}

impl WordList {
    fn build_pattern(partial_word: &str) -> String {
        // match case insensitive
        let mut pattern = String::from("(?i)");

        for ch in partial_word.chars() {
            let escaped = regex::escape(&ch.to_string());
            pattern.push_str(&escaped);
            // After each character, optionally match apostrophes
            pattern.push_str("'*");
        }

        pattern
    }
    /// gets the next lowercase letter that follows the partial_word for all possible words it could create
    /// ignores apostrophes
    pub fn get_allowed_next_chars(&self, partial_word: &str) -> Vec<char> {
        let mut search_pattern = Self::build_pattern(partial_word);
        // Match zero or more characters after
        search_pattern.push_str(".*");

        let dfa = dense::Builder::new().anchored(true).build(&search_pattern).unwrap();
        let keys = self.set.search(dfa).into_stream().into_strs().unwrap();

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
    pub fn is_word(&self, string: &str) -> bool {
        match self.get_word(string) {
            None => false,
            Some(_) => true
        }
    }
    pub fn get_word(&self, word: &str) -> Option<String> {
        let search_pattern = Self::build_pattern(word);
        let dfa = dense::Builder::new().anchored(true).build(&search_pattern).unwrap();
        let keys = self.set.search(dfa).into_stream().into_strs().unwrap();

        match keys.is_empty() {
            true => None,
            false => Some(keys.first().unwrap().clone())
        }
    }
    pub fn create_from_file(fst_file: &Path) -> Result<Self, WordListCreationError> {
        let mmap = unsafe { Mmap::map(&File::open(fst_file)?)? };
        let set = Set::new(mmap)?;
        Ok(Self {
            set
        })
    }
}