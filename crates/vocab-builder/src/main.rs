use fst::{Error, MapBuilder};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir, remove_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

fn main() {
    let out_path = Path::new("./crates/super-swipe-type/assets/dictionaries");
    let input_path = Path::new("./crates/vocab-builder/assets/");

    remove_dir_all(out_path).unwrap();
    create_dir(out_path).unwrap();

    let unigram_path = input_path.join("count_1w.txt");
    let bigram_path = input_path.join("count_2w.txt");

    let unigram_hash_map = create_unigram_hashmap(&*unigram_path);
    let bigram_hash_map = create_bigram_hashmap(&bigram_path, &unigram_hash_map);

    println!("building unigram fst map");
    build_unigram_fst_map(out_path, input_path, unigram_hash_map).unwrap();
    println!("building bigram fst map");
    build_bigram_fst_map(out_path, input_path, bigram_hash_map).unwrap();
}
fn create_bigram_hashmap(bigram_path: &Path, unigram_hash_map: &HashMap<String, u64>) -> HashMap<String, HashMap<String, f32>> {
    let mut out = HashMap::new();
    let reader = BufReader::new(File::open(bigram_path).unwrap());

    for line in reader.lines() {
        let line = line.unwrap();
        let mut line_split_whitespace = line.split_whitespace();
        let word1 = line_split_whitespace.next().unwrap().to_ascii_lowercase();
        let word2 = line_split_whitespace.next().unwrap().to_ascii_lowercase();
        let count = u64::from_str(line_split_whitespace.next().unwrap()).unwrap();

        if let Some(word1_total_count) = unigram_hash_map.get(&word1) {
            let log_prob = (count as f64 / *word1_total_count as f64).ln() as f32;

            if log_prob > 0.0 {
                panic!("count for bigram {word1}:{word2} should always be less than total count for {word1} ")
            }
            let map = out.entry(String::from(word1)).or_insert_with(HashMap::new);

            map.insert(String::from(word2), log_prob);
        }
    }
    out
}
fn create_unigram_hashmap(unigram_path: &Path) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    let reader = BufReader::new(File::open(unigram_path).unwrap());

    let mut total_count = 0;
    for line in reader.lines() {
        let line = line.unwrap();
        let mut line_split_whitespace = line.split_whitespace();
        let word = line_split_whitespace.next().unwrap();
        let count = u64::from_str(line_split_whitespace.next().unwrap()).unwrap();
        total_count += count;

        out.insert(String::from(word), count);
    }
    println!("Total unigram count: {total_count}");

    out
}
fn build_unigram_fst_map(out_path: &Path, input_path: &Path, unigram_hash_map: HashMap<String, u64>) -> Result<(), Error>{
    let mut builder = MapBuilder::new(
        BufWriter::new(File::create(out_path.join("en_us_wordlist.fst"))?)
    )?;
    let wordlist_file = File::open(input_path.join("en_us_wordlist"))?;
    let wordlist = BufReader::new(wordlist_file);

    let mut lines = wordlist.lines();
    while lines.next().unwrap()? != "---" {}

    for line in lines {
        let word = line.unwrap();

        let lowercase_word: String = word.trim().to_ascii_lowercase().chars().filter(|c| *c != '\'').collect();

        builder.insert(word.trim(), *unigram_hash_map.get(&lowercase_word).unwrap_or(&0))?;
        //builder.insert(word.trim())?;
    }
    builder.finish()
}
fn build_bigram_fst_map(out_path: &Path, input_path: &Path, bigram_hash_map: HashMap<String, HashMap<String, f32>>)-> Result<(), Error> {
    // get wordlist words
    let mut builder = MapBuilder::new(
        BufWriter::new(File::create(out_path.join("en_us_bigrams.fst"))?)
    )?;
    let wordlist_file = File::open(input_path.join("en_us_wordlist"))?;
    let wordlist = BufReader::new(wordlist_file);

    let mut lines = wordlist.lines();
    while lines.next().unwrap()? != "---" {}
    let words: Vec<String> = lines.map(|l| l.unwrap().trim().into()).collect();
    let words_set: HashSet<_> = words.into_iter().collect();
    println!("got words");

    // check if there is a bigram for all word combinations in the wordlist
    let mut bigram_list = Vec::new();
    let mut smallest_log_prob = f32::INFINITY;
    for (word1, map) in bigram_hash_map {
        let lowercase_word1: String = word1.to_ascii_lowercase().chars().filter(|c| *c != '\'').collect();
        for (word2, log_prob) in map {
            let lowercase_word2: String = word2.to_ascii_lowercase().chars().filter(|c| *c != '\'').collect();

            if words_set.contains(&lowercase_word1) && words_set.contains(&lowercase_word2) {
                let val = (log_prob * -100000.0) as u64;
                if log_prob < smallest_log_prob {
                    smallest_log_prob = log_prob;
                }
                let key = format!("{word1}-{word2}");

                bigram_list.push((key, val));
            }
        }

    }

    println!("done constructing bigram_list");
    println!("{smallest_log_prob}");
    //sort lexicographically
    bigram_list.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, value) in bigram_list {
        builder.insert(key, value)?;
    }
    builder.finish()?;

    Ok(())
}