use std::collections::HashMap;
use std::fs::{create_dir, remove_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;
use fst::{Error, MapBuilder, Set, SetBuilder};

fn main() {
    let out_path = Path::new("./crates/super-swipe-type/assets/dictionaries");
    let input_path = Path::new("./crates/vocab-builder/assets/");

    remove_dir_all(out_path).unwrap();
    create_dir(out_path).unwrap();

    let unigram_path = input_path.join("count_1w.txt");
    let unigram_hash_map = create_unigram_hashmap(&*unigram_path);

    build_dictionary(out_path, input_path, unigram_hash_map).unwrap();
}
fn create_unigram_hashmap(unigram_path: &Path) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    let reader = BufReader::new(File::open(unigram_path).unwrap());

    let mut total_count = 0;
    for line in reader.lines() {
        let line = line.unwrap();
        let mut line_no_whitespace = line.split_whitespace();
        let word = line_no_whitespace.next().unwrap();
        let count = u64::from_str(line_no_whitespace.next().unwrap()).unwrap();
        total_count += count;

        out.insert(String::from(word), count);
    }
    println!("Total unigram count: {total_count}");

    out
}
fn build_dictionary(out_path: &Path, input_path: &Path, unigram_hash_map: HashMap<String, u64>) -> Result<(), Error>{
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