use std::fs::{create_dir, remove_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;
use fst::{Error, SetBuilder};

fn main() {
    let out_path = Path::new("./crates/super-swipe-type/assets/dictionaries");
    remove_dir_all(out_path).unwrap();
    create_dir(out_path).unwrap();
    build_dictionary(out_path).unwrap();
}
fn build_dictionary(out_path: &Path) -> Result<(), Error>{
    let mut builder = SetBuilder::new(
        BufWriter::new(File::create(out_path.join("en_us_wordlist.fst"))?)
    )?;
    let wordlist_file = File::open("./crates/vocab-builder/assets/en_us_wordlist")?;
    let wordlist = BufReader::new(wordlist_file);

    let mut lines = wordlist.lines();
    while lines.next().unwrap()? != "---" {}

    for line in lines {
        builder.insert(line.unwrap().trim())?;
    }
    builder.finish()
}