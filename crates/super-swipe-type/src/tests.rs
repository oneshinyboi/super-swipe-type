use crate::{SwipeCandidate, SOS_IDX};
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::keyboard_manager::QwertyKeyboardGrid;
use crate::beam_search::BeamSearchEngine;
use crate::swipe_trajectory_processor::{FeaturePoint, SwipeTrajectoryProcessor};
use crate::{EncodeResult, SwipePoint};
use vector2::Vector2;
use ort::session::Session;
use serde::Deserialize;
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use crate::wordlist::WordList;

#[derive(Debug, Deserialize)]
struct SwipeEntry {
    _id: u32,
    word: String,
    data: Vec<SwipePointData>,
    #[serde(default)]
    potentially_invalid_sentence: bool,
}

#[derive(Debug, Deserialize)]
struct SwipePointData {
    t: u64,
    x: f64,
    y: f64,
}

// Helper function to create encoder
fn create_encoder() -> Encoder {
    let session = Session::builder()
        .unwrap()
        .commit_from_file("assets/models/swipe_encoder_android.onnx")
        .unwrap();
    Encoder {
        session,
        max_sequence_length: 250
    }
}

// Helper function to create decoder
fn create_decoder(encode_result: EncodeResult) -> Decoder {
    let session = Session::builder()
        .unwrap()
        .commit_from_file("assets/models/swipe_decoder_android.onnx")
        .unwrap();
    Decoder {
        session,
        encode_result,
    }
}

// Helper function to create test features
fn create_test_features() -> Vec<FeaturePoint> {
    let keyboard_manager = QwertyKeyboardGrid::new();
    vec![
        FeaturePoint {
            point: Vector2 { x: 0.2, y: 0.4 },
            velocity: Default::default(),
            acceleration: Default::default(),
            nearest_key: keyboard_manager.get_nearest_key(&Vector2 { x: 0.2, y: 0.4 }),
        },
        FeaturePoint {
            point: Vector2 { x: 0.7, y: 0.3 },
            velocity: Default::default(),
            acceleration: Default::default(),
            nearest_key: keyboard_manager.get_nearest_key(&Vector2 { x: 0.7, y: 0.3 }),
        }
    ]
}

// Helper function to create encode result
fn create_encode_result() -> EncodeResult {
    let mut encoder = create_encoder();
    let features = create_test_features();
    encoder.encode(features).unwrap()
}

// Helper function to load all swipe entries from JSONL file
fn load_all_swipe_entries() -> Vec<(String, Vec<SwipePoint>)> {
    let file = File::open("./testing/test.jsonl")
        .expect("Failed to open swipes.jsonl");
    let reader = BufReader::new(file);
    
    reader.lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let entry: SwipeEntry = serde_json::from_str(&line).ok()?;

            // Skip potentially invalid entries if desired
            if entry.potentially_invalid_sentence {
                return None;
            }

            // Find the minimum timestamp to normalize
            let min_timestamp = entry.data.iter()
                .map(|p| p.t)
                .min()
                .unwrap_or(0);

            // Convert to SwipePoint with normalized timestamps
            let points: Vec<SwipePoint> = entry.data.iter()
                .map(|p| SwipePoint {
                    point: Vector2 { x: p.x, y: p.y },
                    timestamp: Duration::from_millis(p.t - min_timestamp),
                })
                .collect();

            Some((entry.word, points))
        })
        .collect()
}

// Helper function to process a single swipe entry
fn process_swipe_entry(
    swipe_points: Vec<SwipePoint>,
    encoder: &mut Encoder,
    wordlist: WordList,
) -> Result<Vec<SwipeCandidate>, String> {
    // Create trajectory processor
    let processor = SwipeTrajectoryProcessor::new(250);

    // Extract features from swipe
    let features = processor.extract_features(swipe_points);

    // Encode features
    let encode_result = encoder.encode(features)
        .map_err(|e| format!("Failed to encode features: {:?}", e))?;

    // Create decoder
    let decoder = create_decoder(encode_result);

    // Run beam search
    let beam_width = 5;
    let branching_factor = 8;
    let max_levels = 20;

    let mut beam_search = BeamSearchEngine::new(
        decoder,
        wordlist,
        beam_width,
        branching_factor,
        max_levels
    );

    beam_search.search()
        .map_err(|e| format!("Beam search failed: {:?}", e))
}

#[test]
fn test_wordlist() {
    let mut wordlist = WordList::create_from_file(Path::new("./assets/dictionaries/en_us_wordlist.fst")).unwrap();

    let start = Instant::now();

    let word = "promiscuous";
    let chars = wordlist.get_allowed_next_chars(word);
    println!("valid next chars for {word}: {:?}", chars);

    let count = wordlist.get_unigram_count(word);
    println!("unigram count for {word}: {count}");

    let probability = wordlist.get_unigram_log_probability(word);

    println!("unigram probability of {word} occurring: {probability}");

    let elapsed = start.elapsed();
    println!("operations took {elapsed:?}")

}

#[test]
fn test_encoder() {
    let mut encoder = create_encoder();
    let features = create_test_features();
    assert!(encoder.encode(features).is_ok());
}

#[test]
fn test_decoder() {
    let encode_result = create_encode_result();
    let mut decoder = create_decoder(encode_result);

    let decode_result = decoder.decode(&vec![SOS_IDX.into()]);
    assert!(decode_result.is_ok());
}

#[test]
fn test_beam_search() {
    // Setup: Create encoder and encode features
    let encode_result = create_encode_result();
    
    // Create decoder
    let decoder = create_decoder(encode_result);
    
    // Create beam search engine with reasonable parameters
    let beam_width = 5;
    let branching_factor = 10;
    let max_levels = 15;
    
    let mut beam_search = BeamSearchEngine::new(
        decoder,
        WordList::create_from_file(Path::new("./crates/super-swipe-type/assets/dictionaries/en_us_wordlist.fst")).unwrap(),
        beam_width,
        branching_factor,
        max_levels
    );
    
    // Perform beam search
    let result = beam_search.search();
    assert!(result.is_ok(), "Beam search should complete successfully");
    
    let candidates = result.unwrap();
    
    // Print results
    println!("\n=== Beam Search Results ===");
    println!("Found {} candidate words:\n", candidates.len());
    
    for (i, candidate) in candidates.iter().enumerate() {
        println!("{}. \"{}\" (confidence: {:.4})", 
            i + 1, 
            candidate.word, 
            candidate.confidence
        );
    }
    println!("===========================\n");
    
    // Assertions
    assert!(!candidates.is_empty(), "Should find at least one candidate word");
    
    // Verify candidates are sorted by confidence (descending)
    for i in 0..candidates.len().saturating_sub(1) {
        assert!(
            candidates[i].confidence >= candidates[i + 1].confidence,
            "Candidates should be sorted by confidence in descending order"
        );
    }
    
    // Verify all candidates have non-empty words
    for candidate in &candidates {
        assert!(!candidate.word.is_empty(), "Candidate words should not be empty");
    }
    
    // Verify confidence values are in valid range [0, 1]
    for candidate in &candidates {
        assert!(
            candidate.confidence >= 0.0 && candidate.confidence <= 1.0,
            "Confidence should be between 0 and 1, got {}",
            candidate.confidence
        );
    }
}



#[test]
fn test_all_swipe_entries() {
    println!("\n=== Testing All Swipe Entries ===");

    // Load all swipe entries
    let all_entries = load_all_swipe_entries();
    println!("Loaded {} total swipe entries\n", all_entries.len());

    // Create shared resources
    let mut encoder = create_encoder();

    let mut successful = 0;
    let mut failed = 0;
    let mut top_match = 0;
    let mut top_5_match = 0;
    let mut total_processing_time = Duration::ZERO;

    let start_all = Instant::now();
    let entries_to_compute: Vec<_> = all_entries.iter().take(1000).collect();

    for (expected_word, swipe_points) in entries_to_compute.iter() {
        let start = Instant::now();
        let wordlist = WordList::create_from_file(Path::new("./assets/dictionaries/en_us_wordlist.fst")).unwrap();

        match process_swipe_entry(swipe_points.clone(), &mut encoder, wordlist) {
            Ok(candidates) => {
                let elapsed = start.elapsed();
                total_processing_time += elapsed;
                successful += 1;
                let filtered_expected_word: String = expected_word.chars().filter(|c| *c != ',' && *c != '.' && *c != '!' && *c != '?').collect();
                let top_word = candidates.first().map(|c| c.word.as_str()).unwrap_or("");

                // Check if expected word matches (case-insensitive)
                if !candidates.is_empty() && candidates[0].word.eq_ignore_ascii_case(&filtered_expected_word) {
                    top_match += 1;
                    println!("✓ \"{}\" (top match, {:?})", top_word, elapsed);
                } else if candidates.iter().take(5).any(|c| c.word.eq_ignore_ascii_case(&filtered_expected_word)) {
                    top_5_match += 1;
                    println!("○ \"{}\" (expected \"{}\" in top 5, {:?})", top_word, filtered_expected_word, elapsed);
                } else {
                    println!("✗ \"{}\" (expected \"{}\", {:?})", top_word, filtered_expected_word, elapsed);
                }
            }
            Err(e) => {
                failed += 1;
                println!("✗ Failed: {}", e);
            }
        }
        stdout().flush().unwrap();
    }

    let total_elapsed = start_all.elapsed();

    println!("\n=== All Entries Summary ===");
    println!("Total entries: {}", entries_to_compute.len());
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!("Success rate: {:.1}%", (successful as f64 / entries_to_compute.len() as f64) * 100.0);

    if successful > 0 {
        println!("\nPerformance:");
        println!("  Total time: {:?}", total_elapsed);
        println!("  Average processing time: {:?}", total_processing_time / successful as u32);
    }

    println!("\nAccuracy:");
    println!("  Top match: {}/{} ({:.1}%)",
        top_match,
        entries_to_compute.len(),
        (top_match as f64 / entries_to_compute.len() as f64) * 100.0
    );
    println!("  Top-5: {}/{} ({:.1}%)",
        top_match + top_5_match,
        entries_to_compute.len(),
        ((top_match + top_5_match) as f64 / entries_to_compute.len() as f64) * 100.0
    );

    println!("===========================\n");

    assert!(successful > 0, "At least one entry should process successfully");
}