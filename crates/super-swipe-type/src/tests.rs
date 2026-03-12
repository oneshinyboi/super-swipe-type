use crate::swipe_orchestrator::SwipeOrchestrator;
use crate::{SwipePoint};
use vector2::Vector2;
use serde::Deserialize;
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use crate::dictionary::Dictionary;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct SwipeEntry {
    id: u32,
    word: String,
    data: Vec<SwipePointData>,
    sentence: String,
    word_idx: u32,
    #[serde(default)]
    potentially_invalid_sentence: bool,
}

#[derive(Debug, Deserialize)]
struct SwipePointData {
    t: u64,
    x: f64,
    y: f64,
}

// Statistics structure for aggregating results
#[derive(Debug, Default)]
struct TestStatistics {
    successful: usize,
    failed: usize,
    top_match: usize,
    top_5_match: usize,
    total_processing_time: Duration,
}

impl TestStatistics {
    fn merge(&mut self, other: TestStatistics) {
        self.successful += other.successful;
        self.failed += other.failed;
        self.top_match += other.top_match;
        self.top_5_match += other.top_5_match;
        self.total_processing_time += other.total_processing_time;
    }
}

// Helper function to load all swipe entries from JSONL file
fn load_swipe_entries(count: usize, mut word_list: Dictionary) -> Vec<(String, Option<String>, Vec<SwipePoint>)> {
    let file = File::open("./testing/test.jsonl")
        .expect("Failed to open swipes.jsonl");
    let reader = BufReader::new(file);
    
    reader.lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let entry: SwipeEntry = serde_json::from_str(&line).ok()?;

            // Skip potentially invalid entries if desired
            if entry.potentially_invalid_sentence || !word_list.does_word_exist(&entry.word){
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

            let mut prev_word = None;
            if entry.word_idx != 0 {
                if let Some(word) = entry.sentence.split_whitespace().nth(entry.word_idx as usize - 1) {
                    prev_word = Some(word.into());
                }
            }

            Some((entry.word, prev_word, points))
        })
        .take(count)
        .collect()
}

fn get_dictionary() -> Dictionary {
    let unigrams_path = Path::new("./assets/dictionaries/en_us_wordlist.fst");
    let bigrams_path = Path::new("./assets/dictionaries/en_us_bigrams.fst");

    Dictionary::create_from_file(unigrams_path, bigrams_path).unwrap()
}

// Process a chunk of entries (runs in parallel)
fn process_chunk(
    chunk: &[(String, Option<String>, Vec<SwipePoint>)],
    chunk_id: usize,
    progress_counter: Arc<AtomicUsize>,
    total_entries: usize,
) -> TestStatistics {
    let mut stats = TestStatistics::default();
    let mut orchestrator = SwipeOrchestrator::new().expect("Failed to create SwipeOrchestrator");

    for (expected_word, prev_word, swipe_points) in chunk {
        let current = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let percentage = (current as f64 / total_entries as f64) * 100.0;

        print!("\r[{}/{}] ({:.1}%) Processing...", current, total_entries, percentage);
        stdout().flush().unwrap();

        let start = Instant::now();

        match orchestrator.predict(swipe_points.clone(), &prev_word) {
            Ok(candidates) => {
                let elapsed = start.elapsed();
                stats.total_processing_time += elapsed;
                stats.successful += 1;

                let filtered_expected_word: String = expected_word
                    .chars()
                    .filter(|c| *c != ',' && *c != '.' && *c != '!' && *c != '?')
                    .collect();

                // Check if expected word matches (case-insensitive)
                if !candidates.is_empty() && candidates[0].word.eq_ignore_ascii_case(&filtered_expected_word) {
                    stats.top_match += 1;
                } else if candidates.iter().take(5).any(|c| c.word.eq_ignore_ascii_case(&filtered_expected_word)) {
                    stats.top_5_match += 1;
                }
            }
            Err(_) => {
                stats.failed += 1;
            }
        }
    }

    stats
}

#[test]
fn test_dictionary() {
    let mut dictionary = get_dictionary();

    let start = Instant::now();

    let word = "promiscuous";
    let chars = dictionary.get_allowed_next_chars(word);
    println!("valid next chars for {word}: {:?}", chars);

    let count = dictionary.get_unigram_count(word);
    println!("unigram count for {word}: {count}");

    let probability = dictionary.get_unigram_log_probability(word);

    println!("unigram probability of {word} occurring: {probability}");

    let elapsed = start.elapsed();
    println!("operations took {elapsed:?}");

    let bigram_prob = dictionary.get_bigram_log_prob("maybe", "that");
    println!("{bigram_prob}");
}

#[test]
fn test_swipe_orchestrator() {
    // Create orchestrator
    let mut orchestrator = SwipeOrchestrator::new()
        .expect("Failed to create SwipeOrchestrator");

    // Create simple test swipe points
    let swipe_points = vec![
        SwipePoint {
            point: Vector2 { x: 0.2, y: 0.4 },
            timestamp: Duration::from_millis(0),
        },
        SwipePoint {
            point: Vector2 { x: 0.7, y: 0.3 },
            timestamp: Duration::from_millis(100),
        }
    ];
    
    // Perform prediction
    let result = orchestrator.predict(swipe_points, &None);
    assert!(result.is_ok(), "SwipeOrchestrator prediction should complete successfully");
    
    let candidates = result.unwrap();
    
    // Print results
    println!("\n=== SwipeOrchestrator Results ===");
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
    println!("\n=== Testing All Swipe Entries (Parallel) ===");

    // Load all swipe entries
    let wordlist = get_dictionary();
    let entries_to_load = 500;
    let all_entries = load_swipe_entries(entries_to_load, wordlist);
    let total_entries = all_entries.len();

    println!("Loaded {} total swipe entries", total_entries);

    // Determine number of threads (use all available cores)
    let num_threads = rayon::current_num_threads();
    println!("Using {} threads for parallel processing\n", num_threads);

    let start_all = Instant::now();

    // Create atomic counter for progress tracking
    let progress_counter = Arc::new(AtomicUsize::new(0));

    // Split entries into chunks and process in parallel
    let chunk_size = (total_entries + num_threads - 1) / num_threads;
    let final_stats: TestStatistics = all_entries
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_id, chunk)| {
            process_chunk(chunk, chunk_id, Arc::clone(&progress_counter), total_entries)
        })
        .reduce(
            || TestStatistics::default(),
            |mut acc, stats| {
                acc.merge(stats);
                acc
            }
        );

    let total_elapsed = start_all.elapsed();

    println!("\n\n=== All Entries Summary ===");
    println!("Total entries: {}", total_entries);
    println!("Successful: {}", final_stats.successful);
    println!("Failed: {}", final_stats.failed);
    println!("Success rate: {:.1}%",
        (final_stats.successful as f64 / total_entries as f64) * 100.0);

    if final_stats.successful > 0 {
        println!("\nPerformance:");
        println!("  Total time: {:?}", total_elapsed);
        println!("  Average processing time: {:?}",
            final_stats.total_processing_time / final_stats.successful as u32);
        println!("  Throughput: {:.1} entries/sec",
            total_entries as f64 / total_elapsed.as_secs_f64());
    }

    println!("\nAccuracy:");
    println!("  Top match: {}/{} ({:.1}%)",
        final_stats.top_match,
        total_entries,
        (final_stats.top_match as f64 / total_entries as f64) * 100.0
    );
    println!("  Top-5: {}/{} ({:.1}%)",
        final_stats.top_match + final_stats.top_5_match,
        total_entries,
        ((final_stats.top_match + final_stats.top_5_match) as f64 / total_entries as f64) * 100.0
    );

    println!("===========================\n");

    assert!(final_stats.successful > 0, "At least one entry should process successfully");
}

#[test]
fn test_all_swipe_entries_single_threaded() {
    println!("\n=== Testing All Swipe Entries ===");

    // Load all swipe entries
    let wordlist = get_dictionary();
    let entries_to_load = 500;
    let all_entries = load_swipe_entries(entries_to_load, wordlist);
    println!("Loaded {} total swipe entries\n", all_entries.len());

    // Create orchestrator
    let mut orchestrator = SwipeOrchestrator::new()
        .expect("Failed to create SwipeOrchestrator");

    let mut successful = 0;
    let mut failed = 0;
    let mut top_match = 0;
    let mut top_5_match = 0;
    let mut total_processing_time = Duration::ZERO;

    let start_all = Instant::now();
    let entries_to_compute: Vec<_> = all_entries.iter().collect();
    let total_entries = entries_to_compute.len();

    for (index, (expected_word, prev_word, swipe_points)) in entries_to_compute.iter().enumerate() {
        let current_entry = index + 1;
        let percentage = (current_entry as f64 / total_entries as f64) * 100.0;

        println!("\n[{}/{}] ({:.1}%)", current_entry, total_entries, percentage);

        let start = Instant::now();

        match orchestrator.predict(swipe_points.clone(), &prev_word) {
            Ok(candidates) => {
                let elapsed = start.elapsed();
                total_processing_time += elapsed;
                successful += 1;
                let filtered_expected_word: String = expected_word.chars().filter(|c| *c != ',' && *c != '.' && *c != '!' && *c != '?').collect();

                // Print expected word
                println!("\nExpected: \"{}\"", filtered_expected_word);

                // Print top 5 predictions
                println!("Top 5 predictions:");
                for (i, candidate) in candidates.iter().take(5).enumerate() {
                    let marker = if candidate.word.eq_ignore_ascii_case(&filtered_expected_word) {
                        "✓"
                    } else {
                        " "
                    };
                    println!("  {}{}. \"{}\" (confidence: {:.4})",
                             marker,
                             i + 1,
                             candidate.word,
                             candidate.confidence
                    );
                }

                // Check if expected word matches (case-insensitive)
                if !candidates.is_empty() && candidates[0].word.eq_ignore_ascii_case(&filtered_expected_word) {
                    top_match += 1;
                    println!("Result: ✓ Top match ({:?})", elapsed);
                } else if candidates.iter().take(5).any(|c| c.word.eq_ignore_ascii_case(&filtered_expected_word)) {
                    top_5_match += 1;
                    println!("Result: ○ In top 5 ({:?})", elapsed);
                } else {
                    println!("Result: ✗ Not in top 5 ({:?})", elapsed);
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