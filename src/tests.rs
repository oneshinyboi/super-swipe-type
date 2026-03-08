use crate::SOS_IDX;
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::keyboard_manager::QwertyKeyboardGrid;
use crate::beam_search::BeamSearchEngine;
use crate::swipe_trajectory_processor::{FeaturePoint, SwipeTrajectoryProcessor};
use crate::{EncodeResult, SwipePoint};
use vector2::Vector2;
use ort::session::Session;
use serde::Deserialize;
use std::fs;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct SwipeData {
    published: Vec<SwipePointData>,
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
        .commit_from_file("assets/swipe_encoder_android.onnx")
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
        .commit_from_file("assets/swipe_decoder_android.onnx")
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
    let encoder = create_encoder();
    let features = create_test_features();
    encoder.encode(features).unwrap()
}

// Helper function to load swipe data from JSON
fn load_swipe_data() -> Vec<SwipePoint> {
    let json_content = fs::read_to_string("testing/swipes.json")
        .expect("Failed to read swipes.json");
    
    let swipe_data: SwipeData = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON");
    
    // Find the minimum timestamp to normalize
    let min_timestamp = swipe_data.published.iter()
        .map(|p| p.t)
        .min()
        .unwrap_or(0);
    
    // Convert to SwipePoint with normalized timestamps
    swipe_data.published.iter()
        .map(|p| SwipePoint {
            point: Vector2 { x: p.x, y: p.y },
            timestamp: Duration::from_millis(p.t - min_timestamp),
        })
        .collect()
}

#[test]
fn test_encoder() {
    let encoder = create_encoder();
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
fn test_beam_search_with_different_parameters() {
    let encode_result = create_encode_result();
    let decoder = create_decoder(encode_result);
    
    // Test with narrow beam (should be faster, fewer results)
    let mut narrow_beam = BeamSearchEngine::new(decoder, 2, 5, 10);
    let narrow_result = narrow_beam.search();
    assert!(narrow_result.is_ok());
    
    let narrow_candidates = narrow_result.unwrap();
    println!("\n=== Narrow Beam Search (width=2, branching=5) ===");
    println!("Found {} candidates", narrow_candidates.len());
    for (i, candidate) in narrow_candidates.iter().take(5).enumerate() {
        println!("{}. \"{}\" ({:.4})", i + 1, candidate.word, candidate.confidence);
    }
    
    // Test with wide beam (should find more diverse results)
    let encode_result2 = create_encode_result();
    let decoder2 = create_decoder(encode_result2);
    let mut wide_beam = BeamSearchEngine::new(decoder2, 10, 15, 15);
    let wide_result = wide_beam.search();
    assert!(wide_result.is_ok());
    
    let wide_candidates = wide_result.unwrap();
    println!("\n=== Wide Beam Search (width=10, branching=15) ===");
    println!("Found {} candidates", wide_candidates.len());
    for (i, candidate) in wide_candidates.iter().take(5).enumerate() {
        println!("{}. \"{}\" ({:.4})", i + 1, candidate.word, candidate.confidence);
    }
    println!("===========================\n");
    
    // Wide beam should generally find more candidates
    assert!(
        wide_candidates.len() >= narrow_candidates.len(),
        "Wide beam should find at least as many candidates as narrow beam"
    );
}

#[test]
fn test_beam_search_with_real_swipe_data() {
    println!("\n=== Testing with Real Swipe Data ===");
    
    // Load swipe data from JSON
    let swipe_points = load_swipe_data();
    println!("Loaded {} swipe points", swipe_points.len());
    
    // Create trajectory processor
    let processor = SwipeTrajectoryProcessor::new(250);
    
    // Extract features from swipe
    let features = processor.extract_features(swipe_points);
    println!("Extracted {} feature points", features.len());
    
    // Encode features
    let encoder = create_encoder();
    let encode_result = encoder.encode(features)
        .expect("Failed to encode features");
    println!("Successfully encoded features");
    
    // Create decoder
    let decoder = create_decoder(encode_result);
    
    // Run beam search with optimal parameters
    let beam_width = 10;
    let branching_factor = 15;
    let max_levels = 20;
    
    let mut beam_search = BeamSearchEngine::new(
        decoder,
        beam_width,
        branching_factor,
        max_levels
    );
    
    println!("\nRunning beam search (width={}, branching={}, max_levels={})...", 
        beam_width, branching_factor, max_levels);
    
    let result = beam_search.search();
    assert!(result.is_ok(), "Beam search should complete successfully");
    
    let candidates = result.unwrap();
    
    // Print results
    println!("\n=== Decoded Words from Real Swipe ===");
    println!("Found {} candidate words:\n", candidates.len());
    
    for (i, candidate) in candidates.iter().take(20).enumerate() {
        println!("{}. \"{}\" (confidence: {:.4})", 
            i + 1, 
            candidate.word, 
            candidate.confidence
        );
    }
    
    if candidates.len() > 20 {
        println!("\n... and {} more candidates", candidates.len() - 20);
    }
    
    println!("\n======================================\n");
    
    // Assertions
    assert!(!candidates.is_empty(), "Should find at least one candidate word");
    assert!(candidates[0].confidence > 0.0, "Top candidate should have positive confidence");
    
    // Verify candidates are sorted by confidence
    for i in 0..candidates.len().saturating_sub(1) {
        assert!(
            candidates[i].confidence >= candidates[i + 1].confidence,
            "Candidates should be sorted by confidence in descending order"
        );
    }
}