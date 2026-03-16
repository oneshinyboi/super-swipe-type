# Super-Swipe-Type

A rust library for processing swipe type data and providing word predictions in real time. It is a lightweight port of the swipe to type engine powering [Clever Keys](https://github.com/tribixbite/CleverKeys?tab=readme-ov-file). Only English predictions for english are supported

From the cleverkeys readme:
### Why CleverKeys?

CleverKeys (and by extension this project) uses a custom **transformer neural network** (encoder-decoder architecture) trained specifically for swipe typing. Unlike algorithmic approaches, neural models learn complex patterns from real swipe data. The model architecture, training code, and datasets are all publicly available at [CleverKeys-ML](https://github.com/tribixbite/CleverKeys-ML) — making it fully reproducible and auditable.

**Key differentiators:**
- **Only keyboard with public ML training pipeline** — verify exactly how the model was trained
- **ONNX format** — cross-platform, hardware-accelerated inference via XNNPACK
- **Sub-200ms predictions** — using beam search decoding

## Example usage
```rust
use super_swipe_type::swipe_orchestrator::SwipeOrchestrator;
use super_swipe_type::{SwipeCandidate, SwipePoint};

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
  // many more points
];

// Perform prediction
let result = orchestrator.predict(swipe_points, &None);
let best_predicted_word = result.unwrap().first().unwrap().word
```
The swipe points should be normalized points over the qwerty keyboard area. For example `0,0` is the top left corner of the Q key and `0,1` is top right corner of hte P key


