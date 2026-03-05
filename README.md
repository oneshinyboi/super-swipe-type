# Super-Swipe-Type

A rust library for processing swipe type data and providing word predictions in real time. It is planned to be a lightweight port of the swipe to type engine powering [Clever Keys](https://github.com/tribixbite/CleverKeys?tab=readme-ov-file).

From the cleverkeys readme:
### Why CleverKeys?

CleverKeys (and by extension this project) uses a custom **transformer neural network** (encoder-decoder architecture) trained specifically for swipe typing. Unlike algorithmic approaches, neural models learn complex patterns from real swipe data. The model architecture, training code, and datasets are all publicly available at [CleverKeys-ML](https://github.com/tribixbite/CleverKeys-ML) — making it fully reproducible and auditable.

**Key differentiators:**
- **Only keyboard with public ML training pipeline** — verify exactly how the model was trained
- **ONNX format** — cross-platform, hardware-accelerated inference via XNNPACK
- **Sub-200ms predictions** — optimized for mobile with beam search decoding
