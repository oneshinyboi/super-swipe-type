use std::cell::RefCell;
use ort::memory::{Allocator, MemoryInfo};
use ort::session::Session;
use crate::{OrtEnvironment, SwipeOrchestrator};

impl SwipeOrchestrator {
    pub fn new() -> Self {
        let session = Session::builder().unwrap().commit_from_file("assets/swipe_encoder_android.onnx").unwrap();
        let allocator = Allocator::new(&session, MemoryInfo::default()).unwrap();
        Self {
            ort_environment: RefCell::new(OrtEnvironment {
                session,
                allocator
            })
        }
    }
}