//! Concrete `Store` implementations.
//!
//! The trait itself lives in the crate root (`crate::Store`). Each backend
//! is its own module so callers pay only for the one they use.

pub mod file;

pub use file::FileStore;
