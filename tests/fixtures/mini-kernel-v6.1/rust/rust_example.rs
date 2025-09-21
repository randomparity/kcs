//! A simple "Hello, world!" kernel module in Rust.

use kernel::prelude::*;

module! {
    type: RustExample,
    name: b"rust_example",
    author: b"Jules",
    description: b"A simple Rust kernel module.",
    license: b"GPL",
}

struct RustExample;

impl KernelModule for RustExample {
    fn init(_module: &'static ThisModule) -> Result<Self> {
        pr_info!("Hello, world from a Rust kernel module!\n");
        Ok(RustExample)
    }
}

impl Drop for RustExample {
    fn drop(&mut self) {
        pr_info!("Goodbye, world from a Rust kernel module!\n");
    }
}
