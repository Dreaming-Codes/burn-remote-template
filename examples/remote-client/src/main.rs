//! Example Remote Client for Burn Remote Backend Server
//!
//! This demonstrates how to connect to a remote Burn server
//! and run GPU-accelerated tensor operations.
//!
//! # Usage
//!
//! 1. Start the burn-server on the remote machine:
//!    ```bash
//!    ./start-burn-server.sh 3000
//!    ```
//!
//! 2. Set the REMOTE_BACKEND_URL environment variable:
//!    ```bash
//!    export REMOTE_BACKEND_URL=ws://your-server-ip:3000
//!    ```
//!
//! 3. Run this client:
//!    ```bash
//!    cargo run --release
//!    ```

use burn::backend::RemoteBackend;
use burn::tensor::Tensor;

type Backend = RemoteBackend;

fn main() {
    let url =
        std::env::var("REMOTE_BACKEND_URL").unwrap_or_else(|_| "ws://localhost:3000".to_string());

    println!("Connecting to Burn Remote Backend at {}...", url);

    // The remote device connects to the WebSocket server
    let device = burn::backend::remote::RemoteDevice::new(&url);

    // Create tensors - these operations run on the remote GPU!
    println!("\n--- Creating tensors on remote GPU ---");

    let a: Tensor<Backend, 2> = Tensor::ones([3, 3], &device);
    println!("Tensor A (ones 3x3):\n{:?}", a);

    let b: Tensor<Backend, 2> = Tensor::random(
        [3, 3],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    println!("Tensor B (random 3x3):\n{:?}", b);

    // Matrix operations
    println!("\n--- Matrix operations on remote GPU ---");

    let c = a.clone() + b.clone();
    println!("A + B:\n{:?}", c);

    let d = a.matmul(b);
    println!("A @ B (matmul):\n{:?}", d);

    println!("\nRemote GPU operations completed successfully!");
}
