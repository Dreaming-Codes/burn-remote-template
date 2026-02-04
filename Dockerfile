# Burn Remote Server Docker Image
# Based on vastai/base-image with CUDA support
# Configured for Burn GPU development and remote server

FROM vastai/base-image:cuda-13.1.0-auto

LABEL maintainer="Burn Remote Template"
LABEL description="Burn GPU Development Environment with CUDA support, Rust Jupyter kernel, and burn-server"

# Environment variables for Rust
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH="/opt/cargo/bin:${PATH}" \
    RUST_BACKTRACE=1 \
    REMOTE_BACKEND_PORT=3000

# sccache configuration - shared cache for all Rust builds (including Jupyter evcxr)
# Note: RUSTC_WRAPPER is set AFTER sccache is installed
ENV SCCACHE_DIR=/workspace/.sccache \
    SCCACHE_CACHE_SIZE="10G"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    pkg-config \
    cmake \
    # SSL and crypto
    libssl-dev \
    ca-certificates \
    # Additional dev tools
    git \
    curl \
    wget \
    # For evcxr
    libzmq3-dev \
    # Monitoring
    btop \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain stable \
    --profile default \
    && rustup component add rust-src rustfmt clippy rust-analyzer

# Install sccache for shared compilation cache
RUN cargo install sccache --locked

# Now enable sccache as the rustc wrapper for all subsequent builds
ENV RUSTC_WRAPPER=sccache

# Create sccache directory with proper permissions
RUN mkdir -p /workspace/.sccache && chmod 777 /workspace/.sccache

# Install evcxr_jupyter for Rust Jupyter kernel (uses sccache)
RUN cargo install --locked evcxr_jupyter \
    && evcxr_jupyter --install

# Install additional useful Rust tools for development
RUN cargo install cargo-watch cargo-expand

# Create workspace directory
WORKDIR /workspace

# Create the burn-server project
RUN mkdir -p /workspace/burn-server/src

# Create Cargo.toml for burn-server
COPY <<'EOF' /workspace/burn-server/Cargo.toml
[package]
name = "burn-server"
version = "0.1.0"
edition = "2021"
description = "Burn Remote Backend Server with CUDA support"

[features]
default = ["cuda"]
cuda = ["burn/cuda"]
wgpu = ["burn/wgpu"]

[dependencies]
burn = { version = "0.20", features = ["server"] }
cfg-if = "1.0"
EOF

# Create main.rs for burn-server
COPY <<'EOF' /workspace/burn-server/src/main.rs
//! Burn Remote Backend Server
//! 
//! This server provides GPU-accelerated tensor operations via WebSocket.
//! Connect your remote Burn client to this server to leverage CUDA GPU acceleration.

fn main() {
    burn_server::start();
}
EOF

# Create lib.rs for burn-server (based on burn examples/server)
COPY <<'EOF' /workspace/burn-server/src/lib.rs
#![recursion_limit = "141"]

/// Start the Burn remote backend server.
/// 
/// The server listens on the port specified by the REMOTE_BACKEND_PORT environment variable,
/// defaulting to port 3000 if not set.
/// 
/// # Backends
/// 
/// The backend is selected at compile time via features:
/// - `cuda` (default): NVIDIA CUDA backend for GPU acceleration
/// - `wgpu`: WebGPU backend for cross-platform GPU acceleration
pub fn start() {
    let port = std::env::var("REMOTE_BACKEND_PORT")
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
        .unwrap_or(3000);

    println!("Starting Burn Remote Backend Server on port {}...", port);
    
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            println!("Backend: CUDA (GPU)");
            burn::server::start_websocket::<burn::backend::Cuda>(Default::default(), port);
        } else if #[cfg(feature = "wgpu")] {
            println!("Backend: WebGPU (GPU)");
            burn::server::start_websocket::<burn::backend::Wgpu>(Default::default(), port);
        } else {
            panic!("No backend selected, can't start server on port {port}");
        }
    }
}
EOF

# Build the burn-server to cache dependencies (CUDA backend)
# This also warms up sccache with burn dependencies
WORKDIR /workspace/burn-server
RUN cargo build --release --features cuda && sccache --show-stats

# Create a sample Rust Jupyter notebook for Burn
WORKDIR /workspace
RUN mkdir -p /workspace/notebooks
COPY <<'EOF' /workspace/notebooks/burn_example.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Burn GPU Development with Rust\n",
    "\n",
    "This notebook demonstrates using Burn with CUDA in Jupyter.\n",
    "\n",
    "**Note**: First cell compilation may take a moment, but sccache speeds up subsequent builds!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ":dep burn = { version = \"0.20\", features = [\"cuda\"] }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "use burn::tensor::Tensor;\n",
    "use burn::backend::Cuda;\n",
    "\n",
    "type Backend = Cuda;\n",
    "\n",
    "// Create a simple tensor on the GPU\n",
    "let device = Default::default();\n",
    "let tensor: Tensor<Backend, 2> = Tensor::ones([3, 3], &device);\n",
    "println!(\"Tensor on CUDA:\\n{:?}\", tensor);"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Rust",
   "language": "rust",
   "name": "rust"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create startup script for the burn-server
COPY <<'EOF' /workspace/start-burn-server.sh
#!/bin/bash
# Start the Burn Remote Backend Server
# Usage: ./start-burn-server.sh [port]

export REMOTE_BACKEND_PORT=${1:-3000}
echo "Starting Burn Remote Backend Server on port $REMOTE_BACKEND_PORT..."
cd /workspace/burn-server
cargo run --release --features cuda
EOF
RUN chmod +x /workspace/start-burn-server.sh

# Create a supervisor config for burn-server (optional auto-start)
COPY <<'EOF' /etc/supervisor/conf.d/burn-server.conf.disabled
[program:burn-server]
command=/workspace/burn-server/target/release/burn-server
directory=/workspace/burn-server
autostart=true
autorestart=true
stderr_logfile=/var/log/burn-server.err.log
stdout_logfile=/var/log/burn-server.out.log
environment=REMOTE_BACKEND_PORT="3000"
EOF

# Create supervisor config for Jupyter Notebook (auto-start enabled)
COPY <<'EOF' /etc/supervisor/conf.d/jupyter.conf
[program:jupyter]
command=jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
directory=/workspace
autostart=true
autorestart=true
stderr_logfile=/var/log/jupyter.err.log
stdout_logfile=/var/log/jupyter.out.log
environment=SCCACHE_DIR="/workspace/.sccache",RUSTC_WRAPPER="sccache",PATH="/opt/cargo/bin:%(ENV_PATH)s"
EOF

# Create README for the workspace
COPY <<'EOF' /workspace/README.md
# Burn Remote Development Environment

This environment is configured for GPU-accelerated Burn development with CUDA.

## Services (Auto-started)

- **Jupyter Notebook**: Auto-starts on port 8888 (no password)
  - Access at: http://localhost:8888
  - Rust kernel available via evcxr

## Quick Start

### Start the Burn Remote Backend Server
```bash
./start-burn-server.sh [port]
# Default port: 3000
```

Or enable auto-start:
```bash
mv /etc/supervisor/conf.d/burn-server.conf.disabled /etc/supervisor/conf.d/burn-server.conf
supervisorctl reload
```

### Connect from a Remote Client
```rust
use burn::backend::RemoteBackend;

type Backend = RemoteBackend;
let device = burn::backend::remote::RemoteDevice::new("ws://your-server-ip:3000");
```

## Available Tools

- **Rust**: Stable toolchain with rust-analyzer, clippy, rustfmt
- **Burn**: Pre-compiled with CUDA backend (v0.20)
- **Jupyter**: Rust kernel via evcxr_jupyter (auto-started)
- **sccache**: Shared compilation cache for faster builds
- **btop**: System/GPU monitoring

## Service Management

```bash
# Check service status
supervisorctl status

# Restart Jupyter
supervisorctl restart jupyter

# View Jupyter logs
tail -f /var/log/jupyter.out.log
```

## sccache (Shared Build Cache)

This environment uses sccache to share compiled artifacts between:
- The burn-server project
- Jupyter notebook Rust cells
- Any other Rust projects you create

The cache is stored in `/workspace/.sccache`. Check stats with:
```bash
sccache --show-stats
```

## Jupyter with Rust

Jupyter is auto-started. Just open http://localhost:8888 and select the "Rust" kernel.

See `/workspace/notebooks/burn_example.ipynb` for examples.

Thanks to sccache, burn dependencies are pre-cached, so notebook cells compile faster!

## Environment Variables

- `REMOTE_BACKEND_PORT`: Port for the burn-server (default: 3000)
- `SCCACHE_DIR`: sccache cache directory (default: /workspace/.sccache)
- `SCCACHE_CACHE_SIZE`: Max cache size (default: 10G)

## GPU Monitoring

```bash
btop          # Interactive system monitor
nvidia-smi    # NVIDIA GPU status
```

## Building with Different Backends

```bash
cd /workspace/burn-server

# CUDA (default)
cargo build --release --features cuda

# WebGPU
cargo build --release --features wgpu --no-default-features
```
EOF

# Set working directory
WORKDIR /workspace

# Expose the burn-server port
EXPOSE 3000
# Jupyter port
EXPOSE 8888

# Default command - can be overridden
CMD ["bash"]
