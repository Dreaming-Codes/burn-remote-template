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

# Install cargo-binstall for fast prebuilt binary installs
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install tools via binstall (prebuilt binaries, no compilation)
RUN cargo binstall -y --locked sccache just zellij

# Now enable sccache as the rustc wrapper for all subsequent builds
ENV RUSTC_WRAPPER=sccache

# Create sccache directory with proper permissions
RUN mkdir -p /workspace/.sccache && chmod 777 /workspace/.sccache

# Install evcxr_jupyter for Rust Jupyter kernel
RUN cargo binstall -y --locked evcxr_jupyter \
    && evcxr_jupyter --install


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

# Create a supervisor config for burn-server (auto-start enabled)
COPY <<'EOF' /etc/supervisor/conf.d/burn-server.conf
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

- **Burn Remote Server**: Auto-starts on port 3000
  - Connect via WebSocket: ws://your-server-ip:3000
- **Jupyter Notebook**: Auto-starts on port 8888 (no password)
  - Access at: http://localhost:8888
  - Rust kernel available via evcxr

## Quick Start

### Manual Start (if needed)
```bash
./start-burn-server.sh [port]
# Default port: 3000
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

# Install zellij configuration
RUN mkdir -p /root/.config/zellij
COPY <<'ZELLIJ_EOF' /root/.config/zellij/config.kdl
// If you'd like to override the default keybindings completely, be sure to change "keybinds" to "keybinds clear-defaults=true"
show_startup_tips false
ui {
    pane_frames {
        hide_session_name false
    }
}
keybinds clear-defaults=true {
	normal {
		unbind "Ctrl p"
		unbind "Ctrl o"
		unbind "Ctrl q"
		unbind "Ctrl h"
		// uncomment this and adjust key if using copy_on_select=false
		// bind "Alt c" { Copy; }
	}
	locked {
		bind "Ctrl g" { SwitchToMode "Normal"; }
	}
	resize {
		bind "Ctrl n" { SwitchToMode "Normal"; }
		bind "h" "Left" { Resize "Increase Left"; }
		bind "j" "Down" { Resize "Increase Down"; }
		bind "k" "Up" { Resize "Increase Up"; }
		bind "l" "Right" { Resize "Increase Right"; }
		bind "H" { Resize "Decrease Left"; }
		bind "J" { Resize "Decrease Down"; }
		bind "K" { Resize "Decrease Up"; }
		bind "L" { Resize "Decrease Right"; }
		bind "=" "+" { Resize "Increase"; }
		bind "-" { Resize "Decrease"; }
	}
	pane {
		bind "Ctrl a" { SwitchToMode "Normal"; }
		bind "h" "Left" { MoveFocus "Left"; SwitchToMode "Normal"; }
		bind "l" "Right" { MoveFocus "Right"; SwitchToMode "Normal"; }
		bind "j" "Down" { MoveFocus "Down"; SwitchToMode "Normal"; }
		bind "k" "Up" { MoveFocus "Up"; SwitchToMode "Normal"; }
		bind "p" { SwitchFocus; SwitchToMode "Normal"; }
		bind "n" { NewPane; SwitchToMode "Normal"; }
		bind "d" { NewPane "Down"; SwitchToMode "Normal"; }
		//bind "r" { NewPane "Right"; SwitchToMode "Normal"; }
		bind "x" { CloseFocus; SwitchToMode "Normal"; }
		bind "z" { ToggleFocusFullscreen; SwitchToMode "Normal"; }
		bind "f" { TogglePaneFrames; SwitchToMode "Normal"; }
		bind "w" { ToggleFloatingPanes; SwitchToMode "Normal"; }
		bind "Ctrl a" { ToggleFloatingPanes; SwitchToMode "Normal"; }
		bind "e" { TogglePaneEmbedOrFloating; SwitchToMode "Normal"; }
		bind "r" { SwitchToMode "RenamePane"; PaneNameInput 0;}
	}
	tab {
		bind "Ctrl t" { SwitchToMode "Normal"; }
		bind "r" { SwitchToMode "RenameTab"; TabNameInput 0; }
		bind "h" "Left" "Up" "k" { GoToPreviousTab; SwitchToMode "Normal"; }
		bind "l" "Right" "Down" "j" { GoToNextTab; SwitchToMode "Normal"; }
		bind "n" { NewTab; SwitchToMode "Normal"; SwitchToMode "Normal"; }
		bind "x" { CloseTab; SwitchToMode "Normal"; SwitchToMode "Normal"; }
		bind "s" { ToggleActiveSyncTab; SwitchToMode "Normal"; }
		bind "b" { BreakPane; SwitchToMode "Normal"; }
		bind "]" { BreakPaneRight; SwitchToMode "Normal"; }
		bind "[" { BreakPaneLeft; SwitchToMode "Normal"; }
		bind "1" { GoToTab 1; SwitchToMode "Normal"; }
		bind "2" { GoToTab 2; SwitchToMode "Normal"; }
		bind "3" { GoToTab 3; SwitchToMode "Normal"; }
		bind "4" { GoToTab 4; SwitchToMode "Normal"; }
		bind "5" { GoToTab 5; SwitchToMode "Normal"; }
		bind "6" { GoToTab 6; SwitchToMode "Normal"; }
		bind "7" { GoToTab 7; SwitchToMode "Normal"; }
		bind "8" { GoToTab 8; SwitchToMode "Normal"; }
		bind "9" { GoToTab 9; SwitchToMode "Normal"; }
		bind "a" { ToggleTab; SwitchToMode "Normal"; }
	}
	scroll {
		bind "Ctrl s" { SwitchToMode "Normal"; }
		bind "e" { EditScrollback; SwitchToMode "Normal"; }
		bind "s" { SwitchToMode "EnterSearch"; SearchInput 0; }
		bind "G" { ScrollToBottom; SwitchToMode "Normal"; }
		bind "j" "Down" { ScrollDown; }
		bind "k" "Up" { ScrollUp; }
		bind "Ctrl f" "PageDown" "Right" "l" { PageScrollDown; }
		bind "Ctrl b" "PageUp" "Left" "h" { PageScrollUp; }
		bind "d" { HalfPageScrollDown; }
		bind "u" { HalfPageScrollUp; }
		// uncomment this and adjust key if using copy_on_select=false
		// bind "Alt c" { Copy; }
	}
	search {
		bind "Ctrl /" { SwitchToMode "Normal"; }
		bind "j" "Down" { ScrollDown; }
		bind "k" "Up" { ScrollUp; }
		bind "Ctrl f" "PageDown" "Right" "l" { PageScrollDown; }
		bind "Ctrl b" "PageUp" "Left" "h" { PageScrollUp; }
		bind "d" { HalfPageScrollDown; }
		bind "u" { HalfPageScrollUp; }
		bind "n" { Search "down"; }
		bind "p" { Search "up"; }
		bind "c" { SearchToggleOption "CaseSensitivity"; }
		bind "w" { SearchToggleOption "Wrap"; }
		bind "o" { SearchToggleOption "WholeWord"; }
	}
	entersearch {
		bind "Ctrl s" "Esc" { SwitchToMode "Scroll"; }
		bind "Enter" { SwitchToMode "Search"; }
	}
	renametab {
		bind "Ctrl c" { SwitchToMode "Normal"; }
		bind "Esc" { UndoRenameTab; SwitchToMode "Tab"; }
	}
	renamepane {
		bind "Ctrl c" { SwitchToMode "Normal"; }
		bind "Esc" { UndoRenamePane; SwitchToMode "Pane"; }
	}
	session {
		bind "Ctrl x" { SwitchToMode "Normal"; }
		bind "Ctrl x" { SwitchToMode "Scroll"; }
		bind "d" { Detach; }
		bind "w" {
		    LaunchOrFocusPlugin "zellij:session-manager" {
			floating true
			move_to_focused_tab true
		    };
		    SwitchToMode "Normal"
		}
	}
	tmux {
		bind "[" { SwitchToMode "Scroll"; }
		bind "Ctrl b" { Write 2; SwitchToMode "Normal"; }
		bind "\"" { NewPane "Down"; SwitchToMode "Normal"; }
		bind "%" { NewPane "Right"; SwitchToMode "Normal"; }
		bind "z" { ToggleFocusFullscreen; SwitchToMode "Normal"; }
		bind "c" { NewTab; SwitchToMode "Normal"; }
		bind "," { SwitchToMode "RenameTab"; }
		bind "p" { GoToPreviousTab; SwitchToMode "Normal"; }
		bind "n" { GoToNextTab; SwitchToMode "Normal"; }
		bind "Left" { MoveFocus "Left"; SwitchToMode "Normal"; }
		bind "Right" { MoveFocus "Right"; SwitchToMode "Normal"; }
		bind "Down" { MoveFocus "Down"; SwitchToMode "Normal"; }
		bind "Up" { MoveFocus "Up"; SwitchToMode "Normal"; }
		bind "h" { MoveFocus "Left"; SwitchToMode "Normal"; }
		bind "l" { MoveFocus "Right"; SwitchToMode "Normal"; }
		bind "j" { MoveFocus "Down"; SwitchToMode "Normal"; }
		bind "k" { MoveFocus "Up"; SwitchToMode "Normal"; }
		bind "o" { FocusNextPane; }
		bind "d" { Detach; }
		bind "Space" { NextSwapLayout; }
		bind "x" { CloseFocus; SwitchToMode "Normal"; }
	}
	shared_except "locked" {
		bind "Ctrl g" { SwitchToMode "Locked"; }
		bind "Alt n" { NewPane; }
		bind "Alt h" "Alt Left" { MoveFocusOrTab "Left"; }
		bind "Alt l" "Alt Right" { MoveFocusOrTab "Right"; }
		bind "Alt j" "Alt Down" { MoveFocus "Down"; }
		bind "Alt k" "Alt Up" { MoveFocus "Up"; }
		bind "Alt =" "Alt +" { Resize "Increase"; }
		bind "Alt -" { Resize "Decrease"; }
		bind "Alt [" { PreviousSwapLayout; }
		bind "Alt ]" { NextSwapLayout; }
	}
	shared_except "normal" "locked" {
		bind "Enter" "Esc" { SwitchToMode "Normal"; }
	}
	shared_except "pane" "locked" {
		bind "Ctrl a" { SwitchToMode "Pane"; }
	}
	shared_except "resize" "locked" {
		bind "Ctrl n" { SwitchToMode "Resize"; }
	}
	shared_except "scroll" "locked" {
		bind "Ctrl s" { SwitchToMode "Scroll"; }
	}
	shared_except "session" "locked" {
		bind "Ctrl x" { SwitchToMode "Session"; }
	}
	shared_except "tab" "locked" {
		bind "Ctrl t" { SwitchToMode "Tab"; }
	}
	shared_except "renametab" "locked" {
		bind "Alt r" { SwitchToMode "RenameTab"; }
	}
	shared_except "tmux" "locked" {
		bind "Ctrl b" { SwitchToMode "Tmux"; }
	}
}

plugins {
	tab-bar { path "tab-bar"; }
	status-bar { path "status-bar"; }
	strider { path "strider"; }
	compact-bar { path "compact-bar"; }
}

on_force_close "detach"
simplified_ui false
pane_frames false
theme "catppuccin-mocha"

themes {
    catppuccin-frappe {
        fg 198 208 245
        bg 98 104 128
        black 41 44 60
        red 231 130 132
        green 166 209 137
        yellow 229 200 144
        blue 140 170 238
        magenta 244 184 228
        cyan 153 209 219
        white 198 208 245
        orange 239 159 118
    }
    catppuccin-latte {
        fg 172 176 190
        bg 172 176 190
        black 76 79 105
        red 210 15 57
        green 64 160 43
        yellow 223 142 29
        blue 30 102 245
        magenta 234 118 203
        cyan 4 165 229
        white 220 224 232
        orange 254 100 11
    }
    catppuccin-macchiato {
        fg 202 211 245
        bg 91 96 120
        black 30 32 48
        red 237 135 150
        green 166 218 149
        yellow 238 212 159
        blue 138 173 244
        magenta 245 189 230
        cyan 145 215 227
        white 202 211 245
        orange 245 169 127
    }
    catppuccin-mocha {
        fg 205 214 244
        bg 88 91 112
        black 24 24 37
        red 243 139 168
        green 166 227 161
        yellow 249 226 175
        blue 137 180 250
        magenta 245 194 231
        cyan 137 220 235
        white 205 214 244
        orange 250 179 135
    }
}
ZELLIJ_EOF

# Disable vast.ai's auto-tmux and use zellij instead
RUN touch /root/.no_auto_tmux && \
    cat >> /root/.bashrc <<'BASHRC_EOF'

# Auto-start zellij on interactive login
if [ -z "$ZELLIJ" ] && command -v zellij &> /dev/null && [[ $- == *i* ]]; then
    exec zellij attach -c main
fi
BASHRC_EOF

# Set working directory
WORKDIR /workspace

# Expose the burn-server port
EXPOSE 3000
# Jupyter port
EXPOSE 8888

# Default command - can be overridden
CMD ["bash"]
