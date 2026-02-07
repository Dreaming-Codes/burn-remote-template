# List available recipes
default:
    @just --list

# Build the Docker image
build:
    docker compose build

# Start all services (burn-server + Jupyter)
up *args:
    docker compose up {{args}}

# Start all services in detached mode
up-d:
    docker compose up -d

# Start only the burn-server (no Jupyter)
server *args:
    docker compose --profile server-only up {{args}}

# Stop all services
down:
    docker compose down

# View logs
logs *args:
    docker compose logs {{args}}

# Open a shell in the burn-remote container
shell:
    docker compose exec burn-remote bash

# Restart services
restart:
    docker compose restart

# Build and start services
rebuild:
    docker compose up --build

# Run the example remote client
run-client:
    cargo run --manifest-path examples/remote-client/Cargo.toml

# Clean up containers, volumes, and images
clean:
    docker compose down -v --rmi local
