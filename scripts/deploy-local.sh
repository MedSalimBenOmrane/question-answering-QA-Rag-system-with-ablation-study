#!/bin/bash

set -e

echo "========================================"
echo "RAG System - Local Deployment"
echo "========================================"

check_requirements() {
    echo "Checking requirements..."

    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo "ERROR: Docker Compose is not installed"
        exit 1
    fi

    if [ ! -f .env ]; then
        echo "ERROR: .env file not found"
        echo "Copy .env.example to .env and configure it first"
        exit 1
    fi

    echo "✓ All requirements met"
}

build_images() {
    echo ""
    echo "Building Docker images..."
    docker-compose build --parallel
    echo "✓ Images built successfully"
}

start_services() {
    echo ""
    echo "Starting services..."
    docker-compose up -d
    echo "✓ Services started"
}

wait_for_health() {
    echo ""
    echo "Waiting for services to be healthy..."

    echo -n "Waiting for RAG Pipeline"
    for i in {1..30}; do
        if docker-compose exec -T rag-pipeline curl -f http://localhost:8080/health &> /dev/null; then
            echo " ✓"
            break
        fi
        echo -n "."
        sleep 2
    done

    echo -n "Waiting for Streamlit UI"
    for i in {1..30}; do
        if docker-compose exec -T streamlit-ui curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            echo " ✓"
            break
        fi
        echo -n "."
        sleep 2
    done
}

show_status() {
    echo ""
    echo "========================================"
    echo "Deployment Status"
    echo "========================================"
    docker-compose ps

    echo ""
    echo "Access URLs:"
    echo "  - Streamlit UI: http://localhost:8501"
    echo "  - RAG API: http://localhost:8080"
    echo "  - Health: http://localhost:8080/health"
    echo ""
    echo "Logs:"
    echo "  docker-compose logs -f rag-pipeline"
    echo "  docker-compose logs -f streamlit-ui"
    echo ""
    echo "Stop services:"
    echo "  docker-compose down"
}

main() {
    check_requirements
    build_images
    start_services
    wait_for_health
    show_status

    echo ""
    echo "✓ Local deployment complete!"
}

main
