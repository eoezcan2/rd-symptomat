#!/bin/bash

# RD-Symptomat Docker Deployment Script

set -e

echo "🐳 RD-Symptomat Docker Deployment"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose not found. Using docker build and run instead."
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# Function to build and run with Docker Compose
deploy_with_compose() {
    print_status "Building and starting with Docker Compose..."
    
    # Stop existing containers
    docker-compose down 2>/dev/null || true
    
    # Build and start
    docker-compose up --build -d
    
    print_status "Application is starting..."
    print_status "Waiting for application to be ready..."
    
    # Wait for application to be ready
    for i in {1..30}; do
        if curl -f http://localhost:5000/health >/dev/null 2>&1; then
            print_status "✅ Application is ready!"
            print_status "🌐 Access the application at: http://localhost:5000"
            print_status "📊 Health check: http://localhost:5000/health"
            return 0
        fi
        sleep 2
    done
    
    print_error "Application failed to start within 60 seconds"
    print_status "Check logs with: docker-compose logs"
    exit 1
}

# Function to build and run with Docker directly
deploy_with_docker() {
    print_status "Building Docker image..."
    docker build -t rd-symptomat .
    
    # Stop existing container
    docker stop rd-symptomat-app 2>/dev/null || true
    docker rm rd-symptomat-app 2>/dev/null || true
    
    print_status "Starting container..."
    docker run -d \
        --name rd-symptomat-app \
        -p 5000:5000 \
        --restart unless-stopped \
        rd-symptomat
    
    print_status "Application is starting..."
    print_status "Waiting for application to be ready..."
    
    # Wait for application to be ready
    for i in {1..30}; do
        if curl -f http://localhost:5000/health >/dev/null 2>&1; then
            print_status "✅ Application is ready!"
            print_status "🌐 Access the application at: http://localhost:5000"
            print_status "📊 Health check: http://localhost:5000/health"
            return 0
        fi
        sleep 2
    done
    
    print_error "Application failed to start within 60 seconds"
    print_status "Check logs with: docker logs rd-symptomat-app"
    exit 1
}

# Main deployment logic
if [ "$USE_COMPOSE" = true ]; then
    deploy_with_compose
else
    deploy_with_docker
fi

echo ""
print_status "Deployment completed successfully!"
print_status "To stop the application:"
if [ "$USE_COMPOSE" = true ]; then
    echo "  docker-compose down"
else
    echo "  docker stop rd-symptomat-app"
fi
