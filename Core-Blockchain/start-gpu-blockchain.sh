#!/bin/bash

# Splendor Blockchain GPU + AI Startup Script
# This script builds and starts the blockchain with GPU acceleration and AI load balancing

set -e

echo "🚀 Starting Splendor Blockchain with GPU + AI Support"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "node_src/Makefile.cuda" ]; then
    print_error "Please run this script from the Core-Blockchain directory"
    exit 1
fi

cd node_src

print_status "Checking system requirements..."

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    print_success "CUDA found: version $CUDA_VERSION"
else
    print_error "CUDA not found. Please install CUDA toolkit first."
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check Go installation
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | awk '{print $3}')
    print_success "Go found: $GO_VERSION"
else
    print_error "Go not found. Please install Go 1.15+ first."
    exit 1
fi

# Check Python for vLLM
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.8+ for vLLM."
    exit 1
fi

print_status "Building GPU kernels..."

# Build CUDA kernels
make -f Makefile.gpu clean
if make -f Makefile.gpu cuda; then
    print_success "CUDA kernels built successfully"
else
    print_error "Failed to build CUDA kernels"
    exit 1
fi

# Build OpenCL kernels as fallback
if make -f Makefile.gpu opencl; then
    print_success "OpenCL kernels built successfully"
else
    print_warning "OpenCL kernels failed to build (CUDA will be used)"
fi

print_status "Building Geth with GPU support..."

# Build Geth with CUDA support
if make -f Makefile.cuda geth-cuda; then
    print_success "Geth built with GPU support"
else
    print_error "Failed to build Geth with GPU support"
    exit 1
fi

print_status "Setting up AI Load Balancer..."

# Check if vLLM is installed
if python3 -c "import vllm" 2>/dev/null; then
    print_success "vLLM already installed"
else
    print_status "Installing vLLM..."
    if pip3 install vllm; then
        print_success "vLLM installed successfully"
    else
        print_error "Failed to install vLLM"
        exit 1
    fi
fi

print_status "Starting vLLM server with MobileLLM-R1-950M..."

# Start vLLM server in background
python3 -m vllm.entrypoints.openai.api_server \
    --model facebook/MobileLLM-R1-950M \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.1 \
    --max-model-len 4096 \
    --disable-log-requests \
    > vllm.log 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > vllm.pid

print_success "vLLM server started (PID: $VLLM_PID)"
print_status "Waiting for vLLM server to initialize..."

# Wait for vLLM server to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_success "vLLM server is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "vLLM server failed to start within 30 seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo

print_status "Starting Splendor Blockchain node..."

# Set environment variables for GPU and AI
export ENABLE_GPU=true
export PREFERRED_GPU_TYPE=CUDA
export GPU_MAX_BATCH_SIZE=800000
export GPU_MAX_MEMORY_USAGE=19327352832
export ENABLE_HYBRID_PROCESSING=true
export ENABLE_AI_LOAD_BALANCING=true
export LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
export LLM_MODEL=facebook/MobileLLM-R1-950M
export CUDA_VISIBLE_DEVICES=0

# Create cleanup function
cleanup() {
    print_status "Shutting down..."
    if [ -f vllm.pid ]; then
        VLLM_PID=$(cat vllm.pid)
        print_status "Stopping vLLM server (PID: $VLLM_PID)"
        kill $VLLM_PID 2>/dev/null || true
        rm -f vllm.pid
    fi
    print_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_success "🎉 Starting Splendor Blockchain with GPU + AI acceleration!"
print_status "GPU: CUDA with RTX 4000 SFF Ada"
print_status "AI: MobileLLM-R1-950M via vLLM"
print_status "Press Ctrl+C to stop"
echo

# Start the blockchain node
./build/bin/geth \
    --datadir ./chaindata \
    --networkid 2691 \
    --port 30303 \
    --http \
    --http.addr 0.0.0.0 \
    --http.port 8545 \
    --http.api eth,net,web3,personal,miner \
    --http.corsdomain "*" \
    --ws \
    --ws.addr 0.0.0.0 \
    --ws.port 8546 \
    --ws.api eth,net,web3,personal,miner \
    --ws.origins "*" \
    --mine \
    --miner.threads 1 \
    --miner.etherbase 0x0000000000000000000000000000000000000000 \
    --verbosity 4 \
    --log.file blockchain.log

# This line should never be reached unless geth exits
cleanup
