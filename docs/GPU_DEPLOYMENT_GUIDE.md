# [Archived] GPU Deployment Guide for 2M+ TPS

This guide ensures proper deployment of the enhanced GPU transaction processing system for achieving 2M+ TPS on production servers.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU**: RTX 4000 SFF Ada or equivalent (minimum 16GB VRAM)
- **CUDA Compute Capability**: 8.9 or higher
- **System RAM**: Minimum 32GB for optimal performance
- **Storage**: NVMe SSD for fast I/O operations

### Software Dependencies

#### 1. CUDA Toolkit Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

#### 2. OpenCL Installation
```bash
# Ubuntu/Debian
sudo apt install -y opencl-headers ocl-icd-opencl-dev

# NVIDIA OpenCL support
sudo apt install -y nvidia-opencl-dev
```

#### 3. Build Tools
```bash
sudo apt install -y build-essential gcc g++
```

## Deployment Steps

### 1. Clone Repository
```bash
git clone https://github.com/therealdev101/splendor-blockchain-v4.git
cd splendor-blockchain-v4/Core-Blockchain/node_src
```

### 2. Compile GPU Kernels
```bash
# Compile CUDA kernels
nvcc -arch=sm_89 -O3 -Xcompiler -fPIC -shared common/gpu/cuda_kernels.cu -o common/gpu/libcuda_kernels.so

# Compile OpenCL kernels
gcc -fPIC -shared common/gpu/opencl_kernels.c -lOpenCL -o common/gpu/libopencl_kernels.so
```

### 3. Build Blockchain Node
```bash
make geth
```

### 4. Verify GPU Libraries
```bash
ls -la common/gpu/lib*.so
# Should show:
# libcuda_kernels.so
# libopencl_kernels.so
```

## Required Files for Server Deployment

### Core GPU Processing Files
- `Core-Blockchain/node_src/common/gpu/cuda_kernels.cu` - CUDA kernel implementations
- `Core-Blockchain/node_src/common/gpu/opencl_kernels.c` - OpenCL kernel implementations  
- `Core-Blockchain/node_src/common/gpu/gpu_processor.go` - GPU processor coordination
- `Core-Blockchain/node_src/common/gpu/libcuda_kernels.so` - Compiled CUDA library
- `Core-Blockchain/node_src/common/gpu/libopencl_kernels.so` - Compiled OpenCL library

### Build System Files
- `Core-Blockchain/node_src/Makefile` - Main build configuration
- `Core-Blockchain/node_src/Makefile.gpu` - GPU-specific build rules
- `Core-Blockchain/node_src/Makefile.cuda` - CUDA build configuration

### Runtime Dependencies
- `Core-Blockchain/node_src/build/bin/geth` - Compiled blockchain node
- All Go source files in the node_src directory
- Configuration files (.env, genesis.json)

## GPU Configuration (Archived target)

### Optimal GPU Settings
```go
// In production, use these settings for maximum throughput
config := &GPUConfig{
    PreferredGPUType: GPUTypeCUDA,     // CUDA for maximum performance
    MaxBatchSize:     800000,          // 800K transactions per batch
    MaxMemoryUsage:   18 * 1024 * 1024 * 1024, // 18GB VRAM utilization
    HashWorkers:      80,              // 80 parallel hash workers
    SignatureWorkers: 80,              // 80 signature verification workers
    TxWorkers:        80,              // 80 transaction processing workers
    EnablePipelining: true,            // Enable GPU pipelining
}
```

### Performance Optimization
- **Batch Size**: 800K transactions for optimal GPU saturation
- **Memory Usage**: 18GB VRAM (leave 2GB for system/AI workloads)
- **Worker Threads**: 80 workers per operation type for maximum parallelism
- **Pipelining**: Enabled for overlapped CPU-GPU operations

## Server Environment Setup

### 1. Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_PATH=/tmp/cuda_cache
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors
```

### 2. GPU Memory Configuration
```bash
# Set GPU persistence mode for consistent performance
sudo nvidia-smi -pm 1

# Set maximum performance mode
sudo nvidia-smi -ac 877,1410
```

### 3. System Limits
```bash
# Increase file descriptor limits for high throughput
echo "* soft nofile 1048576" >> /etc/security/limits.conf
echo "* hard nofile 1048576" >> /etc/security/limits.conf
```

## Verification Commands

### 1. GPU Detection
```bash
nvidia-smi
lspci | grep -i nvidia
```

### 2. CUDA Verification
```bash
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

### 3. OpenCL Verification
```bash
clinfo
```

### 4. Library Verification
```bash
ldd common/gpu/libcuda_kernels.so
ldd common/gpu/libopencl_kernels.so
```

## Performance Monitoring

### GPU Utilization
```bash
# Monitor GPU usage during operation
nvidia-smi -l 1

# Monitor GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Transaction Processing Metrics
The system logs comprehensive performance metrics:
- Batch TPS calculations
- GPU kernel execution times
- Memory utilization statistics
- Transaction validation results

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. Library Linking Errors
```bash
# Verify library paths
export LD_LIBRARY_PATH=$PWD/common/gpu:$LD_LIBRARY_PATH
```

#### 3. GPU Memory Issues
```bash
# Clear GPU memory
sudo nvidia-smi --gpu-reset
```

## Status Note

This document described a previous, theoretical performance target. The current codebase accelerates parsing and hashing on GPU; signature verification and EVM execution run on CPU for consensus safety. See docs/TECHNICAL_ARCHITECTURE.md for current throughput guidance.

## Production Deployment Checklist

- [ ] NVIDIA GPU with 16GB+ VRAM installed
- [ ] CUDA Toolkit 12.0+ installed and verified
- [ ] OpenCL runtime installed and functional
- [ ] GPU kernels compiled successfully
- [ ] Blockchain node builds without errors
- [ ] GPU libraries linked correctly
- [ ] Performance monitoring configured
- [ ] System limits optimized for high throughput
- [ ] GPU persistence mode enabled
- [ ] Memory configuration optimized

## Support

For deployment issues or performance optimization, refer to:
- GPU processing logs in the blockchain node output
- NVIDIA system management interface (nvidia-smi)
- CUDA profiler tools for detailed performance analysis
- OpenCL debugging tools for compatibility issues

The enhanced GPU transaction processing system is now ready for production deployment with 2M+ TPS capability.
