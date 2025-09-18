# Splendor Blockchain Technical Architecture

## 🏗️ Overview

Splendor Blockchain V4 combines **RTX 4000 SFF Ada GPU acceleration** and **advanced parallel processing**. This guide explains the technical implementation details based on the actual Go codebase.

## System Overview

1. **Collect Metrics**: TPS, CPU%, GPU%, latency, batch size, queue depth
2. **Batch Processing**: Prepare transactions for efficient GPU/CPU processing
3. **GPU Offload**: Parse/Keccak on GPU (optional, safety‑checked)
4. **CPU Verify/EVM**: Final signature verification and EVM execution on CPU for consensus safety

## ⚡ GPU Acceleration System

### GPU Processor Implementation

**Location**: `Core-Blockchain/node_src/common/gpu/gpu_processor.go`

The GPU system supports both CUDA and OpenCL with RTX 4000 SFF Ada optimization:

```go
// DefaultGPUConfig returns optimized GPU configuration for NVIDIA RTX 4000 SFF Ada (20GB VRAM)
// Balanced for blockchain processing
func DefaultGPUConfig() *GPUConfig {
    return &GPUConfig{
        PreferredGPUType: GPUTypeCUDA,   // Prefer CUDA for RTX 4000 SFF Ada when available
        MaxBatchSize:     800000,        // 4x increase - 800K batches (keeps GPU saturated)
        MaxMemoryUsage:   18 * 1024 * 1024 * 1024, // 18GB GPU memory (leave 2GB for MobileLLM + system)
        HashWorkers:      80,            // 80 workers - balance with AI workload
        SignatureWorkers: 80,            // 80 signature verification workers
        TxWorkers:        80,            // 80 transaction processing workers
        EnablePipelining: true,          // Enable GPU pipelining
    }
}
```

**GPU Capabilities:**
- **VRAM**: 20GB GDDR6
- **CUDA Cores**: 6,144 Ada Lovelace cores
- **Memory Bandwidth**: 360 GB/s
- **Batch Sizes**: Tunable per VRAM limits

### GPU Processing Pipeline

1. **Hash Processing**: Keccak-256 hashing on GPU
2. **Signature Verification**: ECDSA verification in parallel
3. **Transaction Processing**: Full transaction validation
4. **Memory Management**: Efficient GPU memory pooling
5. **Fallback System**: CPU fallback for reliability

## 🔄 Hybrid Processing System

### Hybrid Processor Implementation

**Location**: `Core-Blockchain/node_src/common/hybrid/hybrid_processor.go`

The hybrid system intelligently distributes workload between CPU and GPU:

```go
func DefaultHybridConfig() *HybridConfig {
    return &HybridConfig{
        CPUConfig:             gopool.DefaultProcessorConfig(),
        GPUConfig:             gpu.DefaultGPUConfig(),
        EnableGPU:             true,
        GPUThreshold:          500,   // 10x lower - Use GPU for batches >= 500 (massive GPU utilization)
        CPUGPURatio:           0.90,  // 90% GPU, 10% CPU for maximum GPU utilization
        AdaptiveLoadBalancing: true,
        MaxCPUUtilization:     0.95,  // 95% max CPU usage (higher utilization)
        MaxGPUUtilization:     0.98,  // 98% max GPU usage (push RTX 4000 SFF Ada to limits)
        LatencyThreshold:      25 * time.Millisecond,  // 2x faster latency target
        MaxMemoryUsage:        64 * 1024 * 1024 * 1024, // 64GB total system memory
        GPUMemoryReservation:  18 * 1024 * 1024 * 1024, // 18GB GPU reserved (RTX 4000 SFF Ada 20GB)
    }
}
```

**Processing Strategies:**
- **CPU_ONLY**: Small batches (< 500 transactions)
- **GPU_ONLY**: Large batches (> 2500 transactions)
- **HYBRID**: Medium batches with intelligent splitting

### Load Balancing Logic

```go
func (h *HybridProcessor) determineProcessingStrategy(batchSize int) ProcessingStrategy {
    if !h.config.EnableGPU || h.gpuProcessor == nil {
        return ProcessingStrategyCPUOnly
    }
    
    // Small batches go to CPU
    if batchSize < h.config.GPUThreshold {
        return ProcessingStrategyCPUOnly
    }
    
    cpuUtil := h.loadBalancer.cpuUtilization
    gpuUtil := h.loadBalancer.gpuUtilization
    
    // If CPU is overloaded, prefer GPU
    if cpuUtil > h.config.MaxCPUUtilization {
        if gpuUtil < h.config.MaxGPUUtilization {
            return ProcessingStrategyGPUOnly
        }
    }
    
    // If GPU is underutilized and batch is large, use hybrid
    if batchSize > h.config.GPUThreshold*2 && gpuUtil < 0.5 {
        return ProcessingStrategyHybrid
    }
    
    // Default to GPU for large batches
    if batchSize > h.config.GPUThreshold*5 {
        return ProcessingStrategyGPUOnly
    }
    
    return ProcessingStrategyCPUOnly
}
```

## 🔀 Parallel Processing System

### Parallel State Processor

**Location**: `Core-Blockchain/node_src/core/parallel_state_processor.go`

Advanced parallel processing with multiple strategies:

```go
// DefaultParallelProcessorConfig returns optimized default configuration
func DefaultParallelProcessorConfig() *ParallelProcessorConfig {
    numCPU := runtime.NumCPU()
    return &ParallelProcessorConfig{
        MaxTxConcurrency:     numCPU * 4,
        TxBatchSize:          100,
        TxTimeout:            30 * time.Second,
        MaxValidationWorkers: numCPU * 2,
        ValidationTimeout:    15 * time.Second,
        StateWorkers:         numCPU,
        StateTimeout:         20 * time.Second,
        EnablePipelining:     true,
        EnableTxBatching:     true,
        EnableBloomParallel:  true,
        AdaptiveScaling:      true,
        MaxMemoryUsage:       1024 * 1024 * 1024, // 1GB
        MaxGoroutines:        numCPU * 8,
    }
}
```

### Processing Strategies

#### 1. **Batched Processing**
- Groups transactions for parallel execution
- Maintains state consistency within batches
- Parallel bloom filter creation

#### 2. **Pipelined Processing**
- 3-stage pipeline: validation → execution → collection
- Concurrent processing of different stages
- Higher throughput for large transaction volumes

#### 3. **Sequential Fallback**
- Ensures reliability under all conditions
- Used for small batches or when parallel processing fails

### Parallel Worker Pools

**Location**: `Core-Blockchain/node_src/common/gopool/parallel_processor.go`

Multi-pool worker system with specialized pools:

```go
// ProcessorConfig holds configuration for the parallel processor
type ProcessorConfig struct {
    MaxWorkers        int           `json:"maxWorkers"`
    QueueSize         int           `json:"queueSize"`
    Timeout           time.Duration `json:"timeout"`
    TxWorkers         int           `json:"txWorkers"`         // Transaction processing workers
    ValidationWorkers int           `json:"validationWorkers"` // Validation workers
    StateWorkers      int           `json:"stateWorkers"`      // State processing workers
    ConsensusWorkers  int           `json:"consensusWorkers"`  // Consensus workers
    NetworkWorkers    int           `json:"networkWorkers"`    // Network operation workers
}
```

**Worker Pool Allocation:**
- **Transaction Pool**: 2x CPU cores for transaction processing
- **Validation Pool**: 1x CPU cores for block validation
- **Consensus Pool**: 0.5x CPU cores for consensus operations
- **State Pool**: 1x CPU cores for state operations
- **Network Pool**: 1x CPU cores for network operations

## 🏭 Miner Integration

### AI-Enhanced Mining

**Location**: `Core-Blockchain/node_src/miner/worker.go`

The miner integrates all systems for maximum performance:

```go
// Initialize parallel state processor for advanced parallel processing
parallelConfig := core.DefaultParallelProcessorConfig()
// Optimize for full CPU utilization while reserving resources for MobileLLM-R1-950M (optional)
cpuCores := runtime.NumCPU()
parallelConfig.TxBatchSize = 100000  // 1000x larger - match GPU's capability
parallelConfig.MaxTxConcurrency = cpuCores * 12  // Use 75% of CPU cores (leave 25% for AI)
parallelConfig.MaxMemoryUsage = 6 * 1024 * 1024 * 1024  // 6GB RAM (leave room for AI)
parallelConfig.MaxGoroutines = cpuCores * 24    // Aggressive parallelization
parallelConfig.StateWorkers = cpuCores * 2      // Full CPU state processing
parallelConfig.MaxValidationWorkers = cpuCores * 2 // Full CPU validation
```

### Massive Batch Processing

For transaction batches ≥ 100K transactions:

```go
// Use parallel processor for massive transaction batches (100K+ transactions)
if w.parallelProcessor != nil && totalTxCount >= 100000 {
    log.Info("Using parallel processor for massive transaction batch",
        "totalTxs", totalTxCount,
        "threshold", 100000,
        "parallelBatchSize", 100000)
    
    // Process ALL transactions with parallel processor
    receipts, logs, gasUsed, err := w.parallelProcessor.ProcessParallel(
        tempBlock,
        w.current.state.Copy(),
        *w.current.gasPool,
        vm.Config{},
    )
    
    if err != nil {
        log.Warn("Massive parallel processing failed, falling back to standard processing",
            "error", err,
            "txCount", totalTxCount)
    } else {
        // Parallel processing succeeded for massive batch
        log.Info("MASSIVE parallel processing completed successfully",
            "txCount", len(allTxs),
            "gasUsed", gasUsed,
            "duration", time.Since(start))
        
        // Skip standard processing since parallel processing handled everything
        w.updateSnapshot()
        return
    }
}
```

## 📊 Performance Metrics

### Verified Benchmark Results

**80,000 TPS Benchmark:**
```
+------------------+--------+-----+-------------+---------+
|       TIME       | NUMBER | TXS | GAS LIMIT   | GAS USED|
+------------------+--------+-----+-------------+---------+
| 2025-09-14 15:49:50 | 52672 |   0 | 50000000000 | 0.00%  |
| 2025-09-14 15:49:51 | 52673 |80000| 50000000000 | 0.34%  |
+------------------+--------+-----+-------------+---------+
| DURATION: 1.00 S | TOTAL TX | 80000 |    TPS    | 80000.00|
+------------------+--------+-----+-------------+---------+
```

**100,000 TPS Benchmark:**
```
+------------------+--------+--------+-------------+---------+
|       TIME       | NUMBER |  TXS   | GAS LIMIT   | GAS USED|
+------------------+--------+--------+-------------+---------+
| 2025-09-14 15:55:18 | 52692 |      0 | 50000000000 | 0.00%  |
| 2025-09-14 15:55:19 | 52693 | 100000 | 50000000000 | 0.42%  |
+------------------+--------+--------+-------------+---------+
| DURATION: 1.00 S | TOTAL TX | 100000 |    TPS    | 100000.00|
+------------------+--------+--------+-------------+---------+
```

### System Utilization

**Hardware Configuration:**
- **GPU**: RTX 4000 SFF Ada (20GB VRAM)
- **CPU**: 16+ cores (32+ threads)
- **RAM**: 64GB DDR4/DDR5
- **Storage**: NVMe SSD RAID

**Resource Allocation:**
- **GPU**: 18GB blockchain processing, 2GB AI model
- **CPU**: 75% blockchain processing, 25% AI decisions
- **RAM**: 48GB blockchain, 16GB system/AI

**Performance Targets:**
- **GPU Utilization**: 95-98% (AI-managed)
- **CPU Utilization**: 90-95% (AI-optimized)
- **Latency**: 15-25ms average
- **Block Time**: 1 second (50ms minimum)

## 🔧 Configuration Examples

### Production Configuration

```go
// AI Configuration
aiConfig := &ai.AIConfig{
    LLMEndpoint:         "http://localhost:8000/v1/chat/completions",
    LLMModel:           "facebook/MobileLLM-R1-950M",
    UpdateInterval:     250 * time.Millisecond,
    LearningRate:       0.25,
    ConfidenceThreshold: 0.65,
}

// GPU Configuration
gpuConfig := &gpu.GPUConfig{
    PreferredGPUType: gpu.GPUTypeCUDA,
    MaxBatchSize:     800000,
    MaxMemoryUsage:   18 * 1024 * 1024 * 1024, // 18GB (reserve ~2GB if running vLLM)
    HashWorkers:      80,
    SignatureWorkers: 80,
    TxWorkers:        80,
}

// Hybrid Configuration
hybridConfig := &hybrid.HybridConfig{
    EnableGPU:             true,
    GPUThreshold:          500,
    CPUGPURatio:           0.90, // 90% GPU, 10% CPU
    MaxGPUUtilization:     0.98, // Push RTX 4000 SFF Ada to limits
    AdaptiveLoadBalancing: true,
}

// Parallel Processing Configuration
parallelConfig := &core.ParallelProcessorConfig{
    MaxTxConcurrency:     runtime.NumCPU() * 12,
    TxBatchSize:          100000,
    EnablePipelining:     true,
    EnableTxBatching:     true,
    EnableBloomParallel:  true,
    AdaptiveScaling:      true,
}
```

### Development Configuration

```go
// Reduced resource usage for development
aiConfig := ai.DefaultAIConfig()
aiConfig.UpdateInterval = 1 * time.Second // Less frequent updates

gpuConfig := gpu.DefaultGPUConfig()
gpuConfig.MaxBatchSize = 10000 // Smaller batches
gpuConfig.HashWorkers = 8      // Fewer workers

hybridConfig := hybrid.DefaultHybridConfig()
hybridConfig.CPUGPURatio = 0.5 // Balanced CPU/GPU

parallelConfig := core.DefaultParallelProcessorConfig()
parallelConfig.MaxTxConcurrency = runtime.NumCPU() * 2 // Less aggressive
```

## 🚀 Getting Started

### Prerequisites

1. **Install vLLM and MobileLLM-R1-950M**:
   ```bash
   pip install vllm
   python -m vllm.entrypoints.openai.api_server \
     --model facebook/MobileLLM-R1-950M \
     --port 8000
   ```

2. **Install CUDA/OpenCL drivers** for RTX 4000 SFF Ada

3. **Configure system resources**:
   - 64GB RAM minimum
   - NVMe SSD storage
   - Proper cooling for sustained high performance

### Build and Run

```bash
# Build with GPU support
cd Core-Blockchain/node_src
make -f Makefile.gpu all

# Start the AI-optimized node
./node-start.sh

# Verify AI system is running
curl -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"splendor_getAIStatus","params":[],"id":1}' \
  http://localhost:8545
```

### Performance
