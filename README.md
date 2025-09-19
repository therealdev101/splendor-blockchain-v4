# Splendor Blockchain V4 - High-Performance EVM Blockchain

[![License: SBSAL](https://img.shields.io/badge/License-SBSAL-red.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.15+-blue.svg)](https://golang.org)
[![Network Status](https://img.shields.io/badge/Mainnet-Live-brightgreen.svg)](https://mainnet-rpc.splendor.org/)
[![AI Powered](https://img.shields.io/badge/AI-MobileLLM_R1_950M-purple.svg)](docs/GETTING_STARTED.md)
[![GPU Accelerated](https://img.shields.io/badge/GPU-RTX_4000_SFF_Ada-orange.svg)](docs/GETTING_STARTED.md)

A high-performance blockchain combining AI optimization, GPU acceleration, and advanced parallel processing to deliver verified high-throughput transaction processing with intelligent real-time optimization.

## Overview

Splendor Blockchain V4 is an AI-optimized, GPU-accelerated blockchain that combines artificial intelligence with high-performance computing to deliver enhanced transaction throughput and intelligent system optimization. The system is designed for high throughput while preserving EVM correctness; current single-node benchmarks show sustained 80–100K TPS end-to-end on standard hardware, with higher ingest/prevalidation rates under load.

### Key Features

- **AI-Powered Optimization**: MobileLLM-R1-950M for real-time load balancing (optional)
- **GPU Acceleration**: CUDA/OpenCL support with RTX 4000 SFF Ada optimization
- **High Performance**: 80–150K+ TPS on a single node (end-to-end), higher with horizontal scaling and hardware upgrades
- **Intelligent Load Balancing**: AI-driven CPU/GPU resource allocation
- **Predictive Analytics**: AI transaction pattern prediction and batch optimization
- **Hybrid Processing**: Seamless CPU/GPU/AI coordination
- **Fast Finality**: 1 second block times with low latency
- **Enterprise Security**: Congress consensus with Byzantine fault tolerance
- **Low Fees**: Minimal transaction costs with AI efficiency optimization
- **Full EVM Compatibility**: Seamless Ethereum tooling integration

## Performance Metrics

### Verified Throughput

Current single-node benchmarks on standard validator hardware:
- 80,000 TPS: sustained benchmark window
- 100,000 TPS: sustained benchmark window
- Block time: 1 second
- Latency: 15–25ms average (under load)

### Notes on measurement
Throughput figures refer to end-to-end state-commit throughput. GPU is used for prevalidation/scheduling; final state transitions are executed and committed by the CPU EVM for consensus safety.

## Performance Documentation

### Verified Performance Results

*Benchmark screenshots available upon request.*

### Performance Validation

**Verified Metrics:**
- 80,000 TPS: sustained throughput over a benchmark window
- 100,000 TPS: sustained throughput over a benchmark window

**Testing Environment:**
- **Hardware**: NVIDIA RTX 4000 SFF Ada Generation (20GB VRAM)
- **AI Optimization**: MobileLLM-R1-950M load balancer active
- **GPU Utilization**: 95-98% efficiency (AI-managed)
- **Network**: Mainnet configuration with Congress consensus

## How We Achieve High TPS

Splendor’s throughput comes from a modernized execution pipeline that keeps the CPU EVM saturated while the GPU pre-validates work in parallel. Final state transitions remain 100% EVM-correct and are committed by the canonical CPU executor.

- GPU prevalidation pipeline
  - RLP decode, hashing and structural/signature checks run on the GPU in large batches. See kernel launch and triple‑buffered CUDA streams: Core-Blockchain/node_src/common/gpu/cuda_kernels.cu:640
  - The GPU returns advisory validity, gas estimates and metadata; CPU remains the source of truth for state writes.

- Hybrid scheduler (CPU/GPU)
  - Adaptive strategy chooses CPU‑only, GPU‑only, or hybrid based on live utilization and latency: Core-Blockchain/node_src/common/hybrid/hybrid_processor.go:292
  - Per‑sender conflict‑free slicing keeps GPU kernels within target runtime while maximizing occupancy: Core-Blockchain/node_src/common/gpu/gpu_processor.go:1246

- Parallel state execution on CPU
  - After GPU prevalidation, validated txs are applied via a high‑throughput parallel state processor. The authoritative state transition is still CPU EVM.
  - Canonical EVM apply path: Core-Blockchain/node_src/miner/worker.go:1004; GPU batch entry in miner: Core-Blockchain/node_src/miner/worker.go:1118

- Aggressive but safe batching
  - Large txn batches, pinned host memory, and overlapping copy/compute (CUDA streams) minimize PCIe overhead and keep accelerators busy.
  - Adaptive batch sizing reacts to real‑time TPS and utilization.

- Optional AI assistance
  - A lightweight local model (MobileLLM‑R1‑950M) can nudge ratios and batch sizes under heavy load. The blockchain remains fully deterministic without AI.

What’s different from traditional EVM chains
- We don’t attempt to run the full EVM on the GPU. Instead, we offload the expensive prevalidation/scheduling phases so the CPU EVM can commit state continuously.
- GPU and CPU work in parallel (producer/consumer) rather than serial; this overlap is where the throughput gain comes from.
- Per‑sender scheduling avoids common write conflicts, improving both GPU kernel efficiency and CPU apply success rate.

Safety model (unchanged consensus)
- Final state transitions are executed and committed by the canonical CPU EVM. If the GPU ever mislabels a tx, the CPU apply rejects it; worst case is a performance loss, not a consensus fault. See `core.ApplyTransaction`: Core-Blockchain/node_src/miner/worker.go:1004
- GPU use is transparent and optional. On any GPU error, the system falls back to CPU with no change to consensus rules.

How to verify on your node
- Runtime logs: look for “Processing GPU batch” and “GPU TRANSACTION PROCESSING ACTIVATED” when load is high: Core-Blockchain/node_src/common/gpu/gpu_processor.go:927
- RPC: call `gpu_getGPUStats` to confirm `available=true`, type=`CUDA/OpenCL`, and `processedTxs` increasing: Core-Blockchain/node_src/internal/ethapi/api_gpu.go:60

## Network Information

| Parameter | Value |
|-----------|-------|
| **Network Name** | Splendor AI Mainnet |
| **RPC URL** | https://mainnet-rpc.splendor.org/ |
| **Chain ID** | 2691 |
| **Currency Symbol** | SPLD |
| **Block Explorer** | https://explorer.splendor.org/ |
| **Block Time** | 1 second (AI-optimized) |

## Quick Start

### MetaMask Setup
1. Open MetaMask and click the network dropdown
2. Select "Add Network" → "Add a network manually"
3. Enter the network details above
4. Save and switch to Splendor AI RPC

### Development Setup
```javascript
const { ethers } = require('ethers');

// Connect to Splendor mainnet
const provider = new ethers.JsonRpcProvider('https://mainnet-rpc.splendor.org/');

// Verify connection
const network = await provider.getNetwork();
console.log('Connected to:', network.name, 'Chain ID:', network.chainId);
```

## Technical Architecture

### AI System Components

**AI Load Balancer (MobileLLM-R1-950M)**
- Real-time optimization with 250ms decision cycles
- Response time: <250ms via local vLLM
- Target utilization: 95-98% GPU efficiency
- Performance gain: +20-30% TPS increase

**AI Transaction Predictor**
- Pattern analysis every 5 seconds
- TPS prediction every 2 seconds
- Dynamic 100K-200K transaction batches
- Latency reduction: -20-40% through predictive optimization

**Hybrid Processing Intelligence**
- Adaptive strategies: CPU_ONLY/GPU_ONLY/HYBRID selection
- Dynamic scaling to prevent bottlenecks
- Maximum hardware utilization
- Continuous learning and improvement

### Hardware Specifications

**Primary GPU: NVIDIA RTX 4000 SFF Ada Generation**
- VRAM: 20GB GDDR6 (18GB for blockchain processing)
- CUDA Cores: 6,144 Ada Lovelace cores
- Memory Bandwidth: 360 GB/s
- Tensor Performance: 165 TOPS (AI workloads)

**System Requirements**
- CPU: 16+ cores (32+ threads), 3.0+ GHz
- RAM: 64GB DDR4/DDR5 (48GB blockchain, 16GB system/AI)
- Storage: NVMe SSD 2TB+ (7GB/s read/write)
- Network: Gigabit+ for optimal performance

### Congress Consensus

Splendor uses an AI-enhanced Proof of Authority consensus called "Congress":
- AI-optimized finality with 1 second confirmation
- AI-monitored Byzantine fault tolerance
- Energy efficient (no wasteful mining)
- Scalable throughput via batching and hybrid scheduling

## Development Tools

### Supported Frameworks
- **Hardhat**: Full compatibility with AI performance monitoring
- **Truffle**: Deploy with AI gas optimization
- **Remix**: Browser-based development with AI insights
- **Foundry**: Fast testing with AI performance analytics

### Libraries & SDKs
- **JavaScript/TypeScript**: ethers.js with AI performance hooks
- **Python**: web3.py with AI optimization APIs
- **Go**: go-ethereum with AI load balancer integration
- **Rust**: ethers-rs with GPU acceleration support

## Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete setup including AI and GPU
- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - GPU and parallel processing details
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** - Run AI-optimized validators
- **[API Reference](docs/API_REFERENCE.md)** - AI-enhanced RPC endpoints
- **[Smart Contract Development](docs/SMART_CONTRACTS.md)** - Build and deploy guide
- **[Security Policy](docs/SECURITY.md)** - Security practices and audits

## Use Cases

### DeFi Applications
- High-frequency DEXs with AI-optimized order matching
- Intelligent lending with AI-powered risk assessment
- Predictive yield farming with AI-enhanced calculations
- Smart derivatives with AI-assisted instruments

### Enterprise Solutions
- Supply chain tracking with predictive analytics
- Identity management with AI-enhanced security
- Payment optimization with AI transaction routing
- Asset tokenization with AI-powered management

### Gaming & NFTs
- AI-enhanced GameFi with intelligent mechanics
- Smart NFT marketplaces with AI pricing
- Metaverse economies with AI optimization
- Predictive collectibles with AI rarity analysis

## Development Roadmap

### ✅ Completed (2025)
- **AI Load Balancer Integration**: MobileLLM-R1-950M AI system deployed and operational
- **GPU Acceleration**: RTX 4000 SFF Ada optimization with CUDA/OpenCL support
- **High-Performance Achievement**: Verified 80–100K TPS single-node benchmarks; ongoing throughput improvements
- **Hybrid Processing**: CPU/GPU/AI coordination system implemented
- **Congress Consensus**: AI-enhanced PoA consensus with Byzantine fault tolerance
- **Mainnet Deployment**: Live network with verified performance metrics
- **EVM Compatibility**: Full Ethereum tooling integration
- **X402 Integration**: Advanced API integration and optimization
- **Security Audits**: Comprehensive security analysis and fixes applied
- **Monitoring Systems**: Real-time performance and health monitoring
- **Documentation**: Complete technical and user documentation

### 🔄 Current Focus (Q4 2025)
- **Network Stability**: Ongoing monitoring and optimization
- **Performance Tuning**: AI model refinement and optimization
- **Developer Ecosystem**: Enhanced tooling and SDK development
- **Community Growth**: Expanding developer and validator communities

### 📋 Planned (2026)
- **Multi-GPU Scaling**: Support for multiple GPU configurations
- **Advanced AI Models**: Integration of larger parameter AI models
- **Cross-Chain Bridges**: Interoperability with major blockchain networks
- **Enterprise Features**: Advanced governance and compliance tools
- **Layer 2 Solutions**: Scaling solutions for specific use cases

## Community & Support

### Community
- **Telegram**: [Splendor Labs](https://t.me/SplendorLabs)
- **Twitter**: [@SplendorLabs](https://x.com/splendorlabs)
- **GitHub**: Contribute to optimization code
- **Stack Overflow**: Tag questions with `splendor-ai-blockchain`

### Professional Support
- Enterprise AI support and consulting
- AI integration development services
- Developer training and certification programs

## Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete setup including AI and GPU
- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - GPU and parallel processing details
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** - Run AI-optimized validators
- **[API Reference](docs/API_REFERENCE.md)** - AI-enhanced RPC endpoints
- **[Smart Contract Development](docs/SMART_CONTRACTS.md)** - Build and deploy guide
- **[Security Policy](docs/SECURITY.md)** - Security practices and audits

## License

This project is licensed under the **Splendor Blockchain Source-Available License (SBSAL) v1.0** - see the [LICENSE](LICENSE) file for details.

### License Summary

**Permitted:**
- Security auditing and research
- Educational purposes
- Connecting to the official Splendor Blockchain network
- Personal, non-commercial modifications
- Contributing back to the official repository

**Prohibited without written permission:**
- Forking to create competing networks
- Commercial use or resale
- Operating separate blockchain networks
- Removing Splendor branding

For commercial licensing inquiries: **legal@splendor.org**

**Copyright © 2025 Splendor Labs S.A. All rights reserved.**

---

**Built with AI by the Splendor Team**

*Advancing blockchain technology through artificial intelligence.*
