# Contributing to Splendor Blockchain

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA RTX 4000 SFF Ada (20GB VRAM) or equivalent
- **CPU**: 16+ cores, Intel i5-13500 or equivalent  
- **RAM**: 64GB DDR4 minimum
- **Storage**: 2TB+ NVMe SSD

### Setup
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/splendor-blockchain-v4.git
cd splendor-blockchain-v4

# Install dependencies
sudo apt install -y build-essential git golang-go nodejs npm python3-pip
pip3 install vllm torch transformers

# Build
cd Core-Blockchain/node_src
make geth
```

## 🎯 Contribution Areas

### AI System (MobileLLM-R1-950M)
- Improve load balancing decisions
- Enhance transaction prediction
- Optimize AI response times

### GPU Acceleration (RTX 4000 SFF Ada)
- CUDA/OpenCL kernel optimization
- Memory management improvements
- Multi-GPU support

### Parallel Processing
- Worker pool optimization
- Pipeline improvements
- Batch processing enhancements

## 📝 Guidelines

### Commits
```bash
git commit -m "feat(ai): improve MobileLLM decision accuracy"
git commit -m "perf(gpu): optimize CUDA kernels for RTX 4000"
git commit -m "fix(parallel): resolve worker pool race condition"
```

### Testing
```bash
# Run tests before submitting
cd Core-Blockchain/node_src
go test ./...
go test -bench=. ./common/ai/
go test -bench=. ./common/gpu/
```

## 🔄 Pull Request Process

1. Create feature branch
2. Make changes with tests
3. Verify 80K+ TPS performance
4. Submit PR with description
5. Address review feedback

## 📞 Support

- **GitHub Issues**: Bug reports and features
- **Telegram**: [Splendor Labs](https://t.me/SplendorLabs)
- **Documentation**: [Technical Architecture](TECHNICAL_ARCHITECTURE.md)

---

*Help us build the future of AI-powered blockchain technology!*
