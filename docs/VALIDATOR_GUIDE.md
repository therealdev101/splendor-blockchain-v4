# Splendor AI-Optimized Validator Installation Guide

## 🖥️ **ACTUAL TESTING HARDWARE REQUIREMENTS**

**Verified Configuration (September 2025):**
- **GPU**: NVIDIA RTX 4000 SFF Ada Generation (20GB VRAM)
- **CPU**: 16+ cores (32+ threads), Intel i5-13500 or equivalent
- **RAM**: 64GB DDR4 (minimum for AI + GPU + blockchain processing)
- **Storage**: 2TB+ NVMe SSD (enterprise grade recommended)
- **Network**: Gigabit+ connection for optimal performance
- **OS**: Ubuntu 20.04 LTS or 22.04 LTS

**AI System Requirements:**
- **TinyLlama 1.1B**: 2GB VRAM reserved for AI load balancer
- **vLLM Server**: Running on localhost:8000
- **GPU Memory**: 18GB available for blockchain processing
- **CPU Allocation**: 75% blockchain, 25% AI decisions

**Performance Targets:**
- **Verified TPS**: 80K-100K+ (tested and confirmed)
- **Block Time**: 1 second (AI-optimized)
- **GPU Utilization**: 95-98% (AI-managed)
- **Latency**: 15-25ms average

**IMPORTANT: You must create a fresh wallet and have the Private Key - it will be needed for the validator setup!**

### 1. Switch to root
```bash
sudo -i
```

### 2. Update and upgrade packages
```bash
apt update && apt upgrade -y
```

### 3. Install required packages
```bash
apt install -y git tar curl wget tmux
```

### 4. Reboot the server
```bash
reboot
```

### 5. After reboot, switch to root again
```bash
sudo -i
```

### 6. Clone the Splendor blockchain repository
```bash
git clone https://github.com/Splendor-Protocol/splendor-blockchain-v4.git
```

### 7. Move into the Core-Blockchain directory
```bash
cd splendor-blockchain-v4/Core-Blockchain
```

### 8. Run the validator setup
```bash
./node-setup.sh --validator 1
```

### 9. Start the validator
```bash
./node-start.sh --validator
```

### 10. Attach to the validator session
```bash
tmux attach -t node1
```

### 11. Wait for Block Sealing to Fail
Wait until the output shows 'Block Sealing Failed' multiple times.
**!!WARNING DO NOT DETACH HERE!!**

### 12. Stake Tokens
Then go stake tokens at [https://dashboard.splendor.org/](https://dashboard.splendor.org/)

### 13. Wait for Mined Block
Wait until you see a hammer icon or "mined potential block" in the output.

### 14. Detach from Session
To detach from the session (and leave the node running):
Press `CTRL + b`, release both keys, then press `d`.
