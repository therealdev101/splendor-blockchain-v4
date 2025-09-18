# Troubleshooting Guide

## 🔧 Common Issues and Solutions

### AI System Issues

#### TinyLlama 1.1B Not Responding
**Problem**: AI load balancer not making decisions
**Solution**:
```bash
# Check if vLLM server is running
curl -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"test"}]}' \
  http://localhost:8000/v1/chat/completions

# Start vLLM server if not running
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8000
```

#### GPU Not Detected
**Problem**: RTX 4000 SFF Ada not being used
**Solution**:
```bash
# Check GPU status
nvidia-smi

# Install CUDA drivers if needed
sudo apt update
sudo apt install nvidia-driver-470 nvidia-cuda-toolkit

# Verify OpenCL
clinfo
```

### Performance Issues

#### Low TPS (< 80K)
**Problem**: Not achieving verified 80K-100K TPS
**Causes & Solutions**:

1. **Insufficient GPU Memory**:
   ```bash
   # Check GPU memory usage
   nvidia-smi
   # Should show 18GB available for blockchain, 2GB for AI
   ```

2. **AI System Not Active**:
   ```bash
   # Check AI decision logs
   tail -f Core-Blockchain/logs/ai_decision.log
   # Should show decisions every 250ms
   ```

3. **Wrong Hardware Configuration**:
   - Verify RTX 4000 SFF Ada (20GB VRAM)
   - Ensure 64GB RAM minimum
   - Check 16+ CPU cores

#### High Latency (> 25ms)
**Problem**: Transaction latency too high
**Solution**:
```bash
# Check AI optimization status
curl -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"splendor_getAIStatus","params":[],"id":1}' \
  http://localhost:8545

# Should show AI active and GPU utilization 95-98%
```

#### Diagnose Block Commit Contention
**Goal**: Identify which phase of the block commit critical section is slow.
**Steps**:

1. **Enable metrics scraping** (or watch the `metric` log channel) while committing blocks.
2. **Track lock pressure**:
   - `chain/write/lock/wait`: time spent waiting to enter the critical section.
   - `chain/write/lock/held`: how long the mutex stayed locked per block.
3. **Measure commit sub-phases**:
   - `chain/write/asynccommit`: duration of the `StateDB.AsyncCommit` call.
   - `chain/write/trie/gc`: time spent flushing or GC-ing tries in the `afterCommit` callback.
   - `chain/write/batchwait`: how long the goroutine that writes block data to disk keeps the caller waiting.
   - `chain/head/update`: total time to update the canonical head and emit events.
4. **Correlate with logs**: each timer also emits a `metric` log (`asyncCommit`, `blockBatchWait`, `blockBatchWrite`, `canonicalUpdate`, `trieGC`) tagged with block number and hash so you can line up spikes with specific blocks.
5. **Act**: whichever metric dominates the `chain/write/lock/held` time is the bottleneck—optimize or scale that component first.

### Node Sync Issues

#### Node Won't Sync
**Problem**: Blockchain not synchronizing
**Solution**:
```bash
# Check peer connections
./Core-Blockchain/node_src/build/bin/geth attach --datadir ./Core-Blockchain/chaindata/node1
> net.peerCount
> eth.syncing

# Restart sync helper
pm2 restart sync-helper
```

#### Block Sealing Failed
**Problem**: Validator not producing blocks
**Solution**:
1. Ensure you've staked tokens at https://dashboard.splendor.org/
2. Wait for network to recognize your stake
3. Check validator logs for errors

### Installation Issues

#### Build Failures
**Problem**: Compilation errors during setup
**Solution**:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y build-essential git golang-go nodejs npm

# Clean and rebuild
cd Core-Blockchain/node_src
make clean
make geth
```

#### Permission Errors
**Problem**: Access denied during setup
**Solution**:
```bash
# Run as root for initial setup
sudo -i
# Then follow installation guide
```

### Network Issues

#### RPC Not Responding
**Problem**: Can't connect to RPC endpoint
**Solution**:
```bash
# Check if geth is running
ps aux | grep geth

# Test RPC connectivity
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
  http://localhost:80

# Check firewall
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 8545/tcp
sudo ufw allow 30303/tcp
```

#### Peer Connection Issues
**Problem**: No peers connecting
**Solution**:
```bash
# Check P2P port
sudo netstat -tulpn | grep :30303

# Verify bootnode connectivity
ping mainnet-rpc.splendor.org

# Check sync-helper status
pm2 status
pm2 logs sync-helper
```

### Hardware-Specific Issues

#### RTX 4000 SFF Ada Issues
**Problem**: GPU not performing optimally
**Solution**:
```bash
# Check GPU temperature and power
nvidia-smi -l 1

# Ensure proper cooling (should be < 80°C)
# Verify power supply can handle 70W+ GPU load
```

#### Memory Issues
**Problem**: Out of memory errors
**Solution**:
```bash
# Check memory usage
free -h
htop

# Ensure 64GB RAM minimum
# Verify swap is disabled for performance
sudo swapoff -a
```

### Development Issues

#### Smart Contract Deployment Fails
**Problem**: Contracts won't deploy
**Solution**:
```bash
# Check gas price and limit
# Verify account has sufficient SPLD
# Test on smaller contract first

# Use AI-optimized deployment
npx hardhat run scripts/deploy.js --network splendorAI --ai-optimize
```

#### MetaMask Connection Issues
**Problem**: Wallet won't connect
**Solution**:
1. Verify network configuration:
   - RPC URL: https://mainnet-rpc.splendor.org/
   - Chain ID: 2691
   - Currency: SPLD
2. Clear MetaMask cache
3. Try different RPC endpoint

## 📞 Getting Help

### Log Files
Important log locations:
```bash
# AI decision logs
tail -f Core-Blockchain/logs/ai_decision.log

# Node logs
tmux attach -t node1

# Sync helper logs
pm2 logs sync-helper

# System logs
journalctl -f
```

### Diagnostic Commands
```bash
# System health check
./Core-Blockchain/scripts/health-check.sh

# GPU monitoring
watch -n 1 nvidia-smi

# Network status
curl -s http://localhost:80 | jq .

# AI system status
curl -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"splendor_getAIStatus","params":[],"id":1}' \
  http://localhost:8545
```

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Telegram**: [Splendor Labs](https://t.me/SplendorLabs) - Real-time help
- **Twitter**: [@SplendorLabs](https://x.com/splendorlabs) - Updates
- **Documentation**: [Technical Architecture](TECHNICAL_ARCHITECTURE.md)

### Before Asking for Help
1. Check this troubleshooting guide
2. Review relevant documentation
3. Gather system information:
   ```bash
   # System specs
   lscpu | grep "Model name"
   free -h
   nvidia-smi --query-gpu=name,memory.total --format=csv
   
   # Software versions
   go version
   node --version
   git log --oneline -1
   ```
4. Include error messages and log outputs
5. Describe steps to reproduce the issue

---

*For additional help, consult the [Technical Architecture](TECHNICAL_ARCHITECTURE.md) guide or reach out to our community support channels.*
