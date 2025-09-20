# Splendor Validator Installation Guide

**IMPORTANT: You must create a fresh wallet and have the Private Key - it will be needed for the validator setup!**

## Validator Tiers Overview

Splendor Blockchain uses a tiered validator system that automatically assigns validators to different tiers based on their total staking amount (including delegated stakes). Higher tiers provide greater rewards and governance participation.

### Validator Tier Structure

| Tier | Stake Required | Benefits |
|------|----------------|----------|
| **Bronze** | 3,947 SPLD | Entry-level validation, basic rewards |
| **Silver** | 39,474 SPLD | Enhanced rewards, increased voting weight |
| **Gold** | 394,737 SPLD | Premium rewards, governance participation |
| **Platinum** | 3,947,368 SPLD | Elite tier, maximum rewards, priority governance |

### Tier Benefits

**Bronze Validators:**
- Basic block rewards and gas fee distribution
- Entry-level participation in network consensus
- Standard slashing penalties for misbehavior

**Silver Validators:**
- 25% higher rewards than Bronze tier
- Enhanced network priority for block production
- Reduced slashing penalties

**Gold Validators:**
- 50% higher rewards than Bronze tier
- Governance proposal submission rights
- Significant reduction in slashing penalties
- Priority in validator selection algorithms

**Platinum Validators:**
- 100% higher rewards than Bronze tier
- Maximum governance voting weight
- Minimal slashing penalties
- Highest priority in all network operations
- Access to exclusive validator features

### Automatic Tier Management

- **Dynamic Updates**: Validator tiers are automatically updated when staking amounts change
- **Real-time Calculation**: Tier assignment includes both self-stake and delegated stakes
- **Immediate Effect**: Tier changes take effect at the next epoch (50 blocks)
- **No Manual Action**: No need to manually upgrade or downgrade tiers

## Hardware Requirements by Tier

### Bronze/Silver Validators (Entry Level)
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or equivalent)
- **RAM**: 32 GB
- **Storage**: 1 TB NVMe SSD
- **Network**: 100 Mbps dedicated bandwidth

### Gold Validators (Professional)
- **CPU**: 16-32 cores (Intel Xeon/AMD EPYC)
- **RAM**: 64-128 GB
- **Storage**: 2 TB NVMe SSD (RAID-0 recommended)
- **Network**: 1 Gbps dedicated bandwidth

### Platinum Validators (Enterprise)
- **CPU**: 64-128 cores (High-end Intel Xeon/AMD EPYC)
- **RAM**: 256-512 GB
- **Storage**: 4+ TB NVMe SSD (RAID-0, 7+ GB/s write speed)
- **Network**: 10-25 Gbps dedicated bandwidth

**Note**: Higher-tier validators are expected to maintain superior infrastructure to support the network's high-performance requirements.

## Validator Configuration

Validators are configured with high-performance settings to handle massive transaction volumes:

### Gas Configuration
- **Network Gas Limit**: 20,000,000,000 (20B gas per block)
- **Validator Miner Gas Limit**: 20,000,000,000 (`--miner.gaslimit 20000000000`)
- **Gas Price Oracle**: Optimized settings (`--gpo.percentile 0 --gpo.maxprice 100 --gpo.ignoreprice 0`)

### Transaction Pool Settings (High-Performance)
- **Account Slots**: 10,000 pending transactions per account (`--txpool.accountslots=10000`)
- **Global Slots**: 200,000 total pending transactions (`--txpool.globalslots=200000`)
- **Account Queue**: 10,000 queued transactions per account (`--txpool.accountqueue=10000`)
- **Global Queue**: 100,000 total queued transactions (`--txpool.globalqueue=100000`)

**Total Capacity**: 300,000+ simultaneous transactions (200k pending + 100k queued)

### Performance Optimizations
- **Cache Settings**: 512MB total cache (`--cache=512`)
  - Database cache: 256MB (`--cache.database=256`)
  - Trie cache: 128MB (`--cache.trie=128`)
  - Garbage collection cache: 128MB (`--cache.gc=128`)
- **Sync Mode**: Snap sync for faster initial synchronization (`--syncmode=snap`)
- **GC Mode**: Full mode for optimal performance (`--gcmode=full`)

## Installation Steps

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
