# Hardhat Configuration Guide for Splendor Blockchain

## Overview

This comprehensive guide explains how to properly configure Hardhat to work with Splendor Blockchain, including gas price configuration, network setup, and best practices for development and deployment.

## The Problem

**Issue**: Hardhat was sending transactions with 1 gwei gas price while the network was suggesting 2 gwei, causing suboptimal transaction processing.

**Root Cause**: Hardhat was not querying the network's gas price oracle (`eth_gasPrice` RPC method) and was using hardcoded or default values instead.

## Gas Price Mechanism on Splendor Blockchain

### Current Configuration
- **Base Fee**: 1 gwei (from genesis configuration)
- **Suggested Gas Price**: 2 gwei (base fee + priority tip)
- **Gas Price Oracle**: Returns 2 gwei for optimal transaction inclusion

### EIP-1559 Implementation
Splendor Blockchain uses EIP-1559 with these parameters:
- `InitialBaseFee`: 1 gwei
- `MinimumBaseFee`: 1 gwei  
- `BaseFeeChangeDenominator`: 4 (2x faster fee adjustments)
- `ElasticityMultiplier`: 4 (4x gas limit elasticity)

## Hardhat Configuration

### Installation

First, install the required dependencies:

```bash
cd System-Contracts
npm install hardhat@^2.19.0 @nomiclabs/hardhat-ethers@^2.2.3 ethers@^5.7.2 --save-dev --legacy-peer-deps
```

### Updated Configuration

The `hardhat.config.js` has been updated to properly query gas prices:

```javascript
require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: {
    compilers: [
      {
        version: "0.8.17",
        settings: {
          optimizer: {
            enabled: true,
            runs: 200
          },
        },
      }
    ],
  },
  
  defaultNetwork: "hardhat",
  networks: {
    hardhat: {},
    
    // Splendor Blockchain Network Configuration
    splendor: {
      url: "https://mainnet-rpc.splendor.org/",
      chainId: 2691,
      gasPrice: "auto", // Queries eth_gasPrice from network
      gas: "auto",
      gasMultiplier: 1.2, // 20% buffer for gas estimates
      timeout: 60000,
    },
    
    // Local development network (if running locally)
    local: {
      url: "http://localhost:80",
      chainId: 2691,
      gasPrice: "auto",
      gas: "auto",
      gasMultiplier: 1.2,
    }
  },
  
  // Global gas settings
  gasReporter: {
    enabled: true,
    currency: 'USD',
  },
  
  // Mocha timeout for tests
  mocha: {
    timeout: 60000
  }
};
```

### Key Configuration Options

#### Option 1: Automatic Gas Price (Recommended)
```javascript
gasPrice: "auto" // Hardhat queries eth_gasPrice RPC method
```

#### Option 2: EIP-1559 Format (Alternative)
```javascript
maxFeePerGas: "auto",        // Maximum fee willing to pay
maxPriorityFeePerGas: "auto" // Tip for miners
```

#### Option 3: Fixed Gas Price (Not Recommended)
```javascript
gasPrice: 2000000000 // 2 gwei - hardcoded value
```

## Testing Gas Price Behavior

### Running the Test

A test script has been created to verify gas price behavior:

```bash
cd System-Contracts
npx hardhat run test-gas-price.js --network splendor
```

### Expected Output

The test will show:
- Current network gas price (should be 2 gwei)
- Base fee (should be 1 gwei)
- Suggested tip (should be 1 gwei)
- Gas estimation for transactions

### Sample Output
```
🔍 Testing Gas Price Behavior on Splendor Blockchain

📡 Network Information:
   Chain ID: 2691
   Network Name: unknown

⛽ Gas Price Information:
   Current Gas Price: 2000000000 wei
   Current Gas Price: 2.0 gwei

📦 Latest Block Information:
   Block Number: 87483
   Gas Used: 0
   Gas Limit: 20000000000
   Base Fee: 1000000000 wei
   Base Fee: 1.0 gwei
   Suggested Tip: 1000000000 wei
   Suggested Tip: 1.0 gwei

📊 Fee History:
   Base Fee (from history): 1.0 gwei
   Gas Used Ratio: 0

💡 Recommendations:
   ✅ Use gasPrice: "auto" in Hardhat config to query network gas price
   ✅ Current network suggests: 2.0 gwei
   ✅ For EIP-1559: maxFeePerGas >= 2.0 gwei, maxPriorityFeePerGas >= 1.0 gwei

✅ Gas price test completed successfully!
```

## Deployment Best Practices

### 1. Always Use Auto Gas Price
```javascript
// In your deployment scripts
const contract = await ContractFactory.deploy({
  // Don't specify gasPrice - let Hardhat query it automatically
});
```

### 2. For Manual Transactions
```javascript
// Query current gas price
const gasPrice = await ethers.provider.getGasPrice();

// Send transaction with current network gas price
const tx = await contract.someMethod({
  gasPrice: gasPrice
});
```

### 3. EIP-1559 Transactions
```javascript
// Get fee data from network
const feeData = await ethers.provider.getFeeData();

const tx = await contract.someMethod({
  maxFeePerGas: feeData.maxFeePerGas,
  maxPriorityFeePerGas: feeData.maxPriorityFeePerGas
});
```

## Troubleshooting

### Common Issues

#### Issue 1: Transactions Taking Too Long
**Cause**: Gas price too low
**Solution**: Ensure `gasPrice: "auto"` is set in network configuration

#### Issue 2: Transactions Failing
**Cause**: Gas price below base fee
**Solution**: Use automatic gas price querying or set minimum 2 gwei

#### Issue 3: High Transaction Costs
**Cause**: Gas price oracle suggesting high fees during network congestion
**Solution**: Monitor network usage and adjust timing of deployments

### Debugging Commands

```bash
# Check current gas price
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_gasPrice","params":[],"id":1}' \
  https://mainnet-rpc.splendor.org/

# Check latest block
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest", false],"id":1}' \
  https://mainnet-rpc.splendor.org/

# Check pending transactions
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"txpool_status","params":[],"id":1}' \
  https://mainnet-rpc.splendor.org/
```

## Network Configuration Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chain ID | 2691 | Splendor Blockchain identifier |
| RPC URL | https://mainnet-rpc.splendor.org/ | Main RPC endpoint |
| Base Fee | 1 gwei | Minimum fee burned per transaction |
| Suggested Gas Price | 2 gwei | Recommended total fee (base + tip) |
| Gas Limit | 20B | Maximum gas per block |

## Migration Checklist

- [ ] Update `hardhat.config.js` with new network configuration
- [ ] Set `gasPrice: "auto"` for automatic gas price querying
- [ ] Test gas price behavior with the provided test script
- [ ] Update deployment scripts to remove hardcoded gas prices
- [ ] Verify transactions are using correct gas prices
- [ ] Monitor transaction inclusion times and costs

## Additional Resources

- [EIP-1559 Specification](https://eips.ethereum.org/EIPS/eip-1559)
- [Hardhat Network Configuration](https://hardhat.org/config/#networks-configuration)
- [Splendor Blockchain RPC Documentation](../RPC_SETUP_GUIDE.md)

---

**Last Updated**: December 2024  
**Version**: 1.0
