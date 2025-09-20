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
      chainId: 6546,
      // Let Hardhat automatically query gas price from the network
      gasPrice: "auto", // This will query eth_gasPrice RPC method
      
      // Alternative: Use EIP-1559 format (recommended)
      // maxFeePerGas: "auto",
      // maxPriorityFeePerGas: "auto",
      
      // For testing, you can set accounts if needed
      // accounts: ["0x..."] // Add private keys for testing
      
      // Network timeout settings
      timeout: 60000,
      
      // Gas settings
      gas: "auto",
      gasMultiplier: 1.2, // Add 20% buffer to gas estimates
    },
    
    // Local development network (if running locally)
    local: {
      url: "http://localhost:80",
      chainId: 6546,
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
