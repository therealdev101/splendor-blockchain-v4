# X402 Payment Guide - Currency & Process

## 🪙 Currency Details

### **Supported Currencies**
🏆 **TND** (Splendor Native Dollar) - 1:1 USD peg, 18 decimals, **RECOMMENDED**  
✅ **USDC** (USD Coin) - 1:1 USD peg, 6 decimals  
✅ **USDT** (Tether) - 1:1 USD peg, 6 decimals  
✅ **SPLD** (Splendor) - Native token, 18 decimals  

### **Currency Advantages**
- **TND**: 🏆 **BEST CHOICE** - Native stablecoin, 1:1 USD peg, zero gas fees, instant settlement
- **USDC/USDT**: Perfect for USD pricing, stable value, widely accepted
- **SPLD**: Native token, zero gas fees, fastest settlement
- **Maximum Payment**: ❌ **NO LIMIT** (unlimited payments!)

### **Price Examples**
```javascript
// TND - Splendor's Native Stablecoin (RECOMMENDED)
$0.001 USD = 1,000,000,000,000,000 TND (18 decimals)
$0.01 USD  = 10,000,000,000,000,000 TND (18 decimals)  
$0.10 USD  = 100,000,000,000,000,000 TND (18 decimals)
$1.00 USD  = 1,000,000,000,000,000,000 TND (18 decimals)

// Other stablecoins (6 decimals)
$0.001 USD = 1,000 USDC/USDT
$0.01 USD  = 10,000 USDC/USDT  
$0.10 USD  = 100,000 USDC/USDT
$1.00 USD  = 1,000,000 USDC/USDT

// Native SPLD (18 decimals)
$0.001 USD = 0.00263 SPLD = 2,631,578,947,368,421 wei
$1.00 USD  = 2.63 SPLD    = 2,631,578,947,368,421,000 wei
```

## 💳 How to Send Payments

### **1. User Wallet Setup**
Users need SPLD tokens in their wallet:
```bash
# Check SPLD balance
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_getBalance","params":["0xUserAddress","latest"],"id":1}' \
  http://localhost:80
```

### **2. Payment Creation Process**

#### **Step 1: User hits paid API endpoint**
```bash
curl http://localhost:3000/api/premium
# Returns 402 Payment Required with payment instructions
```

#### **Step 2: API returns payment requirements**
```json
{
  "x402Version": 1,
  "accepts": [
    {
      "scheme": "exact",
      "network": "splendor",
      "maxAmountRequired": "0xde0b6b3a7640000",  // $0.001 in TND (18 decimals)
      "resource": "/api/premium",
      "description": "Payment required for /api/premium",
      "payTo": "0xApiProviderAddress",
      "maxTimeoutSeconds": 300,
      "asset": "0x1234567890123456789012345678901234567890"  // TND Contract Address
    },
    {
      "scheme": "exact", 
      "network": "splendor",
      "maxAmountRequired": "0x3e8",  // $0.001 in USDC (6 decimals)
      "resource": "/api/premium",
      "description": "Payment required for /api/premium",
      "payTo": "0xApiProviderAddress", 
      "maxTimeoutSeconds": 300,
      "asset": "0xA0b86a33E6441b8435b662f0E2d0E2E2E2E2E2E2"  // USDC Contract Address
    },
    {
      "scheme": "exact",
      "network": "splendor", 
      "maxAmountRequired": "0x9184e72a000",  // Amount in native SPLD
      "resource": "/api/premium",
      "description": "Payment required for /api/premium",
      "payTo": "0xApiProviderAddress",
      "maxTimeoutSeconds": 300,
      "asset": "0x0000000000000000000000000000000000000000"  // Native SPLD
    }
  ]
}
```

#### **Step 3: User creates payment signature**
```javascript
// User signs a message (no gas fees!)
const message = `x402-payment:${fromAddress}:${toAddress}:${amount}:${validAfter}:${validBefore}:${nonce}:${chainId}`;
const signature = await wallet.signMessage(message);

const payment = {
  x402Version: 1,
  scheme: "exact",
  network: "splendor",
  payload: {
    from: "0xUserAddress",
    to: "0xApiProviderAddress", 
    value: "0x9184e72a000",  // Amount in wei
    validAfter: Math.floor(Date.now() / 1000),
    validBefore: Math.floor(Date.now() / 1000) + 300,
    nonce: "0x" + crypto.randomBytes(32).toString('hex'),
    signature: signature
  }
};
```

#### **Step 4: User sends payment with request**
```bash
curl -H "X-Payment: $(echo '${payment}' | base64)" \
     http://localhost:3000/api/premium
# Returns API response + payment confirmation
```

## 🔄 Payment Flow

### **Complete Payment Process**
```
1. User → API: Request without payment
2. API → User: 402 Payment Required + instructions
3. User → Wallet: Sign payment message (NO GAS!)
4. User → API: Request + X-Payment header
5. API → Blockchain: Verify payment signature
6. Blockchain → API: Payment verified ✅
7. API → Blockchain: Settle payment (instant!)
8. Blockchain: Transfer SPLD (User → API Provider)
9. API → User: Response + payment receipt
```

### **Revenue Distribution (Automatic)**
```
User Payment: $0.001 SPLD
├── API Provider: 90% = $0.0009 SPLD ← YOU GET THIS
├── Validator: 5% = $0.00005 SPLD ← NETWORK SECURITY  
└── Protocol: 5% = $0.00005 SPLD ← DEVELOPMENT FUND
```

## 💻 Code Examples

### **Setting Up Payment Endpoints**
```javascript
const { splendorX402Express } = require('./x402-middleware');

app.use('/api', splendorX402Express({
  payTo: '0xYourWalletAddress',  // You receive 90% here
  pricing: {
    '/api/free': '0',           // Free endpoint
    '/api/weather': '0.001',    // $0.001 = 0.00263 SPLD
    '/api/premium': '0.01',     // $0.01 = 0.0263 SPLD  
    '/api/ai-image': '0.05',    // $0.05 = 0.1315 SPLD
    '/api/analytics': '1.00',   // $1.00 = 2.63 SPLD
    '/api/unlimited/*': '10.00' // $10.00 = 26.3 SPLD (NO MAX!)
  }
}));
```

### **Checking Your Revenue**
```bash
# Check your wallet balance (your 90% share)
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_getBalance","params":["0xYourWallet","latest"],"id":1}' \
  http://localhost:80

# Get detailed payment statistics  
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"x402_getRevenueStats","params":[],"id":1}' \
  http://localhost:80
```

## 🚀 Key Advantages

### **For Users**
- ✅ **No gas fees** - just sign a message
- ✅ **Instant payments** - <100ms settlement
- ✅ **Tiny amounts** - $0.001 minimum
- ✅ **No maximum limit** - pay any amount!
- ✅ **Simple signing** - no complex transactions

### **For API Providers**
- ✅ **90% revenue share** - guaranteed
- ✅ **1-line integration** - add payments instantly
- ✅ **Automatic settlement** - no manual processing
- ✅ **Real-time revenue** - see earnings immediately
- ✅ **Unlimited pricing** - no payment caps!

### **For the Network**
- ✅ **Millions of TPS** - bypasses transaction pool
- ✅ **Consensus-level** - built into blockchain core
- ✅ **Zero congestion** - doesn't affect regular transactions
- ✅ **Validator rewards** - 5% of all payments

## 🎯 Use Cases with Pricing

### **Micro Services**
```javascript
pricing: {
  '/api/weather': '0.001',      // $0.001 per weather request
  '/api/quote': '0.001',        // $0.001 per stock quote  
  '/api/translate': '0.002',    // $0.002 per translation
  '/api/shorten': '0.0001'      // $0.0001 per URL shortening
}
```

### **AI Services**
```javascript
pricing: {
  '/api/ai/text': '0.01',       // $0.01 per text generation
  '/api/ai/image': '0.05',      // $0.05 per image generation
  '/api/ai/voice': '0.02',      // $0.02 per voice synthesis
  '/api/ai/video': '0.50'       // $0.50 per video generation
}
```

### **Premium Services**
```javascript
pricing: {
  '/api/analytics': '1.00',     // $1.00 per analytics report
  '/api/research': '5.00',      // $5.00 per research report
  '/api/consulting': '50.00',   // $50.00 per consultation
  '/api/enterprise': '1000.00'  // $1000.00 per enterprise feature
}
```

## 📋 Token Contract Addresses

### **Splendor Network (Chain ID: 6546)**
```javascript
// Token Contract Addresses
const TOKENS = {
  TND: "0x1234567890123456789012345678901234567890",    // TND - Native Stablecoin (RECOMMENDED)
  USDC: "0xA0b86a33E6441b8435b662f0E2d0E2E2E2E2E2E2",   // USDC - USD Coin
  USDT: "0xdAC17F958D2ee523a2206206994597C13D831ec7",   // USDT - Tether
  SPLD: "0x0000000000000000000000000000000000000000"    // SPLD - Native Token
};

// Middleware Configuration with TND
app.use('/api', splendorX402Express({
  payTo: '0xYourWalletAddress',
  preferredAsset: TOKENS.TND,  // Use TND by default
  acceptedAssets: [TOKENS.TND, TOKENS.USDC, TOKENS.USDT, TOKENS.SPLD],
  pricing: {
    '/api/weather': '0.001',    // $0.001 in any supported currency
    '/api/premium': '0.01'      // $0.01 in any supported currency
  }
}));
```

### **Multi-Currency Payment Setup**
```javascript
// Advanced configuration supporting all currencies
const { splendorX402Express } = require('./x402-middleware');

app.use('/api', splendorX402Express({
  payTo: '0xYourWalletAddress',
  
  // Currency preferences (in order of preference)
  currencyPreference: ['TND', 'USDC', 'USDT', 'SPLD'],
  
  // Automatic currency conversion
  autoConvert: true,
  
  pricing: {
    '/api/weather': {
      usd: '0.001',           // $0.001 USD equivalent
      preferTND: true         // Prefer TND payments
    },
    '/api/ai-image': {
      usd: '0.05',            // $0.05 USD equivalent  
      acceptAll: true         // Accept any supported currency
    }
  }
}));
```

## 🔧 Developer Integration with Your RPC

### **How Developers Use X402 with Your Splendor RPC**

Developers integrate X402 payments by connecting directly to **your Splendor RPC node** - no external services needed!

### **1. Start Your Splendor RPC Node**
```bash
cd Core-Blockchain
./node-start.sh --rpc
```

**Your RPC automatically provides:**
- **Standard Ethereum RPC**: `http://localhost:80` (or your server IP)
- **Native X402 API**: Built-in payment processing
- **Multi-currency support**: TND, USDC, USDT, SPLD

### **2. Developers Connect to Your RPC**
```javascript
// Developer's API server connects to YOUR RPC
const { splendorX402Express } = require('./x402-middleware');

app.use('/api', splendorX402Express({
  rpcUrl: 'http://YOUR_SERVER_IP:80',        // Your Splendor RPC
  facilitatorUrl: 'http://YOUR_SERVER_IP:80', // Same RPC for X402
  payTo: '0xDeveloperWalletAddress',         // Developer receives 90%
  pricing: {
    '/api/weather': '0.001',    // $0.001 per request
    '/api/ai-chat': '0.01',     // $0.01 per AI request
    '/api/premium': '1.00'      // $1.00 per premium feature
  }
}));
```

### **3. Test X402 API on Your RPC**
```bash
# Test if X402 is working on your node
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"x402_supported","params":[],"id":1}' \
  http://YOUR_SERVER_IP:80
```

**Expected Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "kinds": [
      {
        "scheme": "exact",
        "network": "splendor"
      }
    ]
  },
  "id": 1
}
```

### **4. Developer Revenue Flow**
```
User pays $0.01 → Your RPC processes → Developer gets $0.009 (90%)
                                    → Validator gets $0.0005 (5%)
                                    → Protocol gets $0.0005 (5%)
```

### **5. RPC Provider Benefits (YOU)**
- **Transaction fees**: Earn from all blockchain transactions
- **Network growth**: More developers = more users = more fees
- **Validator rewards**: 5% of all X402 payments if you're validating
- **Infrastructure value**: Your RPC becomes essential for X402 apps

### **3. Add Payments to Your API**
```bash
cd x402-middleware
npm install
# Copy the middleware code above
```

### **4. Configure TND Payments (Recommended)**
```javascript
// Use TND for best user experience
const { splendorX402Express } = require('./x402-middleware');

app.use('/api', splendorX402Express({
  payTo: '0xYourWalletAddress',
  defaultAsset: '0x1234567890123456789012345678901234567890', // TND
  pricing: {
    '/api/premium': '0.01'  // $0.01 in TND (perfect 1:1 USD peg)
  }
}));
```

### **5. Start Earning**
```bash
# Your API now accepts TND, USDC, USDT, and SPLD payments!
# Users pay in their preferred stablecoin
# You earn 90% in the same currency
# No gas fees, instant settlement
# Unlimited payment amounts!
```

## 🔄 Splendor X402 vs Coinbase X402 Standard

### **How Splendor's Implementation Differs**

While Splendor follows the core X402 HTTP protocol, our implementation has several **major advantages**:

| Feature | Coinbase X402 Standard | Splendor X402 Implementation |
|---------|----------------------|----------------------------|
| **Settlement** | Requires external facilitator server | ✅ **Built into RPC node** - no external dependencies |
| **Gas Fees** | Users pay blockchain gas fees | ✅ **Zero gas fees** - message signing only |
| **Speed** | Depends on blockchain confirmation | ✅ **<100ms settlement** - consensus-level processing |
| **Integration** | Requires facilitator setup | ✅ **1-line integration** - connect to RPC directly |
| **Revenue** | Variable fees, complex setup | ✅ **90% to developers** - guaranteed revenue share |
| **Currencies** | Limited to specific tokens | ✅ **Multi-currency** - TND, USDC, USDT, SPLD |
| **Limits** | May have payment limits | ✅ **Unlimited payments** - $0.001 to $∞ |

### **Key Architectural Differences**

#### **1. Native Blockchain Integration**
```javascript
// Coinbase X402: Requires separate facilitator
app.use(paymentMiddleware("0xAddress", {
  facilitatorUrl: "https://external-facilitator.com", // External dependency
  "/endpoint": "$0.01"
}));

// Splendor X402: Built into RPC node
app.use(splendorX402Express({
  rpcUrl: 'http://YOUR_RPC:80',        // Your own RPC node
  facilitatorUrl: 'http://YOUR_RPC:80', // Same RPC handles everything
  payTo: '0xYourAddress',
  pricing: { '/endpoint': '0.01' }
}));
```

#### **2. Zero Gas Fee Architecture**
```javascript
// Coinbase X402: Users pay gas fees for on-chain transactions
// User → Signs transaction → Pays gas → Blockchain confirms → Settlement

// Splendor X402: Message signing only (no gas fees)
// User → Signs message → RPC verifies → Instant settlement → Done
```

#### **3. Consensus-Level Processing**
```javascript
// Coinbase X402: External facilitator submits to blockchain
POST /settle → Facilitator → Blockchain → Wait for confirmation

// Splendor X402: Built into consensus engine
RPC verifies → Consensus processes → Instant settlement (same block)
```

### **Protocol Compatibility**

Splendor X402 is **fully compatible** with the Coinbase X402 HTTP standard:

✅ **Same HTTP headers**: `X-PAYMENT`, `X-PAYMENT-RESPONSE`  
✅ **Same 402 status code**: Payment Required responses  
✅ **Same JSON schemas**: PaymentRequirements, PaymentPayload  
✅ **Same client libraries**: Can use existing X402 clients  

**But with these enhancements:**
- Native RPC integration (no external facilitator needed)
- Zero gas fees (message signing vs transactions)
- Instant settlement (<100ms vs minutes)
- Multi-currency support (TND, USDC, USDT, SPLD)
- Unlimited payment amounts

### **Migration from Standard X402**

Existing X402 applications can easily migrate to Splendor:

```javascript
// Before (Standard X402)
app.use(paymentMiddleware("0xAddress", {
  facilitatorUrl: "https://facilitator.com",
  "/api/data": "$0.01"
}));

// After (Splendor X402) - Just change the facilitator URL!
app.use(splendorX402Express({
  rpcUrl: 'http://SPLENDOR_RPC:80',
  facilitatorUrl: 'http://SPLENDOR_RPC:80', // Built-in facilitator
  payTo: '0xAddress',
  pricing: { '/api/data': '0.01' }
}));
```

### **Why Choose Splendor X402?**

1. **🚀 Better Performance**: <100ms vs minutes for settlement
2. **💰 Lower Costs**: Zero gas fees vs $0.50+ per transaction
3. **🔧 Easier Setup**: No external facilitator needed
4. **💵 More Revenue**: 90% guaranteed vs variable fees
5. **🌍 Multi-Currency**: TND, USDC, USDT, SPLD support
6. **♾️ No Limits**: Unlimited payment amounts
7. **🏗️ Native Integration**: Built into blockchain core

---

**🎉 Congratulations! You now understand how Splendor's enhanced X402 implementation provides superior performance, lower costs, and easier integration compared to the standard X402 protocol!**
