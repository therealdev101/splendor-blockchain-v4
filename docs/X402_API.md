X402 JSON‑RPC API

Overview
- Native zero‑fee x402 micro‑payments are exposed via the `x402` JSON‑RPC namespace.
- Settlement uses a typed envelope (EIP‑2718, type 0x50). Consensus (PoSA/Congress) verifies the inner signature and executes a balance transfer on‑chain with anti‑replay.
- Strict verification is enabled by default via `X402_STRICT_VERIFY=1`.

Canonical Message (strict mode)
- Format: `x402-payment:{from}:{to}:{valueHex}:{validAfter}:{validBefore}:{nonceHex}:{chainId}`
- Hash: EIP‑191 prefix (accounts.TextHash)
- Signature: 65 bytes (r||s||v) with v ∈ {27,28}

Types
- PaymentRequirements
  {
    "scheme": "exact",
    "network": "splendor",
    "maxAmountRequired": "0x...",
    "resource": "/api/path",
    "description": "optional",
    "mimeType": "application/json",
    "payTo": "0x...",
    "maxTimeoutSeconds": 300,
    "asset": "0x0000000000000000000000000000000000000000"
  }
- PaymentPayload
  {
    "x402Version": 1,
    "scheme": "exact",
    "network": "splendor",
    "payload": {
      "from": "0x...",
      "to": "0x...",
      "value": "0x...",
      "validAfter": 1710000000,
      "validBefore": 1710000300,
      "nonce": "0x<32‑byte‑hex>",
      "signature": "0x<65‑byte‑sig>"
    }
  }

Methods
1) x402_verify
- Params: [PaymentRequirements, PaymentPayload]
- Result: { "isValid": bool, "invalidReason"?: string, "payerAddress"?: string }
- Example:
  curl -s -X POST -H 'Content-Type: application/json' \
    --data '{
      "jsonrpc":"2.0","method":"x402_verify","params":[REQUIREMENTS_JSON, PAYLOAD_JSON],"id":1
    }' http://<host>:8545

2) x402_settle
- Params: [PaymentRequirements, PaymentPayload]
- Behavior: On success, constructs a typed 0x50 x402 envelope and submits it to the txpool. Congress executes it in consensus.
- Result: { "success": bool, "error"?: string, "txHash"?: "0x...", "networkId"?: string }
- Example:
  curl -s -X POST -H 'Content-Type: application/json' \
    --data '{
      "jsonrpc":"2.0","method":"x402_settle","params":[REQUIREMENTS_JSON, PAYLOAD_JSON],"id":1
    }' http://<host>:8545

End‑to‑End Smoke Test
1. Build canonical message
   MSG="x402-payment:0xFrom:0xTo:0x38d7ea4c68000:$(date +%s -d '-10 sec'):$(date +%s -d '+5 min'):0x<32‑byte‑nonce>:<chainId>"
2. Sign with your offline tool -> SIG (65 bytes)
3. Compose Requirements + Payload JSON (as above)
4. x402_verify → expect isValid=true
5. x402_settle → returns txHash
6. eth_getTransactionReceipt(txHash) → success; on‑chain balance updated

Operational Notes
- Strict mode: keep `X402_STRICT_VERIFY=1` (default) to enforce the canonical message and EIP‑191 hash.
- Anti‑replay: consensus stores keccak(from||nonce) under a reserved registry address; nonces cannot be reused.
- Zero‑fee: settlement transfers 100% of `value` to `payTo`. Treasury/fees are not applied.
- Security: expose x402 only on trusted RPC; rate‑limit and protect behind TLS/auth.

