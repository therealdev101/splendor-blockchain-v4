#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}Testing Splendor x402 Native Implementation${NC}\n"

# Test 1: Check if x402 API is available
echo -e "${CYAN}Test 1: Checking x402 API availability${NC}"
if curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"x402_supported","params":[],"id":1}' \
   http://localhost:80 | grep -q "result"; then
    echo -e "${GREEN}✅ x402 API is available${NC}"
else
    echo -e "${RED}❌ x402 API not available - make sure node is running${NC}"
    exit 1
fi

# Test 2: Test supported methods
echo -e "\n${CYAN}Test 2: Getting supported payment methods${NC}"
SUPPORTED=$(curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"x402_supported","params":[],"id":1}' \
   http://localhost:80)
echo "Response: $SUPPORTED"

# Test 3: Test middleware server
echo -e "\n${CYAN}Test 3: Testing x402 middleware${NC}"
cd x402-middleware
if npm test > /dev/null 2>&1 & then
    MIDDLEWARE_PID=$!
    sleep 3
    
    # Test free endpoint
    if curl -s http://localhost:3000/api/free | grep -q "free"; then
        echo -e "${GREEN}✅ Free endpoint working${NC}"
    else
        echo -e "${RED}❌ Free endpoint failed${NC}"
    fi
    
    # Test paid endpoint (should return 402)
    if curl -s -w "%{http_code}" http://localhost:3000/api/premium | grep -q "402"; then
        echo -e "${GREEN}✅ Paid endpoint correctly returns 402${NC}"
    else
        echo -e "${RED}❌ Paid endpoint not working correctly${NC}"
    fi
    
    kill $MIDDLEWARE_PID 2>/dev/null || true
else
    echo -e "${RED}❌ Middleware test failed${NC}"
fi

echo -e "\n${GREEN}x402 testing completed!${NC}"
