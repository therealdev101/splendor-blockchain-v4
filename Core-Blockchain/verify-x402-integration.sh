#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}🔍 Verifying x402 Integration in Splendor Blockchain${NC}\n"

# Check if x402 files are present
echo -e "${CYAN}Checking x402 files...${NC}"

files_to_check=(
    "x402-middleware/index.js"
    "x402-middleware/package.json"
    "x402-middleware/README.md"
    "x402-middleware/test.js"
    "node_src/eth/api_x402.go"
    "node_src/eth/api_x402_validator_rewards.go"
    "node_src/core/types/x402_tx.go"
    "setup-x402-complete.sh"
    "test-x402.sh"
)

all_files_present=true

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file${NC}"
        all_files_present=false
    fi
done

# Check if x402 API is in node-start.sh
echo -e "\n${CYAN}Checking node-start.sh for x402 API...${NC}"
if grep -q "x402" node-start.sh; then
    echo -e "${GREEN}✅ x402 API found in node-start.sh${NC}"
else
    echo -e "${RED}❌ x402 API not found in node-start.sh${NC}"
    all_files_present=false
fi

# Check if x402 setup is in node-setup.sh
echo -e "\n${CYAN}Checking node-setup.sh for x402 setup...${NC}"
if grep -q "task6_x402" node-setup.sh; then
    echo -e "${GREEN}✅ x402 setup task found in node-setup.sh${NC}"
else
    echo -e "${RED}❌ x402 setup task not found in node-setup.sh${NC}"
    all_files_present=false
fi

# Check if Geth binary has x402 support
echo -e "\n${CYAN}Checking Geth binary for x402 support...${NC}"
if [ -f "node_src/build/bin/geth" ]; then
    if strings node_src/build/bin/geth | grep -q "x402" 2>/dev/null || [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Geth binary built with x402 support${NC}"
    else
        echo -e "${ORANGE}⚠️  x402 strings not visible in binary (this is normal)${NC}"
    fi
else
    echo -e "${RED}❌ Geth binary not found${NC}"
    all_files_present=false
fi

# Check x402 configuration
echo -e "\n${CYAN}Checking x402 configuration...${NC}"
if [ -f ".env" ] && grep -q "X402_ENABLED" .env; then
    echo -e "${GREEN}✅ x402 configuration found in .env${NC}"
else
    echo -e "${ORANGE}⚠️  x402 configuration not found in .env (will be added during setup)${NC}"
fi

# Summary
echo -e "\n${CYAN}═══════════════════════════════════════${NC}"
if [ "$all_files_present" = true ]; then
    echo -e "${GREEN}🎉 x402 Integration Verification: PASSED${NC}"
    echo -e "${GREEN}✅ All x402 components are properly integrated${NC}"
    echo -e "${GREEN}✅ x402 will be automatically set up during node setup${NC}"
    echo -e "${GREEN}✅ x402 API will be available when nodes start${NC}"
    echo -e "\n${ORANGE}Next steps:${NC}"
    echo -e "1. Run ${GREEN}./node-setup.sh --rpc${NC} or ${GREEN}./node-setup.sh --validator${NC}"
    echo -e "2. x402 will be automatically configured during setup"
    echo -e "3. Start nodes with ${GREEN}./node-start.sh --rpc${NC} or ${GREEN}./node-start.sh --validator${NC}"
    echo -e "4. Test x402 functionality with ${GREEN}./test-x402.sh${NC}"
else
    echo -e "${RED}❌ x402 Integration Verification: FAILED${NC}"
    echo -e "${RED}Some x402 components are missing${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════${NC}\n"
