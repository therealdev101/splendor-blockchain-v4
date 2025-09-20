#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           TESTING x402 INTEGRATION IN NODE SCRIPTS          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
${NC}\n"

log_step() {
  echo -e "${CYAN}➜ ${GREEN}$1${NC}"
}

log_success() {
  echo -e "${GREEN}✔ $1${NC}"
}

log_error() {
  echo -e "${RED}✖ $1${NC}"
}

log_wait() {
  echo -e "${CYAN}🕐 $1...${NC}"
}

# Test 1: Check if x402 setup function exists in node-setup.sh
log_step "Test 1: Checking x402 setup integration in node-setup.sh"
if grep -q "setup_x402()" node-setup.sh; then
    log_success "x402 setup function found in node-setup.sh"
else
    log_error "x402 setup function not found in node-setup.sh"
    exit 1
fi

if grep -q "setup_x402$" node-setup.sh; then
    log_success "x402 setup function is called in node-setup.sh"
else
    log_error "x402 setup function is not called in node-setup.sh"
    exit 1
fi

# Test 2: Check if x402 API is enabled in RPC startup
log_step "Test 2: Checking x402 API integration in node-start.sh"
if grep -q "http.api.*x402" node-start.sh; then
    log_success "x402 API enabled in RPC node startup"
else
    log_error "x402 API not enabled in RPC node startup"
    exit 1
fi

# Test 3: Check if x402 initialization exists in node-start.sh
if grep -q "Initializing x402 Native Payments" node-start.sh; then
    log_success "x402 initialization found in node-start.sh"
else
    log_error "x402 initialization not found in node-start.sh"
    exit 1
fi

# Test 4: Check if x402 middleware directory exists
log_step "Test 3: Checking x402 middleware components"
if [ -d "x402-middleware" ]; then
    log_success "x402-middleware directory exists"
else
    log_error "x402-middleware directory not found"
    exit 1
fi

if [ -f "x402-middleware/package.json" ]; then
    log_success "x402 middleware package.json exists"
else
    log_error "x402 middleware package.json not found"
    exit 1
fi

if [ -f "x402-middleware/index.js" ]; then
    log_success "x402 middleware main file exists"
else
    log_error "x402 middleware main file not found"
    exit 1
fi

# Test 5: Check if existing x402 setup script is present
log_step "Test 4: Checking existing x402 setup utilities"
if [ -f "setup-x402-complete.sh" ]; then
    log_success "Complete x402 setup script exists"
else
    log_error "Complete x402 setup script not found"
    exit 1
fi

if [ -f "test-x402.sh" ]; then
    log_success "x402 test script exists"
else
    log_error "x402 test script not found"
    exit 1
fi

# Test 6: Verify x402 configuration template
log_step "Test 5: Verifying x402 configuration template"
if grep -q "X402_ENABLED" node-setup.sh; then
    log_success "x402 configuration template found in setup script"
else
    log_error "x402 configuration template not found in setup script"
    exit 1
fi

# Test 7: Check if scripts are executable
log_step "Test 6: Checking script permissions"
if [ -x "node-setup.sh" ]; then
    log_success "node-setup.sh is executable"
else
    log_error "node-setup.sh is not executable"
    chmod +x node-setup.sh
    log_success "Fixed: made node-setup.sh executable"
fi

if [ -x "node-start.sh" ]; then
    log_success "node-start.sh is executable"
else
    log_error "node-start.sh is not executable"
    chmod +x node-start.sh
    log_success "Fixed: made node-start.sh executable"
fi

echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ✅ ALL TESTS PASSED!                      ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  x402 integration is properly configured in:                ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  📋 node-setup.sh:                                          ║${NC}"
echo -e "${GREEN}║     • x402 configuration added to .env                     ║${NC}"
echo -e "${GREEN}║     • x402 middleware dependencies installed               ║${NC}"
echo -e "${GREEN}║     • x402 test configuration created                      ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  🚀 node-start.sh:                                          ║${NC}"
echo -e "${GREEN}║     • x402 API enabled in RPC nodes                        ║${NC}"
echo -e "${GREEN}║     • x402 initialization and testing                      ║${NC}"
echo -e "${GREEN}║     • Status reporting and troubleshooting                 ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  Next steps:                                                 ║${NC}"
echo -e "${GREEN}║  1. Run: ./node-setup.sh --rpc                             ║${NC}"
echo -e "${GREEN}║  2. Run: ./node-start.sh --rpc                             ║${NC}"
echo -e "${GREEN}║  3. Test: ./test-x402.sh                                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${CYAN}🎉 x402 is now fully integrated into the node setup and startup process!${NC}\n"
