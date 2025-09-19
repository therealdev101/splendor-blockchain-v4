#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🔍 Checking vLLM AI Load Balancer Status${NC}"
echo "=================================================="

# Check if vLLM process is running
if [ -f "/tmp/vllm.pid" ]; then
    VLLM_PID=$(cat /tmp/vllm.pid)
    if ps -p $VLLM_PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM process is running (PID: $VLLM_PID)${NC}"
        
        # Check process details
        echo -e "\n${CYAN}Process Details:${NC}"
        ps -p $VLLM_PID -o pid,ppid,cmd,etime,pmem,pcpu
        
        # Check memory usage
        echo -e "\n${CYAN}Memory Usage:${NC}"
        ps -p $VLLM_PID -o pid,vsz,rss,pmem --no-headers | awk '{printf "VSZ: %d MB, RSS: %d MB, MEM: %s%%\n", $2/1024, $3/1024, $4}'
        
    else
        echo -e "${RED}❌ vLLM process not running (PID $VLLM_PID not found)${NC}"
    fi
else
    echo -e "${ORANGE}⚠️  vLLM PID file not found${NC}"
fi

# Check if vLLM API is responding
echo -e "\n${CYAN}API Status Check:${NC}"
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✅ vLLM API is responding${NC}"
    
    # Get model information
    echo -e "\n${CYAN}Available Models:${NC}"
    curl -s http://localhost:8000/v1/models | jq '.data[].id' 2>/dev/null || echo "Model info not available"
    
else
    echo -e "${RED}❌ vLLM API not responding on port 8000${NC}"
    
    # Check if port is in use
    echo -e "\n${CYAN}Port 8000 Status:${NC}"
    if netstat -tlnp | grep :8000; then
        echo -e "${ORANGE}Port 8000 is in use by another process${NC}"
    else
        echo -e "${ORANGE}Port 8000 is not in use${NC}"
    fi
fi

# Check vLLM logs
echo -e "\n${CYAN}Recent vLLM Logs (last 20 lines):${NC}"
if [ -f "/tmp/vllm.log" ]; then
    echo "----------------------------------------"
    tail -20 /tmp/vllm.log
    echo "----------------------------------------"
else
    echo -e "${ORANGE}vLLM log file not found${NC}"
fi

# Check GPU memory usage
echo -e "\n${CYAN}GPU Memory Status:${NC}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo -e "${ORANGE}nvidia-smi not available${NC}"
fi

# Test vLLM API with a simple request
echo -e "\n${CYAN}Testing vLLM API with simple request:${NC}"
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is accessible${NC}"
    
    # Try a simple completion request
    echo -e "\n${CYAN}Testing AI completion:${NC}"
    curl -s -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/MobileLLM-R1-950M",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0.1
        }' | jq '.choices[0].message.content' 2>/dev/null || echo "API test failed"
else
    echo -e "${RED}❌ API not accessible - still initializing${NC}"
    echo -e "${CYAN}This is normal for first-time model download${NC}"
fi

echo -e "\n${CYAN}💡 Troubleshooting Tips:${NC}"
echo "1. First-time vLLM startup can take 5-15 minutes to download MobileLLM model"
echo "2. Check /tmp/vllm.log for detailed progress"
echo "3. Monitor GPU memory usage with: watch -n 1 nvidia-smi"
echo "4. If stuck, restart vLLM: kill \$(cat /tmp/vllm.pid) && ./node-start.sh --validator"
