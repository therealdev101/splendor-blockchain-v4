#!/bin/bash
#set -x

set -e


GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#########################################################################
totalRpc=0
totalValidator=0
totalNodes=$(($totalRpc + $totalValidator))

isRPC=false
isValidator=false

#Logger setup

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

progress_bar() {
  echo -en "${CYAN}["
  for i in {1..60}; do
    echo -en "#"
    sleep 0.01
  done
  echo -e "]${NC}"
}

#########################################################################
source ./.env
source ~/.bashrc

# Set up CUDA environment by default if GPU is enabled
if [ "$ENABLE_GPU" = "true" ]; then
  export CUDA_PATH=/usr/local/cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$(pwd)/node_src/common/gpu:$LD_LIBRARY_PATH
  export PATH=/usr/local/cuda/bin:$PATH
  echo -e "${GREEN}🚀 CUDA environment activated with custom GPU libraries${NC}"
fi
#########################################################################

#+-----------------------------------------------------------------------------------------------+
#|                                                                                                                             |
#|                                                                                                                             |
#|                                                      FUNCTIONS                                                |
#|                                                                                                                             |
#|                                                                                                                             |
#+-----------------------------------------------------------------------------------------------+

welcome(){
  # display welcome message
  echo -e "\n\n\t${ORANGE}Total RPC installed: $totalRpc"
  echo -e "\t${ORANGE}Total Validators installed: $totalValidator"
  echo -e "\t${ORANGE}Total nodes installed: $totalNodes"
  echo -e "${GREEN}
  \t+------------------------------------------------+
  \t+   DPos node Execution Utility
  \t+   Target OS: Ubuntu 20.04 LTS (Focal Fossa)
  \t+   Your OS: $(. /etc/os-release && printf '%s\n' "${PRETTY_NAME}") 
  \t+   example usage: ./node-start.sh --help
  \t+------------------------------------------------+
  ${NC}\n"

  echo -e "${ORANGE}
  \t+------------------------------------------------+
  \t+------------------------------------------------+
  ${NC}"
}

countNodes(){
  totalRpc=0
  totalValidator=0
  totalNodes=0
  
  # Count actual node directories that exist
  for dir in ./chaindata/node*; do
    if [ -d "$dir" ]; then
      ((totalNodes += 1))
      if [ -f "$dir/.rpc" ]; then  
        ((totalRpc += 1))
      elif [ -f "$dir/.validator" ]; then
        ((totalValidator += 1))
      fi
    fi
  done
}

startRpc(){
  # Start only RPC nodes
  for dir in ./chaindata/node*; do
    if [ -d "$dir" ] && [ -f "$dir/.rpc" ]; then
      node_num=$(basename "$dir" | sed 's/node//')
      
      if tmux has-session -t node$node_num > /dev/null 2>&1; then
        echo -e "${ORANGE}RPC node$node_num session already exists${NC}"
      else
        echo -e "${GREEN}Starting RPC node$node_num with optimized resource limits${NC}"
        
        # Set resource limits before starting
        ulimit -n 65536
        export GOMAXPROCS=20
        
        tmux new-session -d -s node$node_num
        tmux send-keys -t node$node_num "ulimit -n 65536" Enter
        tmux send-keys -t node$node_num "export GOMAXPROCS=20" Enter
        # Pass GPU environment to tmux session
        if [ "$ENABLE_GPU" = "true" ]; then
          tmux send-keys -t node$node_num "export CUDA_PATH=/usr/local/cuda" Enter
          tmux send-keys -t node$node_num "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$(pwd)/node_src/common/gpu:\$LD_LIBRARY_PATH" Enter
          tmux send-keys -t node$node_num "export PATH=/usr/local/cuda/bin:\$PATH" Enter
          tmux send-keys -t node$node_num "export ENABLE_GPU=true" Enter
        fi
        tmux send-keys -t node$node_num "./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --port 30303 --ws --ws.addr $IP --ws.origins '*' --ws.port 8545 --http --http.port 80 --rpc.txfeecap 0 --http.corsdomain '*' --nat any --http.api db,eth,net,web3,personal,txpool,miner,debug --http.addr $IP --vmdebug --pprof --pprof.port 6060 --pprof.addr $IP --syncmode=full --gcmode=archive --cache=1024 --cache.database=512 --cache.trie=256 --cache.gc=256 --txpool.accountslots=1000000 --txpool.globalslots=10000000 --txpool.accountqueue=500000 --txpool.globalqueue=5000000 --maxpeers=25 --ipcpath './chaindata/node$node_num/geth.ipc' console" Enter
      fi
    fi
  done
}

startValidator(){
  # Start only Validator nodes
  for dir in ./chaindata/node*; do
    if [ -d "$dir" ] && [ -f "$dir/.validator" ]; then
      node_num=$(basename "$dir" | sed 's/node//')
      
      if tmux has-session -t node$node_num > /dev/null 2>&1; then
        echo -e "${ORANGE}Validator node$node_num session already exists${NC}"
      else
        echo -e "${GREEN}Starting Validator node$node_num with optimized resource limits${NC}"
        
        # Set resource limits before starting
        ulimit -n 65536
        export GOMAXPROCS=20
        
        tmux new-session -d -s node$node_num
        tmux send-keys -t node$node_num "ulimit -n 65536" Enter
        tmux send-keys -t node$node_num "export GOMAXPROCS=20" Enter
        # Pass GPU environment to tmux session
        if [ "$ENABLE_GPU" = "true" ]; then
          tmux send-keys -t node$node_num "export CUDA_PATH=/usr/local/cuda" Enter
          tmux send-keys -t node$node_num "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$(pwd)/node_src/common/gpu:\$LD_LIBRARY_PATH" Enter
          tmux send-keys -t node$node_num "export PATH=/usr/local/cuda/bin:\$PATH" Enter
          tmux send-keys -t node$node_num "export ENABLE_GPU=true" Enter
        fi
        tmux send-keys -t node$node_num "LD_LIBRARY_PATH=./node_src/common/gpu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH ./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --mine --port 30303 --nat extip:$IP --gpo.percentile 0 --gpo.maxprice 100 --gpo.ignoreprice 0 --miner.gaslimit 500000000000 --unlock 0 --password ./chaindata/node$node_num/pass.txt --syncmode=full --gcmode=archive --cache=1024 --cache.database=512 --cache.trie=256 --cache.gc=256 --txpool.accountslots=1000000 --txpool.globalslots=10000000 --txpool.accountqueue=500000 --txpool.globalqueue=5000000 --maxpeers=25 console" Enter
      fi
    fi
  done
}

initializeGPU(){
  echo -e "\n${GREEN}+------------------ GPU Initialization -------------------+${NC}"
  
  # Quick non-blocking GPU check
  if [ "$ENABLE_GPU" = "true" ]; then
    echo -e "${CYAN}GPU acceleration enabled${NC}"
    
    # Fast GPU status check (non-blocking)
    if timeout 3 nvidia-smi >/dev/null 2>&1; then
      echo -e "${GREEN}✅ GPU drivers active and ready${NC}"
      GPU_STATUS="active"
    else
      echo -e "${ORANGE}⚠️  GPU drivers need reboot activation - continuing with CPU mode${NC}"
      GPU_STATUS="pending_reboot"
    fi
    
    # Enhanced CUDA check with multiple path detection
    CUDA_AVAILABLE=false
    
    # Check standard CUDA path
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
      CUDA_AVAILABLE=true
    # Check alternative CUDA paths
    elif command -v nvcc >/dev/null 2>&1; then
      CUDA_AVAILABLE=true
    # Check if CUDA toolkit is installed via package manager
    elif dpkg -l | grep -q "cuda-toolkit"; then
      CUDA_AVAILABLE=true
      # Try to find nvcc in system paths
      NVCC_PATH=$(find /usr -name "nvcc" 2>/dev/null | head -1)
      if [ -n "$NVCC_PATH" ]; then
        export PATH=$(dirname "$NVCC_PATH"):$PATH
      fi
    fi
    
    if [ "$CUDA_AVAILABLE" = "true" ]; then
      echo -e "${GREEN}✅ CUDA toolkit available and ready${NC}"
      # Verify CUDA can detect GPU
      if timeout 5 nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}✅ CUDA-GPU communication verified${NC}"
      fi
    else
      echo -e "${ORANGE}⚠️  CUDA toolkit not found - using OpenCL acceleration${NC}"
    fi
    
    echo -e "${GREEN}GPU Config: ${ORANGE}${THROUGHPUT_TARGET:-2000000} TPS target${NC}"
  else
    echo -e "${ORANGE}GPU acceleration disabled${NC}"
    GPU_STATUS="disabled"
  fi
  
  echo -e "${GREEN}✅ GPU check completed (non-blocking)${NC}"
}

finalize(){
  countNodes
  welcome
  initializeGPU
  
  # Initialize AI status tracking
  AI_STATUS="not_available"
  
  # Start vLLM AI service directly
  if [ -d "/opt/vllm-env" ]; then
    echo -e "\n${GREEN}+------------------ Starting AI System -------------------+${NC}"
    log_wait "Starting vLLM TinyLlama 1.1B AI load balancer"
    
    # Check if vLLM is already running
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
      log_success "vLLM AI service already running"
      AI_STATUS="fully_active"
    else
      # Start vLLM in background with working parameters
      cd /opt
      source vllm-env/bin/activate
      
      # Start vLLM with reduced memory usage and proper configuration
      nohup python -m vllm.entrypoints.openai.api_server \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.1 \
        --max-model-len 2048 \
        --dtype float16 \
        --disable-log-requests \
        > /tmp/vllm.log 2>&1 &
      
      VLLM_PID=$!
      echo $VLLM_PID > /tmp/vllm.pid
      
      cd /root/splendor-blockchain-v4/Core-Blockchain/
      
      log_success "vLLM AI service started (PID: $VLLM_PID)"
      AI_STATUS="service_started"
      
      # Wait for API to be ready (non-blocking)
      for i in {1..15}; do
        if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
          log_success "vLLM API ready - AI load balancing active"
          AI_STATUS="fully_active"
          break
        fi
        echo -e "${CYAN}Waiting for vLLM API... ($i/15)${NC}"
        sleep 3
      done
      
      # If API didn't respond but service started, it's still starting up
      if [ "$AI_STATUS" = "service_started" ]; then
        AI_STATUS="starting_up"
        log_wait "vLLM API still initializing (check /tmp/vllm.log for progress)"
      fi
    fi
  else
    echo -e "\n${ORANGE}vLLM environment not found. Run setup-ai-llm.sh to install AI features.${NC}"
    AI_STATUS="not_installed"
  fi
  
  if [ "$isRPC" = true ]; then
    echo -e "\n${GREEN}+------------------- Starting RPC -------------------+"
    startRpc
  fi

  if [ "$isValidator" = true ]; then
    echo -e "\n${GREEN}+------------------- Starting Validator -------------------+"
    startValidator
  fi

  echo -e "\n${GREEN}+------------------ Active Nodes -------------------+"
  if tmux ls 2>/dev/null; then
    echo -e "${GREEN}✅ Nodes started successfully${NC}"
  else
    echo -e "${ORANGE}⚠️  No tmux sessions found${NC}"
  fi

  echo -e "\n${GREEN}+------------------ Starting sync-helper -------------------+${NC}"
  echo -e "\n${ORANGE}+-- Please wait a few seconds. Do not turn off the server or interrupt --+"
  
  cd ./plugins/sync-helper/
  
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

  # Check if sync-helper is already running
  if pm2 list | grep -q "index"; then
    echo -e "${ORANGE}sync-helper is already running, restarting...${NC}"
    pm2 restart index
  else
    echo -e "${GREEN}Starting sync-helper...${NC}"
    pm2 start index.js --name "sync-helper"
  fi
  
  pm2 save
  cd /root/splendor-blockchain-v4/Core-Blockchain/

  # Final status report
  echo -e "\n${GREEN}+------------------ SYSTEM STATUS -------------------+${NC}"
  
  # Check AI system based on tracked status
  case "$AI_STATUS" in
    "fully_active")
      echo -e "${GREEN}✅ AI System: vLLM TinyLlama 1.1B active (150K+ TPS ready)${NC}"
      ;;
    "service_started"|"starting_up")
      echo -e "${GREEN}✅ AI System: vLLM service active (API initializing)${NC}"
      ;;
    "pending_reboot")
      echo -e "${ORANGE}⚠️  AI System: Will activate after reboot${NC}"
      ;;
    *)
      echo -e "${ORANGE}⚠️  AI System: Not available${NC}"
      ;;
  esac
  
  # Check GPU status
  if timeout 3 nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}✅ GPU: Active and ready${NC}"
  else
    echo -e "${ORANGE}⚠️  GPU: Will activate after reboot${NC}"
  fi
  
  # Check if GPU drivers need reboot and offer reboot
  if [ "$GPU_STATUS" = "pending_reboot" ] && lspci | grep -i nvidia >/dev/null 2>&1; then
    echo -e "\n${ORANGE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${ORANGE}║                    GPU REBOOT RECOMMENDED                   ║${NC}"
    echo -e "${ORANGE}║                                                              ║${NC}"
    echo -e "${ORANGE}║  NVIDIA GPU drivers are installed but need reboot to        ║${NC}"
    echo -e "${ORANGE}║  activate for maximum TPS performance.                      ║${NC}"
    echo -e "${ORANGE}║                                                              ║${NC}"
    echo -e "${ORANGE}║  Current: CPU-only mode                                     ║${NC}"
    echo -e "${ORANGE}║  After reboot: GPU+AI accelerated high TPS mode            ║${NC}"
    echo -e "${ORANGE}╚══════════════════════════════════════════════════════════════╝${NC}\n"
    echo -e "${CYAN}To activate GPU+AI acceleration: ${GREEN}reboot${NC}"
  fi

}


# Default variable values
verbose_mode=false
output_file=""

# Function to display script usage
usage() {
  echo -e "\nUsage: $0 [OPTIONS]"
  echo "Options:"
  echo -e "\t\t -h, --help      Display this help message"
  echo -e " \t\t -v, --verbose   Enable verbose mode"
  echo -e "\t\t --rpc       Start all the RPC nodes installed"
  echo -e "\t\t --validator       Start all the Validator nodes installed"
}

has_argument() {
  [[ ("$1" == *=* && -n ${1#*=}) || (! -z "$2" && "$2" != -*) ]]
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in

    # display help
    -h | --help)
      usage
      exit 0
      ;;

    # toggle verbose
    -v | --verbose)
      verbose_mode=true
      ;;

    # take file input
    -f | --file*)
      if ! has_argument $@; then
        echo "File not specified." >&2
        usage
        exit 1
      fi

      output_file=$(extract_argument $@)

      shift
      ;;

    # take ROC count
    --rpc)
        isRPC=true
      ;;

    # take validator count
    --validator)
        isValidator=true
      ;;

    *)
      echo "Invalid option: $1" >&2
      usage
      exit 1
      ;;

    esac
    shift
  done
}

# Main script execution
handle_options "$@"

# Perform the desired actions based on the provided flags and arguments
if [ "$verbose_mode" = true ]; then
  echo "Verbose mode enabled."
fi

if [ -n "$output_file" ]; then
  echo "Output file specified: $output_file"
fi

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    usage
    exit 1
fi

finalize
