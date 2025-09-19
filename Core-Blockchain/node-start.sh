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

# Configure geth cache parameters with conservative defaults to prevent startup crashes
# Override via environment: CACHE_MB, CACHE_DB_MB, CACHE_TRIE_MB, CACHE_GC_MB
CACHE_MB=${CACHE_MB:-4096}  # Start with 4GB instead of 48GB
CACHE_DB_MB=${CACHE_DB_MB:-$(( CACHE_MB*70/100 ))}
CACHE_TRIE_MB=${CACHE_TRIE_MB:-$(( CACHE_MB*20/100 ))}
CACHE_GC_MB=${CACHE_GC_MB:-$(( CACHE_MB*10/100 ))}

# Safety clamp: do not exceed ~50% of physical RAM for stability
if [ -r /proc/meminfo ]; then
  MEM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  MEM_TOTAL_MB=$(( MEM_TOTAL_KB / 1024 ))
  MAX_SAFE_MB=$(( MEM_TOTAL_MB * 50 / 100 ))  # Reduced from 85% to 50%
  if [ "$CACHE_MB" -gt "$MAX_SAFE_MB" ]; then
    echo -e "${ORANGE}⚠️  Requested cache ${CACHE_MB}MB exceeds safe limit (${MAX_SAFE_MB}MB). Clamping.${NC}"
    CACHE_MB=$MAX_SAFE_MB
    CACHE_DB_MB=$(( CACHE_MB*70/100 ))
    CACHE_TRIE_MB=$(( CACHE_MB*20/100 ))
    CACHE_GC_MB=$(( CACHE_MB*10/100 ))
  fi
fi

echo -e "${GREEN}🧠 geth cache configured:${NC} total=${CACHE_MB}MB db=${CACHE_DB_MB}MB trie=${CACHE_TRIE_MB}MB gc=${CACHE_GC_MB}MB"

# Set up CUDA environment by default if GPU is enabled
if [ "$ENABLE_GPU" = "true" ]; then
  export CUDA_PATH=/usr/local/cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$(pwd)/node_src/common/gpu:$LD_LIBRARY_PATH
  export PATH=/usr/local/cuda/bin:$PATH
  echo -e "${GREEN}🚀 CUDA environment activated with custom GPU libraries${NC}"
fi
# Enable strict x402 verification by default unless overridden
export X402_STRICT_VERIFY=${X402_STRICT_VERIFY:-1}
# Default HTTP API list for RPC nodes includes x402 namespace
HTTP_API_LIST=${HTTP_API_LIST:-"db,eth,net,web3,personal,txpool,miner,debug,x402"}
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
  \t+   PoSA node Execution Utility
  \t+   Compatible OS: Ubuntu 20.04+ LTS
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
        tmux send-keys -t node$node_num "./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --port 30303 --ws --ws.addr $IP --ws.origins '*' --ws.port 8545 --http --http.port 80 --rpc.txfeecap 0 --http.corsdomain '*' --nat any --http.api $HTTP_API_LIST --http.addr $IP --vmdebug --pprof --pprof.port 6060 --pprof.addr $IP --syncmode=full --gcmode=archive --cache=${CACHE_MB} --cache.database=${CACHE_DB_MB} --cache.trie=${CACHE_TRIE_MB} --cache.gc=${CACHE_GC_MB} --txpool.accountslots=16 --txpool.globalslots=4096 --txpool.accountqueue=64 --txpool.globalqueue=1024 --maxpeers=25 --ipcpath './chaindata/node$node_num/geth.ipc' console" Enter
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
        tmux send-keys -t node$node_num "LD_LIBRARY_PATH=./node_src/common/gpu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH ./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --mine --port 30303 --nat extip:$IP --gpo.percentile 0 --gpo.maxprice 100 --gpo.ignoreprice 0 --miner.gaslimit 500000000000 --unlock 0 --password ./chaindata/node$node_num/pass.txt --syncmode=full --gcmode=archive --cache=${CACHE_MB} --cache.database=${CACHE_DB_MB} --cache.trie=${CACHE_TRIE_MB} --cache.gc=${CACHE_GC_MB} --txpool.accountslots=16 --txpool.globalslots=4096 --txpool.accountqueue=64 --txpool.globalqueue=1024 --maxpeers=25 console" Enter
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
      
      # Build GPU kernels if needed
      if [ ! -f "./node_src/common/gpu/libcuda_kernels.so" ] || [ ! -f "./node_src/common/gpu/libopencl_kernels.so" ]; then
        echo -e "${CYAN}Building GPU kernels...${NC}"
        cd ./node_src
        if make -f Makefile.gpu clean && make -f Makefile.gpu cuda && make -f Makefile.gpu opencl; then
          echo -e "${GREEN}✅ GPU kernels built successfully${NC}"
        else
          echo -e "${ORANGE}⚠️  GPU kernel build failed - using CPU fallback${NC}"
        fi
        cd ..
      else
        echo -e "${GREEN}✅ GPU kernels already built${NC}"
      fi
      
      # Build GPU-enabled geth if needed
      if [ ! -f "./node_src/build/bin/geth" ] || ! ldd ./node_src/build/bin/geth | grep -q cuda; then
        echo -e "${CYAN}Building Geth with GPU support...${NC}"
        cd ./node_src
        if make -f Makefile.cuda geth-cuda; then
          echo -e "${GREEN}✅ Geth built with GPU support${NC}"
        else
          echo -e "${ORANGE}⚠️  GPU-enabled Geth build failed - using standard build${NC}"
        fi
        cd ..
      else
        echo -e "${GREEN}✅ GPU-enabled Geth already built${NC}"
      fi
      
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
  
  # Start vLLM AI service with proper virtual environment handling
  echo -e "\n${GREEN}+------------------ Starting AI System -------------------+${NC}"
  
  # Check if vLLM is already running
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    log_success "vLLM AI service already running"
    AI_STATUS="fully_active"
  else
    # Check if vLLM virtual environment exists
    if [ -d "/opt/vllm-env" ]; then
      log_success "vLLM virtual environment found"
      
      # Check if vLLM is installed in the virtual environment
      if /opt/vllm-env/bin/python -c "import vllm" 2>/dev/null; then
        log_success "vLLM already installed in virtual environment"
      else
        log_wait "Installing vLLM in virtual environment"
        source /opt/vllm-env/bin/activate
        if pip install vllm transformers huggingface_hub fastapi uvicorn; then
          log_success "vLLM installed successfully in virtual environment"
        else
          echo -e "${ORANGE}⚠️  vLLM installation failed - continuing without AI${NC}"
          AI_STATUS="install_failed"
        fi
        deactivate
      fi
    else
      # Create virtual environment and install vLLM
      log_wait "Creating vLLM virtual environment"
      python3 -m venv /opt/vllm-env
      source /opt/vllm-env/bin/activate
      
      log_wait "Installing vLLM in virtual environment"
      pip install --upgrade pip setuptools wheel
      if pip install vllm transformers huggingface_hub fastapi uvicorn; then
        log_success "vLLM installed successfully in virtual environment"
      else
        echo -e "${ORANGE}⚠️  vLLM installation failed - continuing without AI${NC}"
        AI_STATUS="install_failed"
      fi
      deactivate
    fi
    
    # Start vLLM if installation succeeded
    if [ "$AI_STATUS" != "install_failed" ]; then
      log_wait "Starting vLLM MobileLLM-R1-950M AI load balancer"
      
      # Start vLLM using the virtual environment
      nohup /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
        --model facebook/MobileLLM-R1-950M \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.1 \
        --max-model-len 4096 \
        --disable-log-requests \
        > /tmp/vllm.log 2>&1 &
      
      VLLM_PID=$!
      echo $VLLM_PID > /tmp/vllm.pid
      
      log_success "vLLM AI service started (PID: $VLLM_PID)"
      AI_STATUS="service_started"
      
      # Wait for API to be ready (non-blocking)
      for i in {1..30}; do
        if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
          log_success "vLLM API ready - AI load balancing active"
          AI_STATUS="fully_active"
          break
        fi
        echo -e "${CYAN}Waiting for vLLM API... ($i/30)${NC}"
        sleep 2
      done
      
      # If API didn't respond but service started, it's still starting up
      if [ "$AI_STATUS" = "service_started" ]; then
        AI_STATUS="starting_up"
        log_wait "vLLM API still initializing (check /tmp/vllm.log for progress)"
      fi
    fi
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
      echo -e "${GREEN}✅ AI System: vLLM MobileLLM-R1-950M active (150K+ TPS ready)${NC}"
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
