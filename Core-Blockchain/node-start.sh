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

#########################################################################
source ./.env
source ~/.bashrc
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
        tmux send-keys -t node$node_num "./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --port 30303 --ws --ws.addr $IP --ws.origins '*' --ws.port 8545 --http --http.port 80 --rpc.txfeecap 0 --http.corsdomain '*' --nat any --http.api db,eth,net,web3,personal,txpool,miner,debug --http.addr $IP --vmdebug --pprof --pprof.port 6060 --pprof.addr $IP --syncmode=full --gcmode=archive --cache=1024 --cache.database=512 --cache.trie=256 --cache.gc=256 --txpool.accountslots=5000 --txpool.globalslots=100000 --txpool.accountqueue=5000 --txpool.globalqueue=100000 --maxpeers=25 --ipcpath './chaindata/node$node_num/geth.ipc' console" Enter
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
        tmux send-keys -t node$node_num "./node_src/build/bin/geth --datadir ./chaindata/node$node_num --networkid $CHAINID --bootnodes $BOOTNODE --mine --port 30303 --nat extip:$IP --gpo.percentile 0 --gpo.maxprice 100 --gpo.ignoreprice 0 --miner.gaslimit 500000000000 --unlock 0 --password ./chaindata/node$node_num/pass.txt --syncmode=snap --gcmode=full --cache=1024 --cache.database=512 --cache.trie=256 --cache.gc=256 --txpool.accountslots=5000 --txpool.globalslots=100000 --txpool.accountqueue=5000 --txpool.globalqueue=100000 --maxpeers=25 --rpc.txfeecap=0 --http --http.addr 0.0.0.0 --http.api eth,net,web3,txpool,miner,debug --ws --ws.addr 0.0.0.0 --nat any --verbosity=3 console" Enter
      fi
    fi
  done
}

initializeGPU(){
  echo -e "\n${GREEN}+------------------ GPU Initialization -------------------+${NC}"
  
  # Check if GPU acceleration is enabled
  if [ "$ENABLE_GPU" = "true" ]; then
    echo -e "${CYAN}GPU acceleration enabled, checking hardware...${NC}"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
      echo -e "${GREEN}NVIDIA GPU detected:${NC}"
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
      
      # Check CUDA installation
      if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}CUDA toolkit found: $(nvcc --version | grep release)${NC}"
      else
        echo -e "${ORANGE}CUDA toolkit not found, GPU acceleration may not work${NC}"
      fi
    else
      echo -e "${ORANGE}NVIDIA GPU not detected, checking for OpenCL...${NC}"
    fi
    
    # Check for OpenCL
    if command -v clinfo &> /dev/null; then
      echo -e "${GREEN}OpenCL devices found:${NC}"
      clinfo --list 2>/dev/null || echo -e "${ORANGE}No OpenCL devices available${NC}"
    else
      echo -e "${ORANGE}OpenCL not available${NC}"
    fi
    
    # Check if GPU libraries are built
    if [ -f "./node_src/common/gpu/libcuda_kernels.so" ] || [ -f "./node_src/common/gpu/libopencl_kernels.so" ]; then
      echo -e "${GREEN}GPU libraries found, acceleration ready${NC}"
    else
      echo -e "${ORANGE}GPU libraries not found, building now...${NC}"
      cd ./node_src
      if make -f Makefile.gpu all 2>/dev/null; then
        echo -e "${GREEN}GPU libraries built successfully${NC}"
      else
        echo -e "${RED}GPU build failed, continuing with CPU-only mode${NC}"
      fi
      cd ../
    fi
    
    echo -e "${GREEN}GPU Configuration:${NC}"
    echo -e "  Max Batch Size: ${ORANGE}$GPU_MAX_BATCH_SIZE${NC}"
    echo -e "  GPU Memory: ${ORANGE}$(($GPU_MAX_MEMORY_USAGE / 1024 / 1024 / 1024))GB${NC}"
    echo -e "  GPU Workers: ${ORANGE}$GPU_TX_WORKERS${NC}"
    echo -e "  Throughput Target: ${ORANGE}$THROUGHPUT_TARGET TPS${NC}"
  else
    echo -e "${ORANGE}GPU acceleration disabled, using CPU-only mode${NC}"
  fi
}

finalize(){
  countNodes
  welcome
  initializeGPU
  
  if [ "$isRPC" = true ]; then
    echo -e "\n${GREEN}+------------------- Starting RPC -------------------+"
    startRpc
  fi

  if [ "$isValidator" = true ]; then
    echo -e "\n${GREEN}+------------------- Starting Validator -------------------+"
    startValidator
  fi

  echo -e "\n${GREEN}+------------------ Active Nodes -------------------+"
  tmux ls

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
