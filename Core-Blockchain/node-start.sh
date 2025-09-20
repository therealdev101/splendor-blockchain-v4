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
  local i=1
  totalNodes=$(ls -l ./chaindata/ | grep -c ^d)
  while [[ $i -le $totalNodes ]]; do
    
    if [ -f "./chaindata/node$i/.rpc" ]; then  
      ((totalRpc += 1))
    else  
        if [ -f "./chaindata/node$i/.validator" ]; then
        ((totalValidator += 1))
        fi
    fi  
    
    ((i += 1))
  done 
}

startRpc(){
  i=$((totalValidator + 1))
  while [[ $i -le $totalNodes ]]; do
    

    if tmux has-session -t node$i > /dev/null 2>&1; then
        :
    else
        tmux new-session -d -s node$i
        tmux send-keys -t node$i " ./node_src/build/bin/geth --datadir ./chaindata/node$i --networkid $CHAINID --bootnodes $BOOTNODE --port 30303 --ws --ws.addr $IP --ws.origins '*' --ws.port 8545 --http --http.port 80 --rpc.txfeecap 0  --http.corsdomain '*' --nat 'any' --http.api db,eth,net,web3,personal,txpool,miner,debug,x402 --http.addr $IP --vmdebug --pprof --pprof.port 6060 --pprof.addr $IP --syncmode=full --gcmode=archive --cache 8192 --ipcpath './chaindata/node$i/geth.ipc' --txpool.accountslots=10000 --txpool.globalslots=200000 --txpool.accountqueue=10000 --txpool.globalqueue=100000 --txpool.lifetime=1h console" Enter
       
    fi


    ((i += 1))
  done 
}

startValidator(){
  i=1
  while [[ $i -le $totalValidator ]]; do
    
    if tmux has-session -t node$i > /dev/null 2>&1; then
        :
    else
        tmux new-session -d -s node$i
        tmux send-keys -t node$i "./node_src/build/bin/geth --datadir ./chaindata/node$i --networkid $CHAINID --bootnodes $BOOTNODE --mine --port 30303 --nat extip:$IP --gpo.percentile 0 --gpo.maxprice 100 --gpo.ignoreprice 0 --miner.gaslimit=300000000000 --unlock 0 --password ./chaindata/node$i/pass.txt --syncmode=full --gcmode=archive --txpool.accountslots=10000 --txpool.globalslots=200000 --txpool.accountqueue=10000 --txpool.globalqueue=100000 --txpool.lifetime=1h console" Enter
    fi

    ((i += 1))
  done 
}

finalize(){
  countNodes
  welcome
  
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

  # Initialize x402 native payments system
  echo -e "\n${GREEN}+------------------ Initializing x402 Native Payments -------------------+${NC}"
  
  # Check if x402 configuration exists
  if grep -q "X402_ENABLED=true" .env 2>/dev/null; then
    echo -e "${GREEN}✅ x402 configuration found${NC}"
    
    # Wait for node to be ready before testing x402 API
    echo -e "${CYAN}🕐 Waiting for node to be ready for x402 API...${NC}"
    sleep 5
    
    # Test x402 API availability
    if curl -s -X POST -H "Content-Type: application/json" \
       --data '{"jsonrpc":"2.0","method":"x402_supported","params":[],"id":1}' \
       http://localhost:80 2>/dev/null | grep -q "result"; then
      echo -e "${GREEN}✅ x402 API is available and responding${NC}"
    else
      echo -e "${ORANGE}⚠️  x402 API not yet available (node may still be starting)${NC}"
    fi
    
    # Check if x402 middleware dependencies are installed
    if [ -d "x402-middleware/node_modules" ]; then
      echo -e "${GREEN}✅ x402 middleware dependencies ready${NC}"
    else
      echo -e "${ORANGE}⚠️  x402 middleware dependencies not found${NC}"
      echo -e "${CYAN}   Run 'cd x402-middleware && npm install' to install${NC}"
    fi
    
    echo -e "${GREEN}🚀 x402 native payments system initialized!${NC}"
    echo -e "${CYAN}   • Test x402 API: curl -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"method\":\"x402_supported\",\"params\":[],\"id\":1}' http://localhost:80${NC}"
    echo -e "${CYAN}   • Test middleware: cd x402-middleware && npm test${NC}"
    echo -e "${CYAN}   • Full test suite: ./test-x402.sh${NC}"
  else
    echo -e "${ORANGE}⚠️  x402 configuration not found in .env${NC}"
    echo -e "${CYAN}   Run node setup again to configure x402${NC}"
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
