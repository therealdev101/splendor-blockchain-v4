#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color
CYAN='\033[0;36m'
BASE_DIR='/root/splendor-blockchain-v4'

# Flag: skip validator account setup (task8)
NOPK=false
# Recognize --nopk early (non-destructive parsing so existing getopts/case blocks still work)
for __arg in "$@"; do
  if [ "$__arg" = "--nopk" ]; then
    NOPK=true
  fi
done

#########################################################################
totalRpc=0
totalValidator=0
totalNodes=$(($totalRpc + $totalValidator))

#########################################################################

#+-----------------------------------------------------------------------------------------------+
#|                                                                                                                              |
#|                                                      FUNCTIONS                                                |
#|                                                                                                                              |     
#+------------------------------------------------------------------------------------------------+

task1(){
  # update and upgrade the server TASK 1
  log_wait "Updating system packages" && progress_bar
  apt update && apt upgrade -y
  log_success "System packages updated"
}

task2(){
  # installing build-essential TASK 2
  log_wait "Getting dependies" && progress_bar
  
  # Fix bzip2/libbz2-1.0 version conflict on Ubuntu 24.04
  if grep -q "24.04" /etc/os-release; then
    log_wait "Fixing bzip2 dependency conflicts for Ubuntu 24.04"
    apt install libbz2-1.0=1.0.8-5.1 -y --allow-downgrades 2>/dev/null || true
    apt install bzip2 -y 2>/dev/null || true
  fi
  
  apt -y install build-essential tree
  log_success "Done"
}

task3(){
  # getting golang TASK 3
  log_wait "Getting golang" && progress_bar
  cd ./tmp && wget "https://go.dev/dl/go1.17.3.linux-amd64.tar.gz"
  log_success "Done"
}

task4(){
  # setting up golang TASK 4
  log_wait "Installing golang and setting up autostart" && progress_bar
  rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.3.linux-amd64.tar.gz

  LINE='PATH=$PATH:/usr/local/go/bin'
  if grep -Fxq "$LINE" /etc/profile
  then
    # code if found
    echo -e "${ORANGE}golang path is already added"
  else
    # code if not found
    echo -e '\nPATH=$PATH:/usr/local/go/bin' >>/etc/profile
  fi

  echo -e '\nsource ~/.bashrc' >>/etc/profile

  nodePath=$BASE_DIR/Core-Blockchain
  
  if [[ $totalValidator -gt 0 ]]; then
  LINE="cd $nodePath"
  if grep -Fq "$LINE" /etc/profile; then
    log_wait "Validator: working directory already in profile"
  else
    echo -e "\ncd $nodePath" >> /etc/profile
  fi

  LINE="bash $nodePath/node-start.sh --validator"
  if grep -Fq "$LINE" /etc/profile; then
    log_wait "Validator: autostart already in profile"
  else
    echo -e "\nbash $nodePath/node-start.sh --validator" >> /etc/profile
  fi
fi

if [[ $totalRpc -gt 0 ]]; then
  LINE="cd $nodePath"
  if grep -Fq "$LINE" /etc/profile; then
    log_wait "RPC: working directory already in profile"
  else
    echo -e "\ncd $nodePath" >> /etc/profile
  fi

  LINE="bash $nodePath/node-start.sh --rpc"
  if grep -Fq "$LINE" /etc/profile; then
    log_wait "RPC: autostart already in profile"
  else
    echo -e "\nbash $nodePath/node-start.sh --rpc" >> /etc/profile
  fi
fi


  

  export PATH=$PATH:/usr/local/go/bin
  go env -w GO111MODULE=off
  log_success "Done"
  
}

task5(){
  # set proper group and permissions TASK 5
  log_wait "Setting up Permissions" && progress_bar
  ls -all
  cd ../
  ls -all
  chown -R root:root ./
  chmod a+x ./node-start.sh
  log_success "Done"
}

task6(){
  # do make all TASK 6
  log_wait "Building backend" && progress_bar
  cd node_src
  make all
  log_success "Done"
}

install_gpu_dependencies(){
  # Install GPU dependencies automatically TASK 6A
  log_wait "Installing GPU dependencies for high-performance RPC" && progress_bar
  
  # Update package lists
  apt update
  
  # Install NVIDIA drivers
  log_wait "Installing NVIDIA drivers"
  apt install -y nvidia-driver-470 nvidia-utils-470
  
  # Install CUDA Toolkit
  log_wait "Installing CUDA Toolkit 11.8"
  
  # Ensure tmp directory exists
  mkdir -p ./tmp
  cd ./tmp
  
  if [ ! -f "cuda_11.8.0_520.61.05_linux.run" ]; then
    log_wait "Downloading CUDA Toolkit (this may take several minutes)..."
    wget --progress=bar:force https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    log_success "CUDA Toolkit download completed"
  else
    log_success "CUDA Toolkit installer already exists"
  fi
  
  log_wait "Installing CUDA Toolkit (this may take 5-10 minutes)..."
  chmod +x cuda_11.8.0_520.61.05_linux.run
  
  # Run with verbose output instead of silent
  if sh cuda_11.8.0_520.61.05_linux.run --toolkit --no-opengl-libs --override --verbose 2>&1 | tee cuda_install.log; then
    log_success "CUDA Toolkit installation completed"
  else
    log_error "CUDA Toolkit installation failed - check cuda_install.log for details"
    tail -20 cuda_install.log
  fi
  
  # Return to parent directory
  cd ../
  
  # Install OpenCL support
  log_wait "Installing OpenCL support"
  apt install -y opencl-headers ocl-icd-opencl-dev nvidia-opencl-dev mesa-opencl-icd intel-opencl-icd
  
  # Install additional build tools
  apt install -y cmake clinfo
  
  # Set up CUDA environment
  export CUDA_PATH=/usr/local/cuda
  export PATH=$PATH:$CUDA_PATH/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
  
  # Add CUDA to system profile
  echo 'export CUDA_PATH=/usr/local/cuda' >> /etc/profile
  echo 'export PATH=$PATH:$CUDA_PATH/bin' >> /etc/profile
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64' >> /etc/profile
  
  log_success "GPU dependencies installed successfully"
}

configure_gpu_environment(){
  # Configure GPU environment for optimal performance TASK 6B
  log_wait "Configuring GPU environment for high-performance RPC" && progress_bar
  
  # Create GPU configuration in .env file
  cat >> ./.env << EOF

# GPU Acceleration Configuration for High-Performance RPC
ENABLE_GPU=true
PREFERRED_GPU_TYPE=CUDA
GPU_MAX_BATCH_SIZE=10000
GPU_MAX_MEMORY_USAGE=4294967296
GPU_HASH_WORKERS=8
GPU_SIGNATURE_WORKERS=8
GPU_TX_WORKERS=8
GPU_ENABLE_PIPELINING=true

# Hybrid Processing Configuration
ENABLE_HYBRID_PROCESSING=true
GPU_THRESHOLD=1000
CPU_GPU_RATIO=0.7
ADAPTIVE_LOAD_BALANCING=true
PERFORMANCE_MONITORING=true
MAX_CPU_UTILIZATION=0.85
MAX_GPU_UTILIZATION=0.90
THROUGHPUT_TARGET=1000000

# Memory Management
MAX_MEMORY_USAGE=17179869184
GPU_MEMORY_RESERVATION=2147483648

# Performance Optimization
GPU_DEVICE_COUNT=1
GPU_LOAD_BALANCE_STRATEGY=round_robin
EOF
  
  log_success "GPU environment configured for 1M+ TPS target"
}

task6_gpu(){
  # Build GPU acceleration components TASK 6 GPU
  log_wait "Setting up complete GPU acceleration for high-performance RPC" && progress_bar
  
  # Install GPU dependencies automatically
  install_gpu_dependencies
  
  # Configure GPU environment
  configure_gpu_environment
  
  # Check if GPU is available
  if nvidia-smi >/dev/null 2>&1; then
    log_success "NVIDIA GPU detected successfully"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
  else
    log_wait "GPU not detected or drivers need reboot - GPU features will activate after reboot"
  fi
  
  # Build GPU components
  log_wait "Building CUDA and OpenCL kernels for maximum performance"
  cd node_src
  
  # Set environment for build
  export CUDA_PATH=/usr/local/cuda
  export PATH=$PATH:$CUDA_PATH/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
  
  # Build GPU components
  if make -f Makefile.gpu check-deps 2>/dev/null; then
    make -f Makefile.gpu all
    log_success "GPU acceleration components built successfully - Ready for 1M+ TPS"
  else
    log_wait "GPU build will complete after system reboot (driver activation required)"
    echo -e "${ORANGE}System reboot recommended to activate GPU drivers${NC}"
  fi
  
  cd ../
  log_success "GPU RPC setup completed - High-performance mode ready"
}

task7(){
  # setting up directories and structure for node/s TASK 7
  log_wait "Setting up directories for node instances" && progress_bar

  cd ../

  i=1
  while [[ $i -le $totalNodes ]]; do
    mkdir ./chaindata/node$i
    ((i += 1))
  done

  tree ./chaindata
  log_success "Done"
}

task8(){
  # Skip when --nopk is provided
if [ "${NOPK}" = "true" ]; then
  echo "[--nopk] Skipping task8 (validator key import/creation)"
  return 0
fi
log_wait "Setting up Validator Accounts" && progress_bar

  i=1
  while [[ $i -le $totalValidator ]]; do
    echo -e "\n\n${GREEN}+-----------------------------------------------------------------------------------------------------+${NC}"
    echo -e "${ORANGE}Setting up Validator $i${NC}"
    echo -e "${GREEN}Choose how you want to import/create account for validator $i:${NC}"
    echo -e "${ORANGE}1) Create a new account"
    echo -e "2) Import via Private Key"
    echo -e "3) Import via JSON keystore file${NC}"
    read -p "Enter your choice (1/2/3): " choice

    # Validator's node directory
    validator_dir="./chaindata/node$i"

    mkdir -p "$validator_dir"

    case $choice in
      1)
        read -s -p "Enter password to create new validator account: " password
        echo "$password" > "$validator_dir/pass.txt"
        echo
        ./node_src/build/bin/geth --datadir "$validator_dir" account new --password "$validator_dir/pass.txt"
        ;;

      2)
        read -s -p "Enter password to secure the imported account: " password
        echo "$password" > "$validator_dir/pass.txt"
        echo
        read -p "Enter the private key (hex, without 0x): " pk

        # Convert to lowercase, remove any whitespace
        pk_cleaned=$(echo "$pk" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

        if [[ ${#pk_cleaned} -ne 64 ]]; then
          log_error "Invalid private key length. Skipping validator $i."
        else
          echo "$pk_cleaned" | ./node_src/build/bin/geth --datadir "$validator_dir" account import --password "$validator_dir/pass.txt" /dev/stdin
        fi
        ;;

      3)
        read -p "Enter the full path to your JSON keystore file: " json_path
        if [[ ! -f "$json_path" ]]; then
          log_error "File not found: $json_path. Skipping validator $i."
        else
          read -s -p "Enter password to decrypt the keystore file: " password
          echo "$password" > "$validator_dir/pass.txt"
          echo
          cp "$json_path" "$validator_dir/keystore/"
          echo -e "${GREEN}Keystore file copied to $validator_dir/keystore/${NC}"
        fi
        ;;

      *)
        log_error "Invalid option. Skipping validator $i."
        ;;
    esac

    ((i += 1))
  done

  log_success "[TASK 8 PASSED]"
}


labelNodes(){
  i=1
  while [[ $i -le $totalValidator ]]; do
    touch ./chaindata/node$i/.validator
    ((i += 1))
  done 

  i=$((totalValidator + 1))
  while [[ $i -le $totalNodes ]]; do
    touch ./chaindata/node$i/.rpc
    ((i += 1))
  done 
}

displayStatus(){
  echo -e "\n${GREEN}🚀 ALL SET!${NC}"
  echo -e "${ORANGE}➜ To start the node, run:${NC} ${GREEN}./node-start.sh${NC}\n"
}


displayWelcome(){
  # display welcome message
  echo -e "\n\n\t${ORANGE}Total RPC to be created: $totalRpc"
  echo -e "\t${ORANGE}Total Validators to be created: $totalValidator"
  echo -e "\t${ORANGE}Total nodes to be created: $totalNodes"
  echo -e "${GREEN}
  \t+------------------------------------------------+
  \t+   DPos node installation Wizard
  \t+   Target OS: Ubuntu 20.04 LTS (Focal Fossa)
  \t+   Your OS: $(. /etc/os-release && printf '%s\n' "${PRETTY_NAME}") 
  \t+------------------------------------------------+
  ${NC}\n"

  echo -e "${ORANGE}
  \t+------------------------------------------------+
  \t+------------------------------------------------+
  ${NC}"
}

doUpdate(){
  echo -e "${GREEN}
  \t+------------------------------------------------+
  \t+       UPDATING TO LATEST    
  \t+------------------------------------------------+
  ${NC}"
  git pull
}

createRpc(){
  task1
  task2
  task3
  task4
  task5
  task6
  task6_gpu
  task7
  i=$((totalValidator + 1))
  while [[ $i -le $totalNodes ]]; do
    read -p "Enter Virtual Host(example: rpc.yourdomain.tld) without https/http: " vhost
    echo -e "\nVHOST=$vhost" >> ./.env
    ./node_src/build/bin/geth --datadir ./chaindata/node$i init ./genesis.json
    ((i += 1))
  done

}

createValidator(){
  if [[ $totalValidator -gt 0 && "$NOPK" != "true" ]]; then
      if [ "${NOPK}" != "true" ]; then task8; fi
  fi
   i=1
  while [[ $i -le $totalValidator ]]; do
    ./node_src/build/bin/geth --datadir ./chaindata/node$i init ./genesis.json
    ((i += 1))
  done
}

install_nvm() {
  # Check if nvm is installed
  if ! command -v nvm &> /dev/null; then
    echo "Installing NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

    # Source NVM scripts for the current session
    export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" # This loads nvm bash_completion

    # Add NVM initialization to shell startup file
    if [ -n "$BASH_VERSION" ]; then
      SHELL_PROFILE="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
      SHELL_PROFILE="$HOME/.zshrc"
    fi

    if ! grep -q 'export NVM_DIR="$HOME/.nvm"' "$SHELL_PROFILE"; then
      echo 'export NVM_DIR="$HOME/.nvm"' >> "$SHELL_PROFILE"
      echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> "$SHELL_PROFILE"
      echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> "$SHELL_PROFILE"
    fi
  else
    echo "NVM is already installed."
  fi

  # Source NVM scripts (if not sourced already)
  export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
  [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" # This loads nvm bash_completion

  # Install Node.js version 21.7.1 using nvm
  echo "Installing Node.js version 21.7.1..."
  nvm install 21.7.1

  # Use the installed Node.js version
  nvm use 21.7.1

  # Verify the installation
  node_version=$(node --version)
  if [[ $node_version == v21.7.1 ]]; then
    echo "Node.js version 21.7.1 installed successfully: $node_version"
  else
    echo "There was an issue installing Node.js version 21.7.1."
  fi

  source ~/.bashrc

  npm install --global yarn
  npm install --global pm2

  source ~/.bashrc
}

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


finalize(){
  displayWelcome
  createRpc
  createValidator
  labelNodes

  # resource paths
  nodePath=$BASE_DIR/Core-Blockchain
  ipcPath=$nodePath/chaindata/node1/geth.ipc
  chaindataPath=$nodePath/chaindata/node1/geth
  snapshotName=$nodePath/chaindata.tar.gz

  # added gitkeep
  # echo -e "\n\n\t${ORANGE}Removing existing chaindata, if any${NC}"
  
  # rm -rf $chaindataPath/chaindata

  # echo -e "\n\n\t${GREEN}Now importing the snapshot"
  # wget https://snapshots.splendor.org/chaindata.tar.gz

  # Create the directory if it does not exist
  # if [ ! -d "$chaindataPath" ]; then
  #   mkdir -p $chaindataPath
  # fi

  # # Extract archive to the correct directory
  # tar -xvf $snapshotName -C $chaindataPath --strip-components=1

  # Set proper permissions
  # echo -e "\n\n\t${GREEN}Setting directory permissions${NC}"
  # chown -R root:root $nodePath/chaindata
  # chmod -R 755 $nodePath/chaindata

  # echo -e "\n\n\tImport is done, now configuring sync-helper${NC}"
  # sleep 3
  cd $nodePath
  

  install_nvm
  cd $nodePath/plugins/sync-helper
  yarn
  cd $nodePath


  displayStatus
}


#########################################################################

#+-----------------------------------------------------------------------------------------------+
#|                                                                                                                             |
#|                                                                                                                             |
#|                                                      UTILITY                                                        |
#|                                                                                                                             |
#|                                                                                                                             |
#+-----------------------------------------------------------------------------------------------+


# Default variable values
verbose_mode=false
output_file=""

# Function to display script usage
usage() {
  echo -e "\nUsage: $0 [OPTIONS]"
  echo "Options:"
  echo -e "\t\t -h, --help      Display this help message"
  echo -e " \t\t -v, --verbose   Enable verbose mode"
  echo -e "\t\t --rpc      Specify to create RPC node"
  echo -e "\t\t --validator  <whole number>     Specify number of validator node to create"
  echo -e "		 --nopk     Skip validator account import/creation (skip task8)"
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
      totalRpc=1
      totalNodes=$(($totalRpc + $totalValidator))
      ;;

    # take validator count
    --validator*)
      if ! has_argument $@; then
        # default to 1 validator if no number provided
        totalValidator=1
      else
        totalValidator=$(extract_argument $@)
      fi
      totalNodes=$(($totalRpc + $totalValidator))
      shift
      ;;

      # check for update and do update
      --update)
      doUpdate
      exit 0
      ;;
      # skip validator account setup (task8)
      --nopk)
      NOPK=true
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


# bootstraping
finalize
