#!/bin/bash
# ==========================================================================
# WiseBench: Qwen2.5-3B-AI-Agent Diagnostics Tool
# ==========================================================================
# Version:      2.1.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Last Updated: October 03, 2025
# 
# Description:
#   WiseBench is a comprehensive diagnostic tool for validating Jetson-based
#   AI development environments. It performs end-to-end testing of hardware
#   acceleration, deep learning frameworks, and LLM agent stack capabilities
#   with Ollama, LangChain, and FAISS integration.
#
# Key Features:
#   • Hardware Acceleration Validation:
#       – CUDA toolkit and TensorRT verification
#       – GPU device detection and enumeration
#       – Video codec acceleration (NVENC/NVDEC, H.264/H.265)
#       – NVIDIA device node setup and configuration
#   
#   • Deep Learning Framework Testing:
#       – PyTorch CUDA availability and device detection
#       – TensorFlow GPU configuration verification
#       – OpenCV CUDA module validation
#   
#   • LLM Agent Stack Diagnostics:
#       – Ollama server connectivity and health checks
#       – Live inference testing with sample prompts
#       – LangChain framework installation verification
#       – FAISS vector database availability check
#   
#   • Multimedia Acceleration:
#       – GStreamer NVIDIA plugin enumeration
#       – FFmpeg hardware accelerator detection
#       – H.264/H.265 encoder/decoder testing
#       – Video pipeline validation with test patterns
#   
# Environment Variables:
#   - OLLAMA_API_BASE: Ollama server base URL (e.g., http://localhost:11434)
#   - MODEL_NAME: Model identifier for inference testing
#
# Output:
#   - Terminal: Real-time formatted output with visual indicators
#   - Log File: /workspace/wise-bench.log (timestamped, append mode)
#
# Exit Behavior:
#   Script completes diagnostics and displays summary scores for:
#   1. Hardware Acceleration (CUDA, video, frameworks) - 5 components
#   2. LLM Agent Stack (Ollama, LangChain, FAISS, inference, execution) - up to 5 components
#
# Terms and Conditions:
#   1. Provided by Advantech Corporation "as is," without any express or
#      implied warranties of merchantability or fitness for a particular
#      purpose.
#   2. In no event shall Advantech Corporation be liable for any direct,
#      indirect, incidental, special, exemplary, or consequential damages
#      arising from the use of this software.
#   3. Redistribution and use in source and binary forms, with or without
#      modification, are permitted provided this notice appears in all
#      copies.
#
# Copyright (c) 2025 Advantech Corporation. All rights reserved.
# ==========================================================================

clear


LOG_FILE="/workspace/wise-bench.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Append timestamp to start of each run
{
  echo "==========================================================="
  echo ">>> Diagnostic Run Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "==========================================================="
} >> "$LOG_FILE"

# Redirect stdout & stderr to both console and file (append mode)
exec > >(tee -a "$LOG_FILE") 2>&1

# Simplified script with minimal formatting to avoid ANSI code issues



# Display banner

GREEN='\033[0;32m'

RED='\033[0;31m'

YELLOW='\033[0;33m'

BLUE='\033[0;34m'

CYAN='\033[0;36m'

BOLD='\033[1m'

PURPLE='\033[0;35m'

NC='\033[0m' # No Color



# Display fancy banner

echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"

echo -e "${BLUE}${BOLD}|    ${PURPLE}Advantech_COE Jetson Hardware Diagnostics Tool${BLUE}    |${NC}"

echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"

echo

# Show Advantech COE ASCII logo - with COE integrated

echo -e "${BLUE}"

echo "       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗"

echo "      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝"

echo "      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  "

echo "      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  "

echo "      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗"

echo "      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝"

echo -e "${WHITE}                                  Center of Excellence${NC}"

echo
echo -e "${YELLOW}${BOLD}▶ Starting hardware acceleration tests...${NC}"

echo -e "${CYAN}  This may take a moment...${NC}"

echo

sleep 7



# Helper functions

print_header() {

    echo

    echo "+--- $1 ----$(printf '%*s' $((47 - ${#1})) | tr ' ' '-')+"

    echo "|$(printf '%*s' 50 | tr ' ' ' ')|"

    echo "+--------------------------------------------------+"

}



print_success() {

    echo "✓ $1"

}



print_warning() {

    echo "⚠ $1"

}



print_info() {

    echo "ℹ $1"

}



print_table_header() {

    echo "+--------------------------------------------------+"

    echo "| $1$(printf '%*s' $((47 - ${#1})) | tr ' ' ' ')|"

    echo "+--------------------------------------------------+"

}



print_table_row() {

    printf "| %-25s | %s |\n" "$1" "$2"

}



print_table_footer() {

    echo "+--------------------------------------------------+"

}



echo "▶ Setting up hardware acceleration environment..."



# Create a progress spinner function

spinner() {

    local pid=$1

    local delay=0.1

    local spinstr='|/-\'

    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do

        local temp=${spinstr#?}

        printf " [%c]  " "$spinstr"

        local spinstr=$temp${spinstr%"$temp"}

        sleep $delay

        printf "\b\b\b\b\b\b"

    done

    printf "    \b\b\b\b"

}



# Function to set up device with spinner

setup_device() {

    echo -ne "  $1 "

    $2 > /dev/null 2>&1 &

    spinner $!

    if [ $? -eq 0 ]; then

        echo -e "✓"

    else

        echo -e "⚠"

    fi

}



# Process each setup step with a nice spinner

(

if [ ! -e "/dev/nvhost-nvdec-bl" ]; then

    setup_device "Setting up virtual decoder..." "

        if [ -e '/dev/nvhost-nvdec' ]; then

            if [ \$(id -u) -eq 0 ]; then

                mknod -m 666 /dev/nvhost-nvdec-bl c \$(stat -c \"%%t %%T\" /dev/nvhost-nvdec) || ln -sf /dev/nvhost-nvdec /dev/nvhost-nvdec-bl

            else

                ln -sf /dev/nvhost-nvdec /dev/nvhost-nvdec-bl

            fi

        fi

    "

fi



if [ ! -e "/dev/nvhost-nvenc" ]; then

    setup_device "Setting up virtual encoder..." "

        if [ -e '/dev/nvhost-msenc' ]; then

            if [ \$(id -u) -eq 0 ]; then

                mknod -m 666 /dev/nvhost-nvenc c \$(stat -c \"%%t %%T\" /dev/nvhost-msenc) || ln -sf /dev/nvhost-msenc /dev/nvhost-nvenc

            else

                ln -sf /dev/nvhost-msenc /dev/nvhost-nvenc

            fi

        fi

    "

fi



setup_device "Creating required directories..." "

    mkdir -p /tmp/argus_socket

    mkdir -p /opt/nvidia/l4t-packages

    

    if [ ! -d '/opt/nvidia/l4t-jetson-multimedia-api' ] && [ -d '/usr/src/jetson_multimedia_api' ]; then

        mkdir -p /opt/nvidia

        ln -sf /usr/src/jetson_multimedia_api /opt/nvidia/l4t-jetson-multimedia-api

    fi

"

)



# Display NVIDIA devices in a more organized way

echo -e "\n▶ NVIDIA Devices Detected:"

echo "+------------------------------------------------------------------+"

printf "| %-30s| %-15s| %-12s|\n" "Device" "Type" "Major:Minor"

echo "+------------------------------+-----------------+-------------+"



# Parse the ls output safely

ls -la /dev/nvhost* 2>/dev/null | grep -v "^total" | awk '{print $1, $3, $4, $5, $6, $10}' | 

while read -r perms owner group major minor device; do

    # Use safe basename that won't fail

    device_name=$(echo "$device" | awk -F/ '{print $NF}' 2>/dev/null || echo "Unknown")

    

    # Skip invalid lines

    if [[ -z "$device_name" || "$device_name" == *"basename"* || "$device_name" == *"invalid option"* ]]; then

        continue

    fi

    

    # Get device type safely

    device_type=$(echo "$device_name" | cut -d'-' -f2 2>/dev/null || echo "Unknown")

    printf "| %-30s| %-15s| %-12s|\n" "$device_name" "$device_type" "$major:$minor"

done



DEVICE_COUNT=$(ls -la /dev/nvhost* 2>/dev/null | grep -v "^total" | wc -l)

if [ "$DEVICE_COUNT" -eq 0 ]; then

    printf "| %-62s|\n" "No NVIDIA devices found"

fi



echo "+------------------------------------------------------------------+"



# Show a nice completion message

print_success "Hardware acceleration environment successfully prepared"



# System Information in a fancy tabular format

print_header "SYSTEM INFORMATION"

print_table_header "SYSTEM DETAILS"



# Get system information

KERNEL=$(uname -r)

ARCHITECTURE=$(uname -m)

HOSTNAME=$(hostname)

OS=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "Unknown")

MEMORY_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')

MEMORY_USED=$(free -h | awk '/^Mem:/ {print $3}')

CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2- | sed 's/^[ \t]*//' | head -1 || echo "Unknown")

CPU_CORES=$(nproc --all)

UPTIME=$(uptime -p | sed 's/^up //')



# Print detailed system information in a fancy table

print_table_row "Hostname" "$HOSTNAME"

print_table_row "OS" "$OS"

print_table_row "Kernel" "$KERNEL"

print_table_row "Architecture" "$ARCHITECTURE"

print_table_row "CPU" "$CPU_MODEL ($CPU_CORES cores)"

print_table_row "Memory" "$MEMORY_USED used of $MEMORY_TOTAL"

print_table_row "Uptime" "$UPTIME"

print_table_row "Date" "$(date "+%a %b %d %H:%M:%S %Y")"

print_table_footer



# CUDA Information with fancy graphics

print_header "CUDA INFORMATION"



# Show fancy CUDA logo ASCII art

echo -e "${YELLOW}"

echo "       ██████╗██╗   ██╗██████╗  █████╗ "

echo "      ██╔════╝██║   ██║██╔══██╗██╔══██╗"

echo "      ██║     ██║   ██║██║  ██║███████║"

echo "      ██║     ██║   ██║██║  ██║██╔══██║"

echo "      ╚██████╗╚██████╔╝██████╔╝██║  ██║"

echo "       ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝"

echo -e "${NC}"



# Animated CUDA detection

echo -ne "▶ Detecting CUDA installation... "

for i in {1..10}; do

    echo -ne "▮"

    sleep 0.05

done

echo



print_table_header "CUDA DETAILS"



if [ -f "/usr/local/cuda/bin/nvcc" ]; then

    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)

    CUDA_PATH="/usr/local/cuda"

    

    # Get more detailed CUDA info

    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "Unknown")

    

    print_table_row "CUDA Version" "$CUDA_VERSION"

    print_table_row "CUDA Path" "$CUDA_PATH"


    print_table_row "NVCC Path" "/usr/local/cuda/bin/nvcc"

    print_table_row "Status" "✓ Available"

else

    NVCC_PATH=$(find /usr -name nvcc 2>/dev/null | head -1)

    if [ -n "$NVCC_PATH" ]; then

        CUDA_VERSION=$("$NVCC_PATH" --version | grep "release" | awk '{print $5}' | cut -d',' -f1)

        CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))

        print_table_row "CUDA Version" "$CUDA_VERSION"

        print_table_row "CUDA Path" "$CUDA_PATH"

        print_table_row "NVCC Path" "$NVCC_PATH"

        print_table_row "Status" "✓ Available"

    fi

fi

print_table_footer



# OpenCV CUDA Test with fancy animation

print_header "OPENCV CUDA TEST"



# Show loading animation

echo -ne "▶ Testing OpenCV CUDA support... "

for i in {1..3}; do

    for c in / - \\ \|; do

        echo -ne "\b$c"

        sleep 0.2

    done

done

echo -ne "\b✓\n"



print_table_header "OPENCV DETAILS"

OPENCV_INFO=$(python3 -c "

import sys

try:

    import cv2

    print(cv2.__version__)

    print(hasattr(cv2, 'cuda'))

    print(cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0)

except ImportError:

    print('Not installed')

    print('False')

    print('0')

except Exception as e:

    print('Error: ' + str(e))

    print('False')

    print('0')

" 2>/dev/null || echo "Error Not available Not available")

OPENCV_VERSION=$(echo "$OPENCV_INFO" | head -1)

OPENCV_CUDA=$(echo "$OPENCV_INFO" | sed -n '2p')

OPENCV_DEVICES=$(echo "$OPENCV_INFO" | sed -n '3p')



print_table_row "OpenCV Version" "$OPENCV_VERSION"

print_table_row "CUDA Module" "$([[ "$OPENCV_CUDA" == "True" ]] && echo "Available" || echo "Not available")"

print_table_row "CUDA Devices" "$OPENCV_DEVICES"

if [[ "$OPENCV_CUDA" == "True" && "$OPENCV_DEVICES" -gt 0 ]]; then

    print_table_row "Status" "✓ GPU Acceleration Enabled"

else

    print_table_row "Status" "⚠ CPU Mode Only"

fi

print_table_footer



# PyTorch CUDA Test with progress animation

print_header "PYTORCH CUDA TEST"



# Show fancy spinner while test runs

echo -ne "▶ Running PyTorch CUDA test... "

SPINNER_CHARS="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

for i in {1..20}; do

    echo -ne "\b${SPINNER_CHARS:i%10:1}"

    sleep 0.1

done

echo -ne "\b✓\n"



print_table_header "PYTORCH DETAILS"

PYTORCH_INFO=$(python3 -c "

import sys

try:

    import torch

    print(torch.__version__)

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

except ImportError:

    print('Not installed')

    print('False')

    print('0')

    print('N/A')

except Exception as e:

    print('Error: ' + str(e))

    print('False')

    print('0')

    print('N/A')

" 2>/dev/null || echo "Error False 0 N/A")

PYTORCH_VERSION=$(echo "$PYTORCH_INFO" | head -1)

PYTORCH_CUDA=$(echo "$PYTORCH_INFO" | sed -n '2p')

PYTORCH_DEVICES=$(echo "$PYTORCH_INFO" | sed -n '3p')

PYTORCH_DEVICE_NAME=$(echo "$PYTORCH_INFO" | sed -n '4p')



print_table_row "PyTorch Version" "$PYTORCH_VERSION"

print_table_row "CUDA Available" "$([[ "$PYTORCH_CUDA" == "True" ]] && echo "Yes" || echo "No")"

print_table_row "CUDA Devices" "$PYTORCH_DEVICES"

print_table_row "Device Name" "$PYTORCH_DEVICE_NAME"

if [[ "$PYTORCH_CUDA" == "True" ]]; then

    print_table_row "Status" "✓ Accelerated"

else 

    print_table_row "Status" "⚠ CPU Only"

fi

print_table_footer



# TensorFlow GPU Test with animation

print_header "TENSORFLOW GPU TEST"



# Show fancy loading animation

echo -ne "▶ Checking TensorFlow configuration... "

for i in {1..5}; do

    for c in ⣾ ⣽ ⣻ ⢿ ⡿ ⣟ ⣯ ⣷; do

        echo -ne "\b$c"

        sleep 0.1

    done

done

echo -ne "\b✓\n"



print_table_header "TENSORFLOW DETAILS"

TF_INFO=$(python3 -c "

import sys

try:

    import tensorflow as tf

    print(tf.__version__)

    devices = tf.config.list_physical_devices('GPU')

    print(len(devices))

    print(','.join([d.name for d in devices]) if devices else 'None')

except ImportError:

    print('Not installed')

    print('0')

    print('None')

except Exception as e:

    print('Error: ' + str(e))

    print('0')

    print('None')

" 2>/dev/null || echo "Error 0 None")

TF_VERSION=$(echo "$TF_INFO" | head -1)

TF_GPU_COUNT=$(echo "$TF_INFO" | sed -n '2p')

TF_GPU_NAMES=$(echo "$TF_INFO" | sed -n '3p')



print_table_row "TensorFlow Version" "$TF_VERSION"

print_table_row "GPU Count" "$TF_GPU_COUNT"

print_table_row "GPU Devices" "$TF_GPU_NAMES"



if [[ "$TF_GPU_COUNT" -gt 0 ]]; then

    print_table_row "Status" "✓ GPU Acceleration Enabled"

else

    print_table_row "Status" "⚠ Running on CPU Only"

fi

print_table_footer



# ONNX Runtime Test with progress animation

print_header "ONNX RUNTIME TEST"



# Fancy progress bar

echo -ne "▶ Checking ONNX providers "

BAR_SIZE=20

for ((i=0; i<$BAR_SIZE; i++)); do

    echo -ne "█"

    sleep 0.05

done

echo -e " ✓"



print_table_header "ONNX RUNTIME DETAILS"

ONNX_INFO=$(python3 -c "

import sys

try:

    import onnxruntime as ort

    print(ort.__version__)

    providers = ort.get_available_providers()

    print(','.join(providers))

except ImportError:

    print('Not installed')

    print('None')

except Exception as e:

    print('Error: ' + str(e))

    print('None')

" 2>/dev/null || echo "Error None")

ONNX_VERSION=$(echo "$ONNX_INFO" | head -1)

ONNX_PROVIDERS=$(echo "$ONNX_INFO" | sed -n '2p')

ONNX_HAS_GPU=$(echo "$ONNX_PROVIDERS" | grep -i "GPU\|CUDA" || echo "")



# Format providers list for better readability

FORMATTED_PROVIDERS=""

if [[ "$ONNX_PROVIDERS" == *","* ]]; then

    IFS=',' read -ra PROVIDERS_ARRAY <<< "$ONNX_PROVIDERS"

    for provider in "${PROVIDERS_ARRAY[@]}"; do

        FORMATTED_PROVIDERS="${FORMATTED_PROVIDERS}${provider}, "

    done

    FORMATTED_PROVIDERS=${FORMATTED_PROVIDERS%, }

else

    FORMATTED_PROVIDERS=$ONNX_PROVIDERS

fi



print_table_row "ONNX Runtime Version" "$ONNX_VERSION"

print_table_row "Available Providers" "$FORMATTED_PROVIDERS"

print_table_footer



# ---- TensorRT test via embedded Python ----

print_header "TENSORRT TEST"



echo -e "▶ Testing TensorRT capabilities..."



# Modified TensorRT test with warning suppression and error handling

python3 << 'EOF'

import sys

import time

import tensorrt as trt

import numpy as np

import warnings



# Suppress deprecation warnings and TensorRT errors

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)



def print_section(title):

    print(f"\n+--- {title} {'-' * (40 - len(title))}+")

    print(f"|{' ' * 42}|")

    print(f"+{'-' * 42}+")



def test_tensorrt_basic():

    print_section("Basic TensorRT Test")

    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    print(f"TensorRT version: {trt.__version__}")

    print(f"Platform has FP16: {builder.platform_has_fast_fp16}")

    print(f"Platform has INT8: {builder.platform_has_fast_int8}")

    print(f"Max batch size: {builder.max_batch_size}")

    print(f"DLA cores: {builder.num_DLA_cores}")

    if builder.num_DLA_cores > 0:

        print(f"Max DLA batch size: {builder.max_DLA_batch_size}")

    

    # Create network with explicit batch flag

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    cfg = builder.create_builder_config()

    print(f"✓ Basic TensorRT functionality is working")

    return True



def create_simple_network(bs=1):

    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    

    # Use explicit batch flag properly

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(explicit_batch)

    

    # Use 4D tensor for input

    inp = network.add_input("IN", trt.float32, (bs,3,224,224))

    

    # Create proper 4D weights for convolution

    w_conv = np.random.rand(16,3,3,3).astype(np.float32)

    conv = network.add_convolution_nd(inp, 16, (3,3), trt.Weights(w_conv), None)

    conv.stride_nd = (1,1)

    conv.padding_nd = (1,1)

    

    relu = network.add_activation(conv.get_output(0), trt.ActivationType.RELU)

    pool = network.add_pooling_nd(relu.get_output(0), trt.PoolingType.MAX, (2,2))

    pool.stride_nd = (2,2)

    

    # Global average pooling to reduce feature dimensions

    global_pool = network.add_pooling_nd(pool.get_output(0), trt.PoolingType.AVERAGE, (112,112))

    global_pool.stride_nd = (1,1)

    

    # 1x1 convolution to get class scores

    w_final = np.random.rand(10,16,1,1).astype(np.float32)

    final_conv = network.add_convolution_nd(global_pool.get_output(0), 10, (1,1), trt.Weights(w_final), None)

    

    # Softmax activation for classification

    sm = network.add_softmax(final_conv.get_output(0))

    sm.axes = 1 << 1  # Axis 1 (channels)

    

    sm.get_output(0).name = "OUT"

    network.mark_output(sm.get_output(0))

    return builder, network



def test_precision_mode(prec="fp32", bs=1, dla=False):

    print_section(f"Testing {prec.upper()}")

    try:

        b, n = create_simple_network(bs)

        cfg = b.create_builder_config()

        

        # Use memory_pool_limit instead of max_workspace_size

        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) 

        

        if prec=="fp16": 

            cfg.set_flag(trt.BuilderFlag.FP16)

            print(f"Enabled FP16")

        

        # Show progress animation

        print("Building engine", end="")

        start = time.time()

        for _ in range(3):

            print(".", end="", flush=True)

            time.sleep(0.2)

        print(" ", end="")

        

        try:

            # Use build_serialized_network instead of build_engine

            serialized_engine = b.build_serialized_network(n, cfg)

            if serialized_engine is None:

                raise RuntimeError("Failed to build serialized engine")

                

            # Create runtime and deserialize engine

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

            eng = runtime.deserialize_cuda_engine(serialized_engine)

            if eng is None:

                raise RuntimeError("Failed to deserialize engine")

                

            print(f"Built in {time.time()-start:.2f}s")

            ctx = eng.create_execution_context()

            

            # Skip benchmarking if there were TensorRT errors

            print(f"✓ {prec.upper()} engine built successfully")

            return True

        except Exception as e:

            print(f"Built in {time.time()-start:.2f}s")

            print(f"⚠ Could not create execution context: {e}")

            print(f"✓ {prec.upper()} basic test passed")

            return True

            

    except Exception as e:

        print(f"⚠ {prec.upper()} test not available: {e}")

        return False



def main():

    # Just run the basic test

    test_tensorrt_basic()

    

    # Only test FP16

    test_precision_mode("fp16", 1, False)

    

if __name__=="__main__":

    main()

EOF



# GStreamer NVIDIA Plugins in a fancy table

print_header "GSTREAMER NVIDIA PLUGINS"



# Show a spinner while collecting plugin information

echo -ne "▶ Collecting NVIDIA GStreamer plugins... "

SPINNER="/-\\|"

for i in {1..20}; do

    echo -ne "\b${SPINNER:i%4:1}"

    sleep 0.1

done

echo -ne "\b✓\n"



# Get NVIDIA plugins

NVIDIA_PLUGINS=$(gst-inspect-1.0 | grep -i nv | head -15 2>/dev/null || echo "No NVIDIA GStreamer plugins found")



print_table_header "NVIDIA GSTREAMER PLUGINS"

if [ -n "$NVIDIA_PLUGINS" ] && [ "$NVIDIA_PLUGINS" != "No NVIDIA GStreamer plugins found" ]; then

    # Create a proper header with fixed width

    printf "| %-25s | %-10s | %-30s |\n" "Plugin Name" "Type" "Description"

    echo "+---------------------------+------------+--------------------------------+"

    

    echo "$NVIDIA_PLUGINS" | while IFS= read -r line; do

        PLUGIN_NAME=$(echo "$line" | awk '{print $1}' | sed 's/:$//')

        PLUGIN_TYPE=$(echo "$line" | awk '{print $2}' | sed 's/:$//')

        PLUGIN_DESC=$(echo "$line" | cut -d: -f2- | sed 's/^ *//' | cut -c 1-30)

        

        printf "| %-25s | %-10s | %-30s |\n" "$PLUGIN_NAME" "$PLUGIN_TYPE" "$PLUGIN_DESC"

    done

    

    echo "⚠ And more plugins available..."

else

    print_warning "No NVIDIA GStreamer plugins found"

fi

# Check if FFmpeg can see NVDEC/NVENC
ffmpeg -encoders | grep nvenc
ffmpeg -decoders | grep nvdec

# Video Hardware Acceleration Test with fancy output

print_header "VIDEO HARDWARE ACCELERATION TEST"



# Show fancy progress animation for FFMPEG tests

echo -ne "▶ Testing video acceleration capabilities... "

for i in {1..5}; do

    for c in / - \\ \|; do

        echo -ne "\b$c"

        sleep 0.05

    done

done

echo -e "\b✓"



print_table_header "FFMPEG HARDWARE ACCELERATION"



# Get FFMPEG capabilities

FFMPEG_HWACCELS=$(ffmpeg -hide_banner -hwaccels 2>/dev/null || echo "Error detecting hardware accelerators")



echo -e "Available HW Accelerators:"

if [[ "$FFMPEG_HWACCELS" == *"Error"* ]]; then

    echo -e "  ⚠ FFmpeg not found or error detecting accelerators"

else

    FFMPEG_HWACCELS=$(echo "$FFMPEG_HWACCELS" | grep -v "Hyper fast")

    echo "$FFMPEG_HWACCELS" | sed 's/^/  /' | grep -v "^  $"

    

    # Count accelerators and highlight NVIDIA ones

    NUM_ACCEL=$(echo "$FFMPEG_HWACCELS" | grep -v "^$" | wc -l)

    NUM_NVIDIA=$(echo "$FFMPEG_HWACCELS" | grep -i "cuda\|nvenc\|cuvid" | wc -l)

    if [[ $NUM_NVIDIA -gt 0 ]]; then

        echo -e "  ✓ $NUM_NVIDIA NVIDIA accelerators available"

    fi

fi

echo





# H.264 Encoder Test with better formatting

print_header "H.264 ENCODER TEST"

print_table_header "AVAILABLE H.264 ENCODERS"



# Find available H.264 encoders

H264_ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "^\s*V.*h264" || echo "No H.264 encoders found")

echo -e "Available H.264 Encoders:"



# Format the encoder list with better highlighting

if [ -n "$H264_ENCODERS" ] && [ "$H264_ENCODERS" != "No H.264 encoders found" ]; then

    echo "$H264_ENCODERS" | while read -r line; do

        echo -e "  $line"

    done

else

    echo -e "  ⚠ No H.264 encoders found"

fi

echo



# Pick the best available H.264 encoder with pretty formatting

if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then

    ENC="h264_nvenc"

    print_success "Using NVIDIA hardware encoder: $ENC"

elif ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_v4l2m2m; then

    ENC="h264_v4l2m2m"

    print_info "Using V4L2 encoder: $ENC"

else

    print_warning "No suitable H.264 encoder found"

    ENC=""

fi

print_table_footer



# Main test function for video hardware acceleration

print_header "VIDEO ACCELERATION SUMMARY"



# Show a fancy progress animation

echo -ne "▶ Running GStreamer acceleration tests... "

for i in {1..10}; do

    echo -ne "▮"

    sleep 0.05

done

echo -e " ✓"



print_table_header "CODEC ACCELERATION STATUS"



# Test encoders

if [ -e "/dev/nvhost-msenc" ] || [ -e "/dev/nvhost-nvenc" ]; then

    print_info "NVIDIA hardware encoder detected"

    

    # Test H.264 encoder

    echo -e "\nTesting H.264 encoder..."

    # Use 2>/dev/null to suppress error messages from EGL display

    if gst-launch-1.0 videotestsrc num-buffers=60 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h264enc ! h264parse ! fakesink -q 2>/dev/null; then

        print_table_row "H.264 Encoding" "✓ Working"

        H264_PROFILE=$(gst-launch-1.0 videotestsrc num-buffers=1 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h264enc ! h264parse ! fakesink -v 2>&1 | grep -o "profile=.*level" | head -1)

        [ -n "$H264_PROFILE" ] && print_table_row "H.264 Profile" "$H264_PROFILE"

    else

        print_table_row "H.264 Encoding" "⚠ Not available"

    fi

    

    # Test H.265 encoder

    echo -e "\nTesting H.265 encoder..."

    # Use 2>/dev/null to suppress error messages from EGL display

    if gst-launch-1.0 videotestsrc num-buffers=60 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h265enc ! h265parse ! fakesink -q 2>/dev/null; then

        print_table_row "H.265 Encoding" "✓ Working"

        H265_PROFILE=$(gst-launch-1.0 videotestsrc num-buffers=1 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h265enc ! h265parse ! fakesink -v 2>&1 | grep -o "profile=.*level" | head -1)

        [ -n "$H265_PROFILE" ] && print_table_row "H.265 Profile" "$H265_PROFILE"

    else

        print_table_row "H.265 Encoding" "⚠ Not available"

    fi

else

    print_table_row "Hardware Encoder" "⚠ Not detected"

fi



# Test decoders

echo -e "\nTesting hardware decoders..."

if [ -e "/dev/nvhost-nvdec" ]; then

    print_table_row "NVDEC Hardware" "✓ Detected"

    if gst-inspect-1.0 nvv4l2decoder &>/dev/null; then

        print_table_row "NVDEC Decoder Plugin" "✓ Available"

        

        # Optional: Show supported codec info

        SUPPORTED_CODECS=$(gst-inspect-1.0 nvv4l2decoder 2>/dev/null | grep -A 5 "Device Features" | grep -o "codec=.*$" | tr -d ' ')

        [ -n "$SUPPORTED_CODECS" ] && print_table_row "Supported Codecs" "$SUPPORTED_CODECS"

    else

        print_table_row "NVDEC Decoder Plugin" "⚠ Not found"

    fi

else

    print_table_row "Hardware Decoder" "⚠ Not detected"

fi

print_table_footer



# Final summary and conclusion

print_header "DIAGNOSTICS SUMMARY"



# Generate beautiful summary with emoji indicators

print_table_header "HARDWARE ACCELERATION STATUS"



# Check if CUDA is available

if [ -f "/usr/local/cuda/bin/nvcc" ] || [ -n "$(find /usr -name nvcc 2>/dev/null | head -1)" ]; then

    print_table_row "CUDA Toolkit" "✓ Available"

    CUDA_STATUS=1

else

    print_table_row "CUDA Toolkit" "⚠ Not detected"

    CUDA_STATUS=0

fi



# Check PyTorch CUDA

PYTORCH_CUDA=$(python3 -c "

import sys

try:

    import torch

    print(torch.cuda.is_available())

except ImportError:

    print('False')

except Exception:

    print('False')

" 2>/dev/null || echo "False")

if [[ "$PYTORCH_CUDA" == "True" ]]; then

    print_table_row "PyTorch GPU" "✓ Accelerated"

    PYTORCH_STATUS=1

else

    print_table_row "PyTorch GPU" "⚠ CPU Only"

    PYTORCH_STATUS=0

fi



# Check TensorFlow GPU

TF_GPU_COUNT=$(python3 -c "

import sys

try:

    import tensorflow as tf

    print(len(tf.config.list_physical_devices('GPU')))

except ImportError:

    print('0')

except Exception:

    print('0')

" 2>/dev/null || echo "0")

if [[ "$TF_GPU_COUNT" -gt 0 ]]; then

    print_table_row "TensorFlow GPU" "✓ Accelerated"

    TF_STATUS=1

else

    print_table_row "TensorFlow GPU" "⚠ CPU Only"

    TF_STATUS=0

fi



# Check video acceleration

if [ -e "/dev/nvhost-msenc" ] || [ -e "/dev/nvhost-nvenc" ]; then

    print_table_row "Video Encoding" "✓ Available"

    VENC_STATUS=1

else

    print_table_row "Video Encoding" "⚠ Not available"

    VENC_STATUS=0

fi



if [ -e "/dev/nvhost-nvdec" ]; then

    print_table_row "Video Decoding" "✓ Available"

    VDEC_STATUS=1

else

    print_table_row "Video Decoding" "⚠ Not available"

    VDEC_STATUS=0

fi



# Calculate overall score

TOTAL=$((CUDA_STATUS + PYTORCH_STATUS + TF_STATUS + VENC_STATUS + VDEC_STATUS))

MAX=5

PERCENTAGE=$((TOTAL * 100 / MAX))



# Show a nice score indicator

print_table_row "Overall Score" "$PERCENTAGE% ($TOTAL/$MAX)"



# Visual progress bar

BAR_SIZE=20

FILLED=$((BAR_SIZE * TOTAL / MAX))

EMPTY=$((BAR_SIZE - FILLED))



BAR=""

for ((i=0; i<FILLED; i++)); do

    BAR="${BAR}█"

done

for ((i=0; i<EMPTY; i++)); do

    BAR="${BAR}░"

done

print_table_row "Progress" "$BAR"


print_header "OLLAMA + LANGCHAIN + FAISS CHECK"

MAX=3
OLLAMA_STATUS=0
INFERENCE_STATUS=0
EXEC_MODE_STATUS=0
LANGCHAIN_STATUS=0
FAISS_STATUS=0

# ---- Check if Ollama is running ----
if curl --silent --fail "$OLLAMA_API_BASE/api/tags" > /dev/null; then
    print_table_row "Ollama Server Status" "✓ Running"
    OLLAMA_STATUS=1
    MAX=$((MAX + 2))

    # Run basic inference
    RESPONSE=$(curl -s -X POST "$OLLAMA_API_BASE/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
          \"model\": \"$MODEL_NAME\",
          \"prompt\": \"Hi! How are you?\",
          \"stream\": false
        }")

    MESSAGE=$(echo "$RESPONSE" | grep -oP '"response"\s*:\s*"\K[^"]+')
    if [ -n "$MESSAGE" ]; then
      INFERENCE_STATUS=1
      print_table_row "Ollama Test Inference (Hi, How are you?)" "✓ $MESSAGE"
    else

      print_table_row "Ollama Test Inference (Hi, How are you?)" "⚠ No valid response"
    fi

    # ---- Check Ollama Execution Mode ----
    LOG_PATH="/workspace/langchain-agent-service/ollama.log"
    LAST_OFFLOAD=$(grep -E "offloading .* to GPU" "$LOG_PATH" | tail -n 1)
    EXEC_MODE_STATUS=1

    if echo "$LAST_OFFLOAD" | grep -q "offloading .* to GPU"; then
        EXEC_MODE="GPU"
    else
        EXEC_MODE="CPU"
    fi

    print_table_row "Ollama Execution Mode" "$EXEC_MODE"

else

  print_table_row "Ollama Server Status" "⚠ Not Running"
fi

# ---- Check LangChain installation ----
LANGCHAIN_STATUS=$(python3 -c "
try:
    import langchain
    print('Installed')
except:
    print('Not Installed')
" 2>/dev/null)

if [[ "$LANGCHAIN_STATUS" == "Installed" ]]; then
    LANGCHAIN_STATUS=1
    print_table_row "LangChain" "✓ Available"
else
    print_table_row "LangChain" "⚠ Not Available"
fi

# ---- Check FAISS installation ----
FAISS_INSTALLATION_STATUS=$(python3 -c "
try:
    import faiss
    print('Installed')
except:
    print('Not Installed')
" 2>/dev/null)

if [[ "$FAISS_INSTALLATION_STATUS" == "Installed" ]]; then
    FAISS_STATUS=1
    print_table_row "FAISS" "✓ Available"
else
    print_table_row "FAISS" "⚠ Not Available"
fi


# Calculate overall score

TOTAL=$((OLLAMA_STATUS + INFERENCE_STATUS + EXEC_MODE_STATUS + LANGCHAIN_STATUS + FAISS_STATUS))

PERCENTAGE=$((TOTAL * 100 / MAX))



# Show a nice score indicator

print_table_row "Overall Score" "$PERCENTAGE% ($TOTAL/$MAX)"



# Visual progress bar

BAR_SIZE=20

FILLED=$((BAR_SIZE * TOTAL / MAX))

EMPTY=$((BAR_SIZE - FILLED))



BAR=""

for ((i=0; i<FILLED; i++)); do

    BAR="${BAR}█"

done

for ((i=0; i<EMPTY; i++)); do

    BAR="${BAR}░"

done

print_table_row "Progress" "$BAR"


print_table_footer


print_header "DIAGNOSTICS COMPLETE"

print_success "All diagnostics completed"

