#!/usr/bin/env bash

set -e

BOARD_ID="t210"
TOOLCHAIN_URL="https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit"
TOOLCHAIN_DIRECTORY="gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu"
TOOLCHAIN_FILENAME="${TOOLCHAIN_DIRECTORY}.tar.xz"

# An associative array of symlinks to fix. The key is the existing symlink, and
# the value is the new target (relative path)
declare -A SYMLINKS_TO_FIX=(
    ["/etc/alternatives/cuda-10"]="/usr/local/cuda-10.2"
    ["/etc/alternatives/cudnn_version_h"]="/usr/include/aarch64-linux-gnu/cudnn_version_v8.h"
    ["/usr/local/cuda-10.2/lib"]="/usr/local/cuda-10.2/targets/aarch64-linux/lib"
    ["/usr/local/cuda-10.2/lib64"]="/usr/local/cuda-10.2/targets/aarch64-linux/lib"
    ["/usr/local/cuda-10.2/include"]="/usr/local/cuda-10.2/targets/aarch64-linux/include"
)

cd "$(dirname "$0")"

# Build the docker image that will be used to create the sysroot
docker build -t "jetson-${BOARD_ID}-sysroot" -f "Dockerfile.jetson-${BOARD_ID}" .
docker create --name "jetson-${BOARD_ID}-sysroot" "jetson-${BOARD_ID}-sysroot"

mkdir -p "../sysroot"
cd "../sysroot"

# Export the filesystem from the docker image
docker export "jetson-${BOARD_ID}-sysroot" -o "jetson-${BOARD_ID}-sysroot.tar"
docker rm "jetson-${BOARD_ID}-sysroot"

# Extract the filesystem
rm -rf "jetson-${BOARD_ID}"
mkdir -p "jetson-${BOARD_ID}"
tar -xpf "jetson-${BOARD_ID}-sysroot.tar" -C "jetson-${BOARD_ID}"

# Fix the symlinks
SYSROOT_DIR="$(pwd)/jetson-${BOARD_ID}"
for SYMLINK in "${!SYMLINKS_TO_FIX[@]}"; do
    TARGET="${SYMLINKS_TO_FIX[$SYMLINK]}"
    FULL_SYMLINK_PATH="${SYSROOT_DIR}${SYMLINK}"

    # Remove the existing absolute symlink
    rm -f "${FULL_SYMLINK_PATH}"

    # Create a new relative symlink
    ln -sr "${SYSROOT_DIR}${TARGET}" "${FULL_SYMLINK_PATH}"
    echo "Fixed symlink ${SYMLINK} -> ${TARGET}"
done

echo "sysroot created at $(pwd)/jetson-${BOARD_ID}"

# Check if the toolchain directory exists
if [[ ! -d "${TOOLCHAIN_DIRECTORY}" ]]; then
    # Download and extract the toolchain
    if [[ ! -f "${TOOLCHAIN_FILENAME}" ]]; then
        wget -O "${TOOLCHAIN_FILENAME}" "${TOOLCHAIN_URL}"
    fi
    tar -xpf "${TOOLCHAIN_FILENAME}"
    echo "toolchain extracted to $(pwd)/${TOOLCHAIN_DIRECTORY}"
fi
