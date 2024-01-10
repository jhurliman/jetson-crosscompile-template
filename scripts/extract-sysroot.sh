#!/usr/bin/env bash

set -e

BOARD_ID="t210"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --board-id [id]   Set the Jetson board ID (default: 't210', valid: 't210', 't186', 't194', 't234')"
    echo "  --help            Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --board-id)
            BOARD_ID="$2"
            shift # past argument
            shift # past value
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)    # unknown option
            show_help
            exit 1
            ;;
    esac
done

TOOLCHAIN_URL="https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit"
TOOLCHAIN_DIRECTORY="gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu"
TOOLCHAIN_FILENAME="${TOOLCHAIN_DIRECTORY}.tar.xz"

# An associative array of symlinks to fix. The key is the existing symlink, and
# the value is the new target (relative path)
declare -A SYMLINKS_TO_FIX=(
    ["/etc/alternatives/cuda-10"]="/usr/local/cuda-10.2"
    ["/etc/alternatives/cudnn_version_h"]="/usr/include/aarch64-linux-gnu/cudnn_version_v8.h"
    ["/usr/lib/aarch64-linux-gnu/libBrokenLocale.so"]="/lib/aarch64-linux-gnu/libBrokenLocale.so.1"
    ["/usr/lib/aarch64-linux-gnu/libanl.so"]="/lib/aarch64-linux-gnu/libanl.so.1"
    ["/usr/lib/aarch64-linux-gnu/libblas.so.3"]="/etc/alternatives/libblas.so.3-aarch64-linux-gnu"
    ["/usr/lib/aarch64-linux-gnu/libcidn.so"]="/lib/aarch64-linux-gnu/libcidn.so.1"
    ["/usr/lib/aarch64-linux-gnu/libcrypt.so"]="/lib/aarch64-linux-gnu/libcrypt.so.1"
    ["/usr/lib/aarch64-linux-gnu/libcudnn.so"]="/etc/alternatives/libcudnn_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_adv_infer.so"]="/etc/alternatives/libcudnn_adv_infer_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_adv_train.so"]="/etc/alternatives/libcudnn_adv_train_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_cnn_infer.so"]="/etc/alternatives/libcudnn_cnn_infer_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_cnn_train.so"]="/etc/alternatives/libcudnn_cnn_train_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_ops_infer.so"]="/etc/alternatives/libcudnn_ops_infer_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_ops_train.so"]="/etc/alternatives/libcudnn_ops_train_so"
    ["/usr/lib/aarch64-linux-gnu/libcudnn_static.a"]="/etc/alternatives/libcudnn_stlib"
    ["/usr/lib/aarch64-linux-gnu/libdl.so"]="/lib/aarch64-linux-gnu/libdl.so.2"
    ["/usr/lib/aarch64-linux-gnu/liblapack.so.3"]="/etc/alternatives/liblapack.so.3-aarch64-linux-gnu"
    ["/usr/lib/aarch64-linux-gnu/libm.so"]="/lib/aarch64-linux-gnu/libm.so.6"
    ["/usr/lib/aarch64-linux-gnu/libnsl.so"]="/lib/aarch64-linux-gnu/libnsl.so.1"
    ["/usr/lib/aarch64-linux-gnu/libnss_compat.so"]="/lib/aarch64-linux-gnu/libnss_compat.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnss_dns.so"]="/lib/aarch64-linux-gnu/libnss_dns.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnss_files.so"]="/lib/aarch64-linux-gnu/libnss_files.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnss_hesiod.so"]="/lib/aarch64-linux-gnu/libnss_hesiod.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnss_nis.so"]="/lib/aarch64-linux-gnu/libnss_nis.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnss_nisplus.so"]="/lib/aarch64-linux-gnu/libnss_nisplus.so.2"
    ["/usr/lib/aarch64-linux-gnu/libnvvpi.so"]="/etc/alternatives/libnvvpi.so"
    ["/usr/lib/aarch64-linux-gnu/libpcre.so"]="/lib/aarch64-linux-gnu/libpcre.so.3"
    ["/usr/lib/aarch64-linux-gnu/libresolv.so"]="/lib/aarch64-linux-gnu/libresolv.so.2"
    ["/usr/lib/aarch64-linux-gnu/librt.so"]="/lib/aarch64-linux-gnu/librt.so.1"
    ["/usr/lib/aarch64-linux-gnu/libthread_db.so"]="/lib/aarch64-linux-gnu/libthread_db.so.1"
    ["/usr/lib/aarch64-linux-gnu/libutil.so"]="/lib/aarch64-linux-gnu/libutil.so.1"
    ["/usr/lib/aarch64-linux-gnu/libz.so"]="/lib/aarch64-linux-gnu/libz.so.1.2.11"
    ["/usr/local/cuda-10.2/lib"]="/usr/local/cuda-10.2/targets/aarch64-linux/lib"
    ["/usr/local/cuda-10.2/lib64"]="/usr/local/cuda-10.2/targets/aarch64-linux/lib"
    ["/usr/local/cuda-10.2/include"]="/usr/local/cuda-10.2/targets/aarch64-linux/include"
)

cd "$(dirname "$0")"

# Build the docker image that will be used to create the sysroot
DOCKER_BUILDKIT=1 docker build --platform linux/arm64 -t "jetson-${BOARD_ID}-sysroot" -f "Dockerfile.jetson-${BOARD_ID}" .
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
