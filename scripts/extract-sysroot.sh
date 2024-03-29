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

# For T210 (Jetson Nano)
TOOLCHAIN_URL="https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit"
TOOLCHAIN_DIRECTORY="gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu"
TOOLCHAIN_FILENAME="${TOOLCHAIN_DIRECTORY}.tar.xz"

# Handle T194 (Jetson AGX Xavier)
if [[ "$BOARD_ID" == "t194" ]]; then
    TOOLCHAIN_URL="https://developer.nvidia.com/embedded/jetson-linux/bootlin-toolchain-gcc-93"
    TOOLCHAIN_DIRECTORY="bootlin-toolchain-gcc-93"
    TOOLCHAIN_FILENAME="${TOOLCHAIN_DIRECTORY}.tar.gz"
fi

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

# Fix absolute symlinks in the sysroot
SYSROOT_DIR="$(pwd)/jetson-${BOARD_ID}"
find "$SYSROOT_DIR" -type l -print0 | while IFS= read -r -d '' symlink; do
    target="$(readlink "$symlink")"
    # Check if the symlink is absolute and does not start with SYSROOT_DIR
    if [[ "$target" == /* && "$target" != "$SYSROOT_DIR"* ]]; then
        # Make the target relative to the symlink
        fixed_target="$SYSROOT_DIR$target"

        ln -sf "$fixed_target" "$symlink" 2>/dev/null || {
            sudo ln -sf "$fixed_target" "$symlink"
        }
    fi
done

echo "sysroot created at $(pwd)/jetson-${BOARD_ID}"

# Check if the toolchain directory exists
if [[ ! -d "${TOOLCHAIN_DIRECTORY}" ]]; then
    # Download and extract the toolchain
    if [[ ! -f "${TOOLCHAIN_FILENAME}" ]]; then
        wget -O "${TOOLCHAIN_FILENAME}" "${TOOLCHAIN_URL}"
    fi
    mkdir -p "${TOOLCHAIN_DIRECTORY}"
    tar -xpf "${TOOLCHAIN_FILENAME}" -C "${TOOLCHAIN_DIRECTORY}"
    echo "toolchain extracted to $(pwd)/${TOOLCHAIN_DIRECTORY}"
fi
