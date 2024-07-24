#!/usr/bin/env bash

set -e

CUDA_VERSION="10.2"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --cuda [version]  Set the CUDA version (default: '10.2', valid: '10.2', '11.4')"
    echo "  --help            Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
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

# Sanity check
if [[ "${CUDA_VERSION}" != "10.2" && "${CUDA_VERSION}" != "11.4" ]]; then
    echo "Invalid CUDA version: ${CUDA_VERSION}"
    show_help
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p "../nvidia"
cd "../nvidia"

# Handle macOS
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "${CUDA_VERSION}" != "10.2" ]]; then
        echo "CUDA ${CUDA_VERSION} is not supported on macOS"
        exit 1
    fi

    # Fetch the CUDA toolkit 10.2 macOS .dmg from NVIDIA
    FILENAME="cuda_10.2.89_mac.dmg"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/${FILENAME}"
    if [[ ! -f "${FILENAME}" ]]; then
        echo "Downloading CUDA ${FILENAME}..."
        wget --continue -O "${FILENAME}" "${CUDA_URL}"
    fi

    # Mount the .dmg file
    hdiutil attach "${FILENAME}"
    VOLUME=$(ls /Volumes | grep "CUDA")
    CUDA_VOLUME="/Volumes/${VOLUME}"
    echo "Mounted ${CUDA_VOLUME}"

    # Extract the CUDA Toolkit tarball to ./cuda-10.2_macos
    mkdir -p "cuda-10.2_macos"
    tar -xf "${CUDA_VOLUME}/CUDAMacOSXInstaller.app/Contents/Resources/payload/cuda_mac_installer_tk.tar.gz" \
        --strip-components=3 \
        -C ./cuda-10.2_macos \
        Developer/NVIDIA/CUDA-10.2/

    # Unmount the .dmg file
    hdiutil detach "${CUDA_VOLUME}"

    echo "CUDA toolkit extracted to $(pwd)/cuda-10.2_macos"

    exit 0
fi

# Build the docker image that will contain the CUDA toolkit
docker build --platform linux/amd64 -t "cuda-${CUDA_VERSION}_amd64" -f "../scripts/Dockerfile.cuda-${CUDA_VERSION}_amd64" .
docker create --name "cuda-${CUDA_VERSION}_amd64-container" "cuda-${CUDA_VERSION}_amd64"

# Extract the CUDA toolkit from the docker image
rm -rf "cuda-${CUDA_VERSION}_amd64"
docker cp "cuda-${CUDA_VERSION}_amd64-container:/usr/local/cuda-${CUDA_VERSION}" "cuda-${CUDA_VERSION}_amd64"
docker rm "cuda-${CUDA_VERSION}_amd64-container"

echo "CUDA toolkit extracted to $(pwd)/cuda-${CUDA_VERSION}_amd64"
