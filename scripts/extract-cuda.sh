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

# Build the docker image that will contain the CUDA toolkit
docker build -t "cuda-${CUDA_VERSION}_amd64" -f "../scripts/Dockerfile.cuda-${CUDA_VERSION}_amd64" .
docker create --name "cuda-${CUDA_VERSION}_amd64-container" "cuda-${CUDA_VERSION}_amd64"

# Extract the CUDA toolkit from the docker image
rm -rf "cuda-${CUDA_VERSION}_amd64"
docker cp "cuda-${CUDA_VERSION}_amd64-container:/usr/local/cuda-${CUDA_VERSION}" "cuda-${CUDA_VERSION}_amd64"
docker rm "cuda-${CUDA_VERSION}_amd64-container"

echo "CUDA toolkit extracted to $(pwd)/cuda-${CUDA_VERSION}_amd64"
