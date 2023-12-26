#!/usr/bin/env bash

set -e

BASE_CROSS_URL="https://developer.nvidia.com/assets/embedded/secure/tools/files/jetpack-sdks/jetpack-4.6/JETPACK_46_b194/"
BASE_CUDA_URL="https://developer.nvidia.com/assets/embedded/secure/tools/files/jetpack-sdks/jetpack-4.6/JETPACK_46_b194/ubuntu1804/"
CROSS_DEB="cuda-repo-cross-aarch64-ubuntu1804-10-2-local_10.2.460-1_all.deb"
CUDA_DEB="cuda-repo-ubuntu1804-10-2-local_10.2.460-450.115-1_amd64.deb"

cd "$(dirname "$0")"

mkdir -p "../nvidia"
cd "../nvidia"

# Ensure the required `.deb` files exist
DEBS_EXIST=1
if [[ ! -f "./${CROSS_DEB}" ]]; then
    echo "ERROR: $(realpath ./${CROSS_DEB}) does not exist. Download it from <${BASE_CROSS_URL}${CROSS_DEB}> (requires NVIDIA login)"
    DEBS_EXIST=0
fi
if [[ ! -f "./${CUDA_DEB}" ]]; then
    echo "ERROR: $(realpath ./${CUDA_DEB}) does not exist. Download it from <${BASE_CUDA_URL}${CUDA_DEB}> (requires NVIDIA login)"
    DEBS_EXIST=0
fi
if [[ "${DEBS_EXIST}" -eq 0 ]]; then
    exit 1
fi

# Build the docker image that will contain the CUDA toolkit
docker build -t "cuda-10.2_amd64" -f "../scripts/Dockerfile.cuda-10.2_amd64" .
docker create --name "cuda-10.2_amd64-container" "cuda-10.2_amd64"

# Extract the CUDA toolkit from the docker image
rm -rf "cuda-10.2_amd64"
docker cp "cuda-10.2_amd64-container:/usr/local/cuda-10.2" "cuda-10.2_amd64"
docker rm "cuda-10.2_amd64-container"

echo "CUDA toolkit extracted to $(pwd)/cuda-10.2_amd64"
