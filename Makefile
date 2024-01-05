.PHONY: all host jetson-nano test-cpu test-cuda clean

# Default to building for host
all: host

# Build for host
host:
	cmake --preset host
	cmake --build build/host

# Build for jetson-nano
jetson-nano:
	cmake --preset jetson-nano
	cmake --build build/jetson-nano

# Test CPU backend on host
test-cpu: host
	./build/host/tests/unit_tests_cpu

# Test CUDA backend on host
test-cuda: host
	./build/host/tests/unit_tests_cuda

# Clean up build directories
clean:
	rm -rf build
