.PHONY: all host host-cpu jetson-nano test test-cpu tidy clean

# Default to building for host
all: host

# Build for host
host:
	cmake --preset host
	cmake --build build/host

# Build for host (CPU only)
host-cpu:
	cmake --preset host-cpu
	cmake --build build/host

# Build for jetson-nano
jetson-nano:
	cmake --preset jetson-nano
	cmake --build build/jetson-nano

# Test CPU and CUDA backends on host
test: host
	ctest --preset host -j 2

test-cpu: host-cpu
	ctest --preset host-cpu

# Run clang-tidy on host
tidy: host
	clang-tidy -p build/host --config-file=.clang-tidy --use-color $$(find src/ -name '*.cpp')

# Clean up build directories
clean:
	rm -rf build
