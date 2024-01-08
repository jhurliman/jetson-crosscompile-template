.PHONY: all host jetson-nano test tidy clean

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

# Test CPU and CUDA backends on host
test: host
	ctest --preset host

# Run clang-tidy on host
tidy: host
	clang-tidy -p build/host --config-file=.clang-tidy --use-color $$(find include/ -name '*.hpp') $$(find src/ -name '*.cpp')

# Clean up build directories
clean:
	rm -rf build
