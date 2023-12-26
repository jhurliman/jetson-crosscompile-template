.PHONY: all host jetson-nano clean

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

# Clean up build directories
clean:
	rm -rf build
