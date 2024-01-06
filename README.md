# NVIDIA Jetson C++ Cross-Compilation Template

> _A starting point for C++ / CUDA cross-compilation and debugging for NVIDIA Jetson_

## Introduction

NVIDIA Jetson devices are great for edge computing, but the development experience of storing source code and compiling on device is not ideal. This template project provides a starting point for cross-compiling and remotely debugging C++ / CUDA applications for NVIDIA Jetson devices using Visual Studio Code, CMake, clang, lld, and gdb.

## Prerequisites

### Development Machine

The following prerequisites must be installed on the development (or host) machine:

- [Visual Studio Code](https://code.visualstudio.com/) (optional)
- [CMake](https://cmake.org/) (version 3.18 or higher)
- [clang](https://clang.llvm.org/)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- [cmake-format](https://github.com/cheshirekow/cmake_format)
- [lld](https://lld.llvm.org/)
- [gdb-multiarch](https://www.gnu.org/software/gdb/) (for debugging)
- [Docker](https://www.docker.com/) (for initial sysroot creation)
- [QEMU](https://www.qemu.org/) (for initial sysroot creation, only required on non-arm64 host architectures)

On Ubuntu, you can install these prerequisites with the following command:

```bash
CLANG_VERSION=15 sh -c 'sudo apt update && sudo apt install -y binfmt-support cmake clang-$CLANG_VERSION clang-tools-$CLANG_VERSION clang-format-$CLANG_VERSION clang-tidy-$CLANG_VERSION lld-$CLANG_VERSION cmake-format docker.io gdb-multiarch qemu qemu-user-static'
```

If you install a specific version of clang (such as `clang-15 clang-tools-15 clang-format-15`), you will need to use `update-alternatives` to set the default clang version:

```bash
CLANG_VERSION=15 sh -c 'sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$CLANG_VERSION 100 && sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$CLANG_VERSION 100 && sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-$CLANG_VERSION 100 && sudo update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-$CLANG_VERSION 100'
```

If you just installed Docker, you will need to add your user to the `docker` group and open a new shell with the updated group membership (or log out and back in):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

If you just installed QEMU, you will need to register binfmt with QEMU for different architectures:

```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

### Jetson Device

No prerequisites are strictly required on the Jetson device, but you may need to install the NVIDIA JetPack SDK to get CUDA and other libraries. The sysroot that is created on the development machine includes the full JetPack SDK, but any dynamic libraries that are used must be installed on the Jetson device.

To enable debugging, you will need to install gdbserver on the Jetson device:

```bash
sudo apt update && sudo apt install gdbserver
```

## Usage

Use this template to create a new repository, then clone it to your local machine. There is a one-time setup process to create the sysroot for your target Jetson device, and a matching version of the CUDA toolkit for the development machine's architecture. In a terminal, run the following scripts:

```bash
# Create a sysroot targeting Jetson Nano (T210)
./scripts/extract-sysroot.sh --board-id t210

# Extract the CUDA 10.2 toolkit for the host architecture
./scripts/extract-cuda.sh
```

The second command will initially complain about missing `.deb` files, with links to download them from the NVIDIA developer site (login required). Once you have downloaded the missing files, run the command again to extract the CUDA toolkit.

With the prerequisites installed and one-time setup completed, you are ready for development. Look at the [CMakeLists.txt](CMakeLists.txt) file to see how the starter example application is configured. The [Makefile](Makefile) provides easy access to common tasks:

```bash
# Build the project for current (host) architecture
make

# Build the project for Jetson Nano
make jetson-nano

# Run unit tests via ctest
make test

# Clean the project
make clean
```

These commands are simple wrappers around CMake, which can be used directly if you prefer. The generated build files are stored in `build/host` and `build/jetson-nano` for host and Jetson architectures, respectively.

Alternatively, you can use the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension for Visual Studio Code to build the project. The [tasks.json](.vscode/tasks.json) file provides tasks for CMake configuration and building for host or Jetson architectures. Other VS Code settings files are provided to enable error checking and formatting using clang.

## License

Released by [John Hurliman](https://jhurliman.org/) under the [MIT license](LICENSE).

## TODO

- [x] ctest unit test
- [x] CPM dependency management
- [ ] IWYU integration
- [ ] `make jetson-nano` in CI
- [ ] fix <<<...>>> clang warning for CUDA kernel launch
- [ ] C++23
- [ ] Add support for more Jetson devices
