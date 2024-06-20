# Build

Building NNVISR requires a compiler supporting C++20 and CUDA Compiler >= 12.

## Automatic build

We provide an automatic build script for NNVISR. 

Before running the script,
you should have CMake and a working C++ compiler installed and visible.
On Windows, this means you should run the script in
Visual Studio Developer Command Prompt.
We strongly recommend installing [`ninja`](https://ninja-build.org/). The build script
will automatically use `ninja` during build if present.
The script will automatically fetch CUDA compiler and dependencies.

If you are building with TensorRT 8.6 on Linux, `patchelf` tool should be installed
or built binary may not be able to find their dependencies.

To run the automatic build, please clone the repository and run the following command:

```bash
cmake -P cmake/build_standalone.cmake
```

After the build, you can find NNVISR and dependency libraries under `artifact`
folder.

## Manual build

You can also build NNVISR manually. Be aware that on Windows, if you plan to
place dependency DLLs in `PATH` rather than along with NNVISR, you should use
`altsearchpath=True` option when loading NNVISR in VapourSynth.

### Dependencies

#### Linux

You can follow the instructions on
[CUDA website](https://developer.nvidia.com/cuda-downloads)
to install CUDA Toolkit on your system.
You can also install from your distribution's default repository,
given it provides a recent enough version of CUDA Toolkit (>= 12.0).

If you are on Ubuntu or RHEL, you can also
install TensorRT packages from CUDA repository for TensorRT,
or individual components required by NNVISR:

```
libnvinfer-dev libnvonnxparsers-dev
```

For other systems, you can download TensorRT from
[TensorRT Website](https://developer.nvidia.com/tensorrt).

# Windows

Follow the instructions on
[CUDA website](https://developer.nvidia.com/cuda-downloads)
to install CUDA Toolkit on your system, and download
TensorRT from [TensorRT Website](https://developer.nvidia.com/tensorrt).

# Build

NNVISR uses CMake to build. You can clone the repo and run the following
commands to build NNVISR:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target vs-nnvisr
```

The NNVISR plugin executable `vs-nnvisr.dll` or `libvs-nnvisr.so` should be
available under build folder.

If CMake is unable to find CUDA Toolkit on your system,
you can add the definition

```
-DCUDAToolkit_ROOT=/location/to/your/cudatoolkit/installation
```

To tell CMake the location of your CUDAToolkit installation.
Similarly, you can use `-DTensorRT_ROOT` and `-DVapourSynth_ROOT` to tell
CMake the location of your TensorRT and VapourSynth installation.

NNVISR supports NVIDIA GPUs starting from compute capability 6.1 (Pascal).
By default, NNVISR will build CUDA code for all supported architectures.
You can build only for your GPU architecture by adding
`-DCMAKE_CUDA_ARCHITECTURES=xy` option where `x.y`
is the compute capability of your GPU.
You can look up the compute capability of your GPU on
[NVIDIA Website](https://developer.nvidia.com/cuda-gpus).
