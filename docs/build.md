# Build

Building NNVISR requires a compiler supporting C++20 and CUDA Compiler >= 12.

## Dependencies

### Linux

You can follow the instructions on
[CUDA website](https://developer.nvidia.com/cuda-downloads)
to install the complete CUDAToolkit on your system.
You can also install from your distribution's default repository,
given it provides a recent enough version of CUDAToolkit.
If you prefer a minimal installation with only components required to
build and run NNVISR, you can set up CUDA repository by choosing the
network installer type, but instead of the all-in-one cuda package,
install these packages:

```
cuda-compiler-12-1 cuda-cudart-dev-12-1 libcublas-dev-12-1
```

If you are on Ubuntu 18.04, 20.04, 22.04, RHEL 7 or 8, you can also
install tensorrt packages from CUDA repository for TensorRT,
or individual components required by NNVISR:

```
libnvinfer-dev libnvonnxparsers-dev
```

For other systems, you can download TensorRT from
[TensorRT Website](https://developer.nvidia.com/tensorrt).

# Windows

Follow the instructions on
[CUDA website](https://developer.nvidia.com/cuda-downloads)
to install CUDAToolkit on your system.

Download TensorRT from [TensorRT Website](https://developer.nvidia.com/tensorrt).

# Build

NNVISR uses CMake to build. You can clone the repo and run the following
commands to build NNVISR:

```bash
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target vs-nnvisr
```

The NNVISR plugin executable `vs-nnvisr.dll` or `libvs-nnvisr.so` should be
available under build folder.

If CMake is unable to find CUDAToolkit on your system,
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
