# Installation

NNVISR depends on TensorRT, so it can be installed on systems
supported by TensorRT: Windows x64, Linux x64 and aarch64.

## Install using conda

Install using conda is the recommended method if you are on Windows x64
or Linux x64 platform, and you already have a conda installation on your system.

To create a new environment named `nnvisr` and install NNVISR on Linux, run:

```
conda create -n nnvisr -c conda-forge -c tongyuantongyu --overide-channels vapoursynth-nnvisr
```

CUDA packages on `conda-forge` channel is currently incomplete on Windows,
so you need to add NVIDIA's conda channel as well:

```
conda create -n nnvisr -c nvidia -c conda-forge -c tongyuantongyu --overide-channels vapoursynth-nnvisr
```

You can also install the VapourSynth BestSource plugin `vapoursynth-bestsource`
to load video files, and MVTools plugin `vapoursynth-mvtools` to
detect scene changes from my channel:

```
conda install -c conda-forge -c tongyuantongyu --override-channels vapoursynth-bestsource vapoursynth-mvtools
```

Note that NNVISR depends on the conda-forge channel which is **INCOMPATIBLE**
with Anaconda's default channel. If you are going to install NNVISR into
an existing environment, please make sure the environment is using packages
from conda-forge channel or installation may fail or does not work properly.

## Manual Install

Alternatively, you can manually install NNVISR into your existing
VapourSynth installation on Windows. Prebuilt binary for Windows x64
is provided in
[Release](https://github.com/tongyuantongyu/vs-NNVISR/releases).
Follow the VapourSynth documentation for how to load
NNVISR plugin. For Linux, you have to
[build NNVISR](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/build.md) yourself.

NNVISR depends on TensorRT 8.6 and CUDA 12 (cudart and cuBLAS components).
You can either download and install TensorRT and CUDA,
and make sure their DLL libraries are in PATH,
or place the DLL libraries along with NNVISR.
Additionally, NNVISR may use cuDNN depending on the network.

The TensorRT runtime DLLs for NNVISR can be downloaded at
[Release](https://github.com/tongyuantongyu/vs-NNVISR/releases). 

The cudart component of CUDA can be downloaded from
[cudart release page](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.1.105-archive.zip).

The cuBLAS component of CUDA can be downloaded from
[cuBLAS release page](https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.1.3.1-archive.zip).
Some version of cuBLAS need nvrtc, which can be downloaded from
[nvrtc release page](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.1.105-archive.zip).

cuDNN can be downloaded from
[cuDNN release page](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.2.26_cuda12-archive.zip).
