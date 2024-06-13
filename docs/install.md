# Installation

NNVISR can run on Windows x64, Linux x64 and aarch64.

## Install using conda

Install using conda is the recommended method if you are on Windows x64
or Linux x64 platform, and you already have a conda installation on your system.

To create a new environment named `nnvisr` and install NNVISR, run:

```
conda create -n nnvisr -c conda-forge -c tongyuantongyu --overide-channels vapoursynth-nnvisr
```

You can also install the VapourSynth BestSource plugin `vapoursynth-bestsource`
to load video files, and MVTools plugin `vapoursynth-mvtools` to
detect scene changes:

```
conda install -c conda-forge -c tongyuantongyu --override-channels vapoursynth-bestsource vapoursynth-mvtools
```

**WARNING:** NNVISR depends on packages on conda-forge channel, which is **INCOMPATIBLE**
with Anaconda's default channel. If you are going to install NNVISR into
an existing environment, please make sure the environment is using packages
from conda-forge channel or installation may fail or does not work properly.

## Manual Install

On Windows, alternatively, you can manually install NNVISR into your existing
VapourSynth installation. Prebuilt binary for Windows x64
is provided in
[Release](https://github.com/tongyuantongyu/vs-NNVISR/releases).
Follow the VapourSynth documentation for how to load NNVISR plugin.

For Linux, you have to [build NNVISR](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/build.md) yourself.

NNVISR depends on TensorRT version 8.6 or 10.0 and CUDA 12 (cudart and cuBLAS components).
For manual installation you have to place the DLL libraries along with NNVISR.
Additionally, for TensorRT 8.6, NNVISR may use cuDNN depending on the network.

Dependencies for TensorRT 8.6 version can be downloaded at [here](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/deps.windows.trt8.6.7z).
Dependencies for TensorRT 10.0 version can be downloaded at [here](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/deps.windows.trt10.0.7z).

You can also download libraries from their official source:

`TensorRT` can be downloaded at
[TensorRT Download page](https://developer.nvidia.com/tensorrt/download).

`cudart` can be downloaded at [cudart release page](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/).

`cuBLAS` can be downloaded at [cuBLAS release page](https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/).
Some version of cuBLAS requires nvrtc, which can be downloaded at
[nvrtc release page](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/).

cuDNN can be downloaded at
[cuDNN release page](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/).
cuDNN depends on `zlib`, namely `zlibwapi.dll`, which can be downloaded [here](http://www.winimage.com/zLibDll/zlib123dllx64.zip).
