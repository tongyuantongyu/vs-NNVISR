# Installation

NNVISR depends on TensorRT, so it can be installed on systems
supported by TensorRT: Windows x64, Linux x64 and aarch64.

## Install using conda

Install using conda is the recommended method if you are on Windows x64
or Linux x64 platform, and you already have a conda installation on your system.

To create a new environment and install NNVISR, run:

```
conda create -n nnvisr -c conda-forge -c tongyuantongyu --overide-channels vapoursynth-nnvisr
```

You can also install the VapourSynth FFMS2 plugin `vapoursynth-ffms2` to load
video files, and MVTools plugin `vapoursynth-mvtools` to detect scene changes
from my channel:

```
conda install -c conda-forge -c tongyuantongyu --override-channels vapoursynth-ffms2 vapoursynth-mvtools
```

Note that NNVISR depends on the conda-forge channel which is **INCOMPATIBLE**
with Anaconda's default channel. If you are going to install NNVISR into
an existing environment, please make sure the environment is using packages
from conda-forge channel or installation may fail or does not work properly.

## Manual Install

Alternatively, you can manually install NNVISR into your existing
VapourSynth installation on Windows. Prebuilt binary for Windows x64
is provided in Release. Follow the VapourSynth documentation for how to load
NNVISR plugin. For Linux, you have to
[build NNVISR](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/build.md) yourself.

NNVISR depends on TensorRT 8.6 and CUDA 12 (cudart and cuBLAS components).
You can either download and install TensorRT and CUDA,
and make sure their DLL libraries are in PATH,
or place the DLL libraries along with NNVISR.
The dependency DLLs can also be downloaded in Release.
NNVISR does not use cuDNN, but TensorRT may try to load cuDNN and fail
if you don't have cuDNN installed.
cuDNN can be downloaded at [cuDNN website](https://developer.nvidia.com/cudnn).
