# Build Conda Package

NNVISR can be built using `conda-build`.

## Install `conda-build`

Follow the [Getting Started](https://docs.conda.io/projects/conda-build/en/stable/user-guide/getting-started.html)
guide to set up `conda-build`.

## Build TensorRT development package

NVIDIA does not provide conda package for TensorRT, and TensorRT EULA
prohibit distribution of its development files, so you have to build
TensorRT development conda package yourself. The conda recipe for TensorRT
is available at
[TensorRT-conda-recipe](https://github.com/tongyuantongyu/TensorRT-conda-recipe).
Follow the instructions to build the `libnvinfer-dev` conda package in your
local conda channel.




