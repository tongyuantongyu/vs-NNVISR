# Build Conda Package

NNVISR can be built using `conda-build`.

## Install `conda-build`

Follow the [Getting Started](https://docs.conda.io/projects/conda-build/en/stable/user-guide/getting-started.html)
guide to set up `conda-build`.

NNVISR depends on packages from `conda-forge` channel. Following the
[`conda-forge` documentation](https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge)
to set up `conda-forge` channel.

## Build TensorRT development package

NVIDIA does not provide conda package for TensorRT, and TensorRT EULA
prohibit distribution of its development files, so you have to build
TensorRT development conda package yourself. The conda recipe for TensorRT
is available at
[TensorRT-conda-recipe](https://github.com/tongyuantongyu/TensorRT-conda-recipe).
Follow the instructions to build the `libnvinfer-dev` conda package in your
local conda channel.

### Clone and build

Run the following command to start build process on Linux:

```
conda build -c conda-forge -c local <path-to-cloned-repository>
```

CUDA packages on `conda-forge` channel is currently incomplete on Windows,
so you need to add NVIDIA's conda channel as well:

```
conda build -c conda-forge -c nvidia -c local <path-to-cloned-repository>
```

The build process will take a while. After it finished, you will have
`vapoursynth-nnvisr` package available in your `local` conda channel.
Now you can run:

```
conda install -c conda-forge -c local vapoursynth-nnvisr
```

on Linux or

```
conda install -c conda-forge -c nvidia -c local vapoursynth-nnvisr
```

on Windows to install your just built `vapoursynth-nnvisr` package.
