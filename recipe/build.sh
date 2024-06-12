#!/bin/bash
set -ex

mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=$PREFIX/targets/x86_64-linux-gnu -DCONDA_BUILD_TWEAK=ON .. -DCMAKE_CUDA_ARCHITECTURES=all
ninja vs-nnvisr

mkdir -p $PREFIX/lib/vapoursynth
cp libvs-nnvisr.so $PREFIX/lib/vapoursynth/libvs-nnvisr.so
