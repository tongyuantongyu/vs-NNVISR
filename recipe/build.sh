#!/bin/bash
set -ex

mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja vs-nnvisr

mkdir -p $PREFIX/lib/vapoursynth
cp libvs-nnvisr.so $PREFIX/lib/vapoursynth/libvs-nnvisr.so
