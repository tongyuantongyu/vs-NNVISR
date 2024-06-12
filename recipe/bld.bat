mkdir build
cd build

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONDA_BUILD_TWEAK=ON .. -DCMAKE_CUDA_ARCHITECTURES=all
if %ERRORLEVEL% neq 0 exit 1
ninja vs-nnvisr
if %ERRORLEVEL% neq 0 exit 1

mkdir %PREFIX%\vs-plugins
copy vs-nnvisr.dll %PREFIX%\vs-plugins\vs-nnvisr.dll
