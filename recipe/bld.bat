mkdir build
cd build

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONDA_BUILD_TWEAK=ON ..
if %ERRORLEVEL% neq 0 exit 1
ninja vs-nnvisr
if %ERRORLEVEL% neq 0 exit 1

mkdir %PREFIX%\vapoursynth64\plugins
copy vs-nnvisr.dll %PREFIX%\vapoursynth64\plugins\vs-nnvisr.dll
