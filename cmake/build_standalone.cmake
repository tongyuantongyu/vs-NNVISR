cmake_minimum_required(VERSION 3.18)

set(CUDA_CUDART_VERSION 12.5.39)
set(CUDA_CCCL_VERSION 12.5.39)
set(CUDA_NVCC_VERSION 12.5.40)
set(CUDA_CUBLAS_VERSION 12.5.2.13)
set(CUDA_NVRTC_VERSION 12.5.40)
set(CUDNN_VERSION 8.9.7.29)
set(VAPOURSYNTH_VERSION 68)
set(TENSORRT_VERSION 8.6)

if (WIN32)
    set(CUDA_ARCH windows-x86_64)
    set(CMAKE_EXECUTABLE_SUFFIX .exe)
    set(CUDA_ARCHIVE_SUFFIX zip)
else ()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CUDA_ARCH linux-${CMAKE_SYSTEM_PROCESSOR})
    set(CMAKE_EXECUTABLE_SUFFIX)
    set(CUDA_ARCHIVE_SUFFIX tar.xz)
endif ()

include(FetchContent)

FetchContent_Populate(
        cuda-cudart
        URL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/${CUDA_ARCH}/cuda_cudart-${CUDA_ARCH}-${CUDA_CUDART_VERSION}-archive.${CUDA_ARCHIVE_SUFFIX}"
        SOURCE_DIR build_dependencies/_cuda-cudart
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)
FetchContent_Populate(
        cuda-cccl
        URL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/${CUDA_ARCH}/cuda_cccl-${CUDA_ARCH}-${CUDA_CCCL_VERSION}-archive.${CUDA_ARCHIVE_SUFFIX}"
        SOURCE_DIR build_dependencies/_cuda-cccl
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)
FetchContent_Populate(
        cuda-nvcc
        URL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/${CUDA_ARCH}/cuda_nvcc-${CUDA_ARCH}-${CUDA_NVCC_VERSION}-archive.${CUDA_ARCHIVE_SUFFIX}"
        SOURCE_DIR build_dependencies/_cuda-nvcc
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)
FetchContent_Populate(
        cuda-nvrtc
        URL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/${CUDA_ARCH}/cuda_nvrtc-${CUDA_ARCH}-${CUDA_NVRTC_VERSION}-archive.${CUDA_ARCHIVE_SUFFIX}"
        SOURCE_DIR build_dependencies/_cuda-nvrtc
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)
FetchContent_Populate(
        libcublas
        URL "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/${CUDA_ARCH}/libcublas-${CUDA_ARCH}-${CUDA_CUBLAS_VERSION}-archive.${CUDA_ARCHIVE_SUFFIX}"
        SOURCE_DIR build_dependencies/_libcublas
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)

execute_process(COMMAND cmake -E make_directory build_dependencies/cuda)
execute_process(COMMAND cmake -E copy_directory build_dependencies/_cuda-cudart build_dependencies/cuda)
execute_process(COMMAND cmake -E copy_directory build_dependencies/_cuda-cccl build_dependencies/cuda)
execute_process(COMMAND cmake -E copy_directory build_dependencies/_cuda-nvcc build_dependencies/cuda)
execute_process(COMMAND cmake -E copy_directory build_dependencies/_cuda-nvrtc build_dependencies/cuda)
execute_process(COMMAND cmake -E copy_directory build_dependencies/_libcublas build_dependencies/cuda)
execute_process(COMMAND cmake -E create_symlink lib lib64 WORKING_DIRECTORY build_dependencies/cuda)

if (${TENSORRT_VERSION} STREQUAL 8.6)
    if (WIN32)
        set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip")
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
            set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz")
        else ()
            set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Ubuntu-20.04.aarch64-gnu.cuda-12.0.tar.gz")
        endif ()
    endif ()

    FetchContent_Populate(
            cudnn
            URL "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${CUDA_ARCH}/cudnn-${CUDA_ARCH}-${CUDNN_VERSION}_cuda12-archive.${CUDA_ARCHIVE_SUFFIX}"
            SOURCE_DIR build_dependencies/_cudnn
            BINARY_DIR build_dependencies/cache
            SUBBUILD_DIR build_dependencies/cache
    )
    execute_process(COMMAND cmake -E copy_directory build_dependencies/_cudnn build_dependencies/cuda)

    if (WIN32)
        FetchContent_Populate(
                zlibwapi
                URL "http://www.winimage.com/zLibDll/zlib123dllx64.zip"
                SOURCE_DIR build_dependencies/_zlibwapi
                BINARY_DIR build_dependencies/cache
                SUBBUILD_DIR build_dependencies/cache
        )
        execute_process(COMMAND cmake -E copy build_dependencies/_zlibwapi/dll_x64/zlibwapi.dll build_dependencies/cuda/bin/zlibwapi.dll)
    endif ()
else ()
    if (WIN32)
        set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-12.4.zip")
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
            set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL sbsa)
            set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Ubuntu-22.04.aarch64-gnu.cuda-12.4.tar.gz")
        else ()
            set(TENSORRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.l4t.aarch64-gnu.cuda-12.4.tar.gz")
        endif ()
    endif ()
endif ()

FetchContent_Populate(
        tensorrt
        URL ${TENSORRT_URL}
        SOURCE_DIR build_dependencies/tensorrt
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)

FetchContent_Populate(
        vapoursynth
        URL "https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R${VAPOURSYNTH_VERSION}.tar.gz"
        SOURCE_DIR build_dependencies/vapoursynth
        BINARY_DIR build_dependencies/cache
        SUBBUILD_DIR build_dependencies/cache
)

set(CMAKE_GENERATOR_CMD)
find_program(NINJA_PROGRAM ninja)
if (NINJA_PROGRAM)
    set(CMAKE_GENERATOR_CMD -G;Ninja;-DCMAKE_MAKE_PROGRAM=${NINJA_PROGRAM})
endif ()

execute_process(
        COMMAND cmake -B ${CMAKE_SOURCE_DIR}/build -S ${CMAKE_SOURCE_DIR} ${CMAKE_GENERATOR_CMD}
        "-DCUDAToolkit_ROOT=${CMAKE_SOURCE_DIR}/build_dependencies/cuda"
        "-DCMAKE_CUDA_COMPILER=${CMAKE_SOURCE_DIR}/build_dependencies/cuda/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}"
        "-DTensorRT_ROOT=${CMAKE_SOURCE_DIR}/build_dependencies/tensorrt"
        "-DVapourSynth_ROOT=${CMAKE_SOURCE_DIR}/build_dependencies/vapoursynth"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_INSTALL_RPATH=$ORIGIN"
)

execute_process(COMMAND cmake --build ${CMAKE_SOURCE_DIR}/build -t vs-nnvisr --config Release)
execute_process(COMMAND cmake --install ${CMAKE_SOURCE_DIR}/build --prefix ${CMAKE_SOURCE_DIR}/artifact --config Release)

if (WIN32)
    file(GLOB DEPENDENCY_LIBRARIES "${CMAKE_SOURCE_DIR}/build_dependencies/cuda/bin/*.dll")
    file(GLOB DEPENDENCY_LIBRARIES_1 "${CMAKE_SOURCE_DIR}/build_dependencies/tensorrt/lib/*.dll")
    list(APPEND DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES_1})
else ()
    file(GLOB DEPENDENCY_LIBRARIES "${CMAKE_SOURCE_DIR}/build_dependencies/cuda/lib/*.so.12")
    file(GLOB DEPENDENCY_LIBRARIES_1 "${CMAKE_SOURCE_DIR}/build_dependencies/cuda/lib/*.so.8")
    list(APPEND DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES_1})
    file(GLOB DEPENDENCY_LIBRARIES_1 "${CMAKE_SOURCE_DIR}/build_dependencies/tensorrt/lib/*.so.*")
    list(APPEND DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES_1})
endif ()

file(COPY ${DEPENDENCY_LIBRARIES} DESTINATION ${CMAKE_SOURCE_DIR}/artifact)

if (NOT WIN32 AND ${TENSORRT_VERSION} STREQUAL 8.6)
    find_program(PATCHELF_PROGRAM patchelf)
    if (PATCHELF_PROGRAM)
        file(GLOB TRT_ONNXPARSER "${CMAKE_SOURCE_DIR}/artifact/libnvonnxparser.so.*.*.*")
        file(GLOB TRT_PLUGIN "${CMAKE_SOURCE_DIR}/artifact/libnvinfer_plugin.so.*.*.*")
        execute_process(
                COMMAND ${PATCHELF_PROGRAM} --add-rpath "$ORIGIN" ${TRT_ONNXPARSER}
                COMMAND ${PATCHELF_PROGRAM} --add-rpath "$ORIGIN" ${TRT_PLUGIN}
        )
    else()
        message(WARNING "Can't find patchelf tool, unable to make dependency relocatable")
    endif()
endif()
