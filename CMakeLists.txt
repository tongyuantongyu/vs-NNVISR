cmake_minimum_required(VERSION 3.18)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
        set(CMAKE_CUDA_ARCHITECTURES 53 62 72 87)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89 90)
    endif()
endif()

project(Cycmunet-TRT LANGUAGES CXX CUDA)

enable_testing()

option(CUDA_DEVICE_DEBUG "Enable device debug" OFF)
if(CUDA_DEVICE_DEBUG)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/source-charset:utf-8 /execution-charset:us-ascii /wd4996>")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8$<SEMICOLON>/execution-charset:us-ascii$<SEMICOLON>/wd4996>)
endif()

find_package(CUDAToolkit 12.0 REQUIRED COMPONENTS cublas)
find_package(TensorRT 8.6 REQUIRED COMPONENTS OnnxParser)
find_package(VapourSynth 4 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(COMMON_HEADERS include/helper.h include/md_view.h include/utils.h include/logging.h)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(COMMON_HEADERS ${COMMON_HEADERS} include/debug/reveal.h)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/debug)
endif()

add_library(_dcn_layer_impl OBJECT
            ${COMMON_HEADERS}
            layers/include/internal/config.h

            layers/include/internal/dcn_layer_impl.h
            layers/impl/dcn_layer.cu)
target_include_directories(_dcn_layer_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/include/internal)
target_link_libraries(_dcn_layer_impl CUDA::cublas)

option(AUTO_REGISTER_PLUGIN "Automatically register plugin on load of shared library" OFF)
if(AUTO_REGISTER_PLUGIN)
    add_library(trt_layers_plugin SHARED
                ${COMMON_HEADERS}
                layers/include/internal/config.h

                layers/include/internal/dcn_layer.h
                layers/include/internal/dcn_layer_impl.h
                layers/src/dcn_layer.cpp

                layers/src/layers.cpp)
    target_compile_definitions(trt_layers_plugin PRIVATE AUTO_REGISTER_PLUGIN)
else()
    add_library(trt_layers_plugin STATIC
                ${COMMON_HEADERS}
                layers/include/internal/config.h

                layers/include/internal/dcn_layer.h
                layers/include/internal/dcn_layer_impl.h
                layers/src/dcn_layer.cpp

                layers/src/layers.cpp)
endif()

target_compile_definitions(trt_layers_plugin PRIVATE BUILDING_PLUGIN)
target_include_directories(trt_layers_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/include/internal)
target_include_directories(trt_layers_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/layers/include)
target_link_libraries(trt_layers_plugin PUBLIC _dcn_layer_impl TensorRT::NvInfer)
set_target_properties(trt_layers_plugin PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

add_library(nnvisr STATIC include/config.h include/optimize.h include/inference.h src/optimize.cpp src/inference.cpp)
target_link_libraries(nnvisr PUBLIC trt_layers_plugin TensorRT::OnnxParser TensorRT::NvInfer CUDA::cudart)
set_target_properties(nnvisr PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

add_library(reformat_cuda OBJECT include/reformat.h src/reformat.cu)

add_library(vs-nnvisr MODULE src/vs-plugin.cpp)
target_link_libraries(vs-nnvisr PRIVATE nnvisr reformat_cuda VapourSynth)
set_target_properties(vs-nnvisr PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden)

option(CONDA_BUILD_TWEAK "Tweak build options for releasing as conda package." OFF)
if(MSVC AND CONDA_BUILD_TWEAK)
    target_compile_definitions(vs-nnvisr PRIVATE IS_CONDA_BUILD)
    target_link_libraries(vs-nnvisr PUBLIC delayimp)
    target_link_options(vs-nnvisr PUBLIC
            /DELAYLOAD:cudart64_12.dll
            /DELAYLOAD:cublas64_12.dll
            /DELAYLOAD:$<TARGET_FILE_NAME:TensorRT::NvInfer>
            /DELAYLOAD:$<TARGET_FILE_NAME:TensorRT::OnnxParser>
    )
endif()

install(TARGETS vs-nnvisr LIBRARY DESTINATION ".")
