find_path(TensorRT_INCLUDE_DIR
        NAMES NvInfer.h)

find_library(TensorRT_LIBRARY
        NAMES nvinfer)

if (TensorRT_LIBRARY)
    set(TensorRT_LIBRARIES
            ${TensorRT_LIBRARIES}
            ${TensorRT_LIBRARY})
endif (TensorRT_LIBRARY)

function(_tensorrt_get_version)
    unset(TensorRT_VERSION_STRING PARENT_SCOPE)
    set(_hdr_file "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")

    if (NOT EXISTS "${_hdr_file}")
        return()
    endif ()

    file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

    foreach(TYPE MAJOR MINOR PATCH BUILD)
        string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
        string(REGEX MATCH "[0-9]" TensorRT_VERSION_${TYPE} ${TRT_TYPE_STRING})
    endforeach(TYPE)

    set(TensorRT_VERSION_MAJOR ${TensorRT_VERSION_MAJOR} PARENT_SCOPE)
    set(TensorRT_VERSION_MINOR ${TensorRT_VERSION_MINOR} PARENT_SCOPE)
    set(TensorRT_VERSION_PATCH ${TensorRT_VERSION_PATCH} PARENT_SCOPE)
    set(TensorRT_VERSION_BUILD ${TensorRT_VERSION_BUILD} PARENT_SCOPE)

    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}" PARENT_SCOPE)
endfunction(_tensorrt_get_version)

_tensorrt_get_version()

if(TensorRT_FIND_COMPONENTS)
    list(REMOVE_ITEM TensorRT_FIND_COMPONENTS "nvinfer")

    if ("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(TensorRT_OnnxParser_INCLUDE_DIR
                NAMES NvOnnxParser.h)

        find_library(TensorRT_OnnxParser_LIBRARY
                NAMES nvonnxparser)
        if (TensorRT_OnnxParser_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES
                    ${TensorRT_LIBRARIES}
                    ${TensorRT_OnnxParser_LIBRARY})
            set(TensorRT_OnnxParser_FOUND TRUE)
        endif ()
    endif()

    if ("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(TensorRT_Plugin_INCLUDE_DIR
                NAMES NvInferPlugin.h)

        find_library(TensorRT_Plugin_LIBRARY
                NAMES nvinfer_plugin)

        if (TensorRT_Plugin_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES
                    ${TensorRT_LIBRARIES}
                    ${TensorRT_Plugin_LIBRARY})
            set(TensorRT_Plugin_FOUND TRUE)
        endif ()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
        FOUND_VAR TensorRT_FOUND
        REQUIRED_VARS TensorRT_LIBRARY TensorRT_LIBRARIES TensorRT_INCLUDE_DIR
        VERSION_VAR TensorRT_VERSION_STRING
        HANDLE_COMPONENTS)

add_library(TensorRT::NvInfer UNKNOWN IMPORTED)
target_include_directories(TensorRT::NvInfer SYSTEM INTERFACE "${TensorRT_INCLUDE_DIR}")
set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")

if ("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::OnnxParser UNKNOWN IMPORTED)
    target_include_directories(TensorRT::OnnxParser SYSTEM INTERFACE "${TensorRT_OnnxParser_INCLUDE_DIR}")
    target_link_libraries(TensorRT::OnnxParser INTERFACE TensorRT::NvInfer)
    set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_LIBRARY}")
endif()

if ("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::Plugin UNKNOWN IMPORTED)
    target_include_directories(TensorRT::Plugin SYSTEM INTERFACE "${TensorRT_Plugin_INCLUDE_DIR}")
    target_link_libraries(TensorRT::Plugin INTERFACE TensorRT::NvInfer)
    set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION "${TensorRT_Plugin_LIBRARY}")
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_LIBRARIES)