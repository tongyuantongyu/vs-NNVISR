find_path(TensorRT_INCLUDE_DIR
        NAMES NvInfer.h)

function(_tensorrt_get_version)
    unset(TensorRT_VERSION_STRING PARENT_SCOPE)
    set(_hdr_file "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")

    if (NOT EXISTS "${_hdr_file}")
        return()
    endif ()

    file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

    foreach(TYPE MAJOR MINOR PATCH BUILD)
        string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING ${VERSION_STRINGS})
        string(REGEX MATCH "[0-9]+" TensorRT_VERSION_${TYPE} ${TRT_TYPE_STRING})
    endforeach(TYPE)

    set(TensorRT_VERSION_MAJOR ${TensorRT_VERSION_MAJOR} PARENT_SCOPE)
    set(TensorRT_VERSION_MINOR ${TensorRT_VERSION_MINOR} PARENT_SCOPE)
    set(TensorRT_VERSION_PATCH ${TensorRT_VERSION_PATCH} PARENT_SCOPE)

    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}" PARENT_SCOPE)
endfunction(_tensorrt_get_version)

_tensorrt_get_version()

macro(_tensorrt_find_dll VAR)
    find_file(${VAR}
            NAMES ${ARGN}
            HINTS ${TensorRT_ROOT}
            PATH_SUFFIXES bin)
endmacro(_tensorrt_find_dll)

find_library(TensorRT_LIBRARY
        NAMES "nvinfer_${TensorRT_VERSION_MAJOR}" nvinfer)

if (WIN32)
    _tensorrt_find_dll(TensorRT_DLL
            "nvinfer_${TensorRT_VERSION_MAJOR}.dll" nvinfer.dll)
endif ()

if (TensorRT_LIBRARY)
    set(TensorRT_LIBRARIES
            ${TensorRT_LIBRARIES}
            ${TensorRT_LIBRARY})
endif (TensorRT_LIBRARY)

if(TensorRT_FIND_COMPONENTS)
    list(REMOVE_ITEM TensorRT_FIND_COMPONENTS "nvinfer")

    if ("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(TensorRT_OnnxParser_INCLUDE_DIR
                NAMES NvOnnxParser.h)

        find_library(TensorRT_OnnxParser_LIBRARY
                NAMES "nvonnxparser_${TensorRT_VERSION_MAJOR}" nvonnxparser)
        if (TensorRT_OnnxParser_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES
                    ${TensorRT_LIBRARIES}
                    ${TensorRT_OnnxParser_LIBRARY})
            set(TensorRT_OnnxParser_FOUND TRUE)
        endif ()

        if (WIN32)
            _tensorrt_find_dll(TensorRT_OnnxParser_DLL
                    "nvonnxparser_${TensorRT_VERSION_MAJOR}.dll" nvonnxparser.dll)
        endif ()
    endif()

    if ("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(TensorRT_Plugin_INCLUDE_DIR
                NAMES NvInferPlugin.h)

        find_library(TensorRT_Plugin_LIBRARY
                NAMES "nvinfer_plugin_${TensorRT_VERSION_MAJOR}" nvinfer_plugin)

        if (TensorRT_Plugin_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES
                    ${TensorRT_LIBRARIES}
                    ${TensorRT_Plugin_LIBRARY})
            set(TensorRT_Plugin_FOUND TRUE)
        endif ()

        if (WIN32)
            _tensorrt_find_dll(TensorRT_Plugin_DLL
                    "nvinfer_plugin_${TensorRT_VERSION_MAJOR}.dll" nvinfer_plugin.dll)
        endif ()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
        FOUND_VAR TensorRT_FOUND
        REQUIRED_VARS TensorRT_LIBRARY TensorRT_LIBRARIES TensorRT_INCLUDE_DIR
        VERSION_VAR TensorRT_VERSION_STRING
        HANDLE_COMPONENTS)

add_library(TensorRT::NvInfer SHARED IMPORTED)
target_include_directories(TensorRT::NvInfer SYSTEM INTERFACE "${TensorRT_INCLUDE_DIR}")
if (WIN32)
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION "${TensorRT_DLL}")
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY}")
else()
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
endif()

if ("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::OnnxParser SHARED IMPORTED)
    target_include_directories(TensorRT::OnnxParser SYSTEM INTERFACE "${TensorRT_OnnxParser_INCLUDE_DIR}")
    target_link_libraries(TensorRT::OnnxParser INTERFACE TensorRT::NvInfer)
    if (WIN32)
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_DLL}")
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_IMPLIB "${TensorRT_OnnxParser_LIBRARY}")
    else()
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_LIBRARY}")
    endif()
endif()

if ("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::Plugin SHARED IMPORTED)
    target_include_directories(TensorRT::Plugin SYSTEM INTERFACE "${TensorRT_Plugin_INCLUDE_DIR}")
    target_link_libraries(TensorRT::Plugin INTERFACE TensorRT::NvInfer)
    if (WIN32)
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION "${TensorRT_Plugin_DLL}")
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_IMPLIB "${TensorRT_Plugin_LIBRARY}")
    else()
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION "${TensorRT_Plugin_LIBRARY}")
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_LIBRARIES)