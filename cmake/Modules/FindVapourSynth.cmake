find_path(VapourSynth_INCLUDE_DIR
          NAMES VapourSynth4.h
          PATH_SUFFIXES vapoursynth)


function(_vapoursynth_get_version)
    unset(VapourSynth_VERSION_STRING PARENT_SCOPE)
    set(_hdr_file "${VapourSynth_INCLUDE_DIR}/VapourSynth4.h")

    if(NOT EXISTS "${_hdr_file}")
        return()
    endif()

    file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define VAPOURSYNTH_API_.*")

    foreach(TYPE MAJOR MINOR)
        string(REGEX MATCH "VAPOURSYNTH_API_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
        string(REGEX MATCH "[0-9]" VapourSynth_VERSION_${TYPE} ${TRT_TYPE_STRING})
    endforeach(TYPE)

    set(VapourSynth_VERSION_MAJOR ${VapourSynth_VERSION_MAJOR} PARENT_SCOPE)
    set(VapourSynth_VERSION_MINOR ${VapourSynth_VERSION_MINOR} PARENT_SCOPE)

    set(VapourSynth_VERSION_STRING "${VapourSynth_VERSION_MAJOR}.${VapourSynth_VERSION_MINOR}" PARENT_SCOPE)
endfunction(_vapoursynth_get_version)

_vapoursynth_get_version()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VapourSynth
                                  FOUND_VAR VapourSynth_FOUND
                                  REQUIRED_VARS VapourSynth_INCLUDE_DIR
                                  VERSION_VAR VapourSynth_VERSION_STRING
                                  HANDLE_COMPONENTS)

add_library(VapourSynth INTERFACE IMPORTED)
target_include_directories(VapourSynth SYSTEM INTERFACE "${VapourSynth_INCLUDE_DIR}")

mark_as_advanced(VapourSynth_INCLUDE_DIR)
