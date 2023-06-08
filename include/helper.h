#pragma once

#if defined(_WIN32)
#if defined(BUILDING_PLUGIN)
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#if defined(BUILDING_PLUGIN)
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define PLUGIN_EXPORT
#endif
#else
#define PLUGIN_EXPORT
#endif

#if defined(_MSC_VER)
#define PLUGIN_UNREACHABLE __assume(false)
#elif defined(__GNUC__)
#define PLUGIN_UNREACHABLE __builtin_unreachable()
#else
#define PLUGIN_UNREACHABLE void(0)
#endif


#if defined(__CUDACC__)
#define util_attrs __host__ __device__ inline
#else
#define util_attrs inline
#endif
