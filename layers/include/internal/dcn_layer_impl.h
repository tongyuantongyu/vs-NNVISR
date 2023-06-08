#pragma once

#include <cuda_runtime_api.h>
#include "config.h"

template<class F>
void compute(DCNLayerInput<F> inputs,
             DCNLayerOutput<F> outputs,
             DCNLayerConfig config,
             DCNLayerExtra extra,
             cudaStream_t stream);

