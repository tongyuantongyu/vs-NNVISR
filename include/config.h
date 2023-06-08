#pragma once

#include <cstdint>

#include "utils.h"

struct optimization_axis {
  optimization_axis(int32_t min, int32_t opt, int32_t max) : min(min), opt(opt), max(max) {}
  optimization_axis(int32_t same) : min(same), opt(same), max(same) {}
  optimization_axis() : min(0), opt(0), max(0) {}
  int32_t min, opt, max;
};

enum IOFormat {
  RGB,
  YUV420,
};

struct OptimizationConfig {
  optimization_axis input_width;
  optimization_axis input_height;

  optimization_axis batch_extract;
  optimization_axis batch_fusion;

  int32_t input_count;
  int32_t feature_count;
  int32_t extraction_layers;
  bool extra_frame;
  bool double_frame;
  bool interpolation;

  float scale_factor_w;
  float scale_factor_h;
  IOFormat format;

  bool use_fp16;
  bool low_mem;
};

struct InferenceConfig {
  int32_t input_width;
  int32_t input_height;

  int32_t batch_extract;
  int32_t batch_fusion;

  int32_t input_count;
  int32_t feature_count;
  int32_t extraction_layers;
  bool extra_frame;
  bool double_frame;
  bool interpolation;

  float scale_factor_w;
  float scale_factor_h;
  IOFormat format;

  bool use_fp16;
  bool low_mem;
};
