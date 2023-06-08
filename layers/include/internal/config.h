#pragma once

#include "md_view.h"
#include "utils.h"

struct DCNLayerInternal {
  int32_t data_type;

  // N, Cin, Hin, Win
  shape_t<4> input;
  // N, deformable_groups, Hker, Wker, 2, Hout, Wout
  shape_t<7> offset;
  // N, deformable_groups, Hker, Wker, Hout, Wout
  shape_t<6> mask;
  // Cout, Cin, Hker, Wker
  shape_t<4> weight;
  // Cout
  shape_t<1> bias;

  // N, Cout, Hout, Wout
  shape_t<4> output;

  // N, Cin, Hker, Wker, Hout, Wout
  shape_t<6> im2col_buffer;
};

struct DCNLayerConfig {
  hw<> stride;
  hw<> padding;
  hw<> dilation;
  int32_t deformable_groups;
  int32_t activation_type;
  float alpha, beta;
};

template<class F>
struct DCNLayerInput {
  md_view<const F, 4> input;
  // offset_h, offset_w
  md_view<const F, 7> offset;
  md_view<const F, 6> mask;
  md_view<const F, 4> weight;
  md_view<const F, 1> bias;

  md_view<F, 6> im2col_buffer;
};

template<class F>
struct DCNLayerOutput {
  md_view<F, 4> output;
};

struct DCNLayerExtra {
  void* cublasHandle;
  int blasMode;
};
