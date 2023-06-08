#include "reformat.h"
#include <cuda_fp16.h>
#include <type_traits>

half __device__ round(half f) {
  const half v0_5 = float(0.5);
  return hfloor(f + v0_5);
}

template<class F, class U>
static void __global__ fma_from(md_view<F, 2> dst, md_view<const U, 2> src, F a, F b) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  auto [src_h, src_w] = src.shape;
  uint32_t src_x = dst_x >= src_w ? src_w - 1 : dst_x;
  uint32_t src_y = dst_y >= src_h ? src_h - 1 : dst_y;

  F value = static_cast<F>(src.at(src_y, src_x));
  value = a * value + b;
  dst.at(dst_y, dst_x) = value;
}

template<class F, class U>
static void __global__ fma_to(md_view<U, 2> dst, md_view<const F, 2> src, F a, F b, F min, F max) {
  uint32_t dst_x = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t dst_y = threadIdx.y + blockDim.y * blockIdx.y;

  auto [dst_h, dst_w] = dst.shape;
  if (dst_x >= dst_w || dst_y >= dst_h) {
    return;
  }

  F value = static_cast<F>(src.at(dst_y, dst_x));
  value = a * value + b;
  if constexpr (std::is_integral_v<U>) {
    value = round(value);
  }

  if (value < min) {
    value = min;
  }
  else if (value > max) {
    value = max;
  }

  if constexpr (std::is_integral_v<U> && sizeof(U) == 1) {
    dst.at(dst_y, dst_x) = static_cast<U>(static_cast<int16_t>(value));
  } else {
    dst.at(dst_y, dst_x) = static_cast<U>(value);
  }
}

template<class F, class U>
void import_pixel(md_view<F, 2> dst, md_view<const U, 2> src, float a, float b, cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [dst_h, dst_w] = dst.shape;
  dimGrid.x = (dst_w + 31) / 32;
  dimGrid.y = (dst_h + 31) / 32;

  fma_from<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b));
}

template void import_pixel<float, uint8_t>(md_view<float, 2> dst, md_view<const uint8_t, 2> src, float a, float b,
                                           cudaStream_t stream);
template void import_pixel<half, uint8_t>(md_view<half, 2> dst, md_view<const uint8_t, 2> src, float a, float b,
                                          cudaStream_t stream);
template void import_pixel<float, uint16_t>(md_view<float, 2> dst, md_view<const uint16_t, 2> src, float a, float b,
                                            cudaStream_t stream);
template void import_pixel<half, uint16_t>(md_view<half, 2> dst, md_view<const uint16_t, 2> src, float a, float b,
                                           cudaStream_t stream);
template void import_pixel<float, half>(md_view<float, 2> dst, md_view<const half, 2> src, float a, float b,
                                        cudaStream_t stream);
template void import_pixel<half, half>(md_view<half, 2> dst, md_view<const half, 2> src, float a, float b,
                                       cudaStream_t stream);
template void import_pixel<float, float>(md_view<float, 2> dst, md_view<const float, 2> src, float a, float b,
                                         cudaStream_t stream);
template void import_pixel<half, float>(md_view<half, 2> dst, md_view<const float, 2> src, float a, float b,
                                        cudaStream_t stream);

template<class F, class U>
void export_pixel(md_view<U, 2> dst, md_view<const F, 2> src, float a, float b, float min, float max, cudaStream_t stream) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  auto [dst_h, dst_w] = dst.shape;
  dimGrid.x = (dst_w + 31) / 32;
  dimGrid.y = (dst_h + 31) / 32;

  fma_to<<<dimGrid, dimBlock, 0, stream>>>(dst, src, F(a), F(b), F(min), F(max));
}

template void export_pixel<float, uint8_t>(md_view<uint8_t, 2> dst, md_view<const float, 2> src, float a, float b, float min,
                                    float max, cudaStream_t stream);
template void export_pixel<half, uint8_t>(md_view<uint8_t, 2> dst, md_view<const half, 2> src, float a, float b, float min,
                                   float max, cudaStream_t stream);
template void export_pixel<float, uint16_t>(md_view<uint16_t, 2> dst, md_view<const float, 2> src, float a, float b, float min,
                                     float max, cudaStream_t stream);
template void export_pixel<half, uint16_t>(md_view<uint16_t, 2> dst, md_view<const half, 2> src, float a, float b, float min,
                                    float max, cudaStream_t stream);
template void export_pixel<float, half>(md_view<half, 2> dst, md_view<const float, 2> src, float a, float b, float min,
                                    float max, cudaStream_t stream);
template void export_pixel<half, half>(md_view<half, 2> dst, md_view<const half, 2> src, float a, float b, float min,
                                   float max, cudaStream_t stream);
template void export_pixel<float, float>(md_view<float, 2> dst, md_view<const float, 2> src, float a, float b, float min,
                                     float max, cudaStream_t stream);
template void export_pixel<half, float>(md_view<float, 2> dst, md_view<const half, 2> src, float a, float b, float min,
                                    float max, cudaStream_t stream);