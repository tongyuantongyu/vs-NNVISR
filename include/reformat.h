//
// Created by TYTY on 2023-01-04 004.
//

#ifndef CYCMUNET_TRT_INCLUDE_REFORMAT_H_
#define CYCMUNET_TRT_INCLUDE_REFORMAT_H_

#include "md_view.h"

template<class F, class U>
void import_pixel(md_view<F, 2> dst, md_view<const U, 2> src, float a, float b, cudaStream_t stream);

template<class F, class U>
void export_pixel(md_view<U, 2> dst, md_view<const F, 2> src, float a, float b, float min, float max,
                  cudaStream_t stream);

#endif//CYCMUNET_TRT_INCLUDE_REFORMAT_H_
