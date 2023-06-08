#pragma once

#include <fstream>
#include <memory>
#include "cuda_runtime_api.h"
#include "md_view.h"

template<class T, std::size_t D>
void dump_value(md_view<T, D> t, std::wstring name) {
  auto size = t.size();
  auto host_pointer = std::make_unique<T[]>(size);
  cudaMemcpy(host_pointer.get(), t.data, size * sizeof(T), cudaMemcpyDeviceToHost);
  auto h = host_pointer.get();

  // Now feel free to examine host_pointer (or through h)
  std::ofstream p(name, std::ios::binary);
  p.write((const char*)(h), size * sizeof(T));
  p.close();
}

template<class T>
void dump_value(const T* t, size_t size, std::wstring name) {
  auto host_pointer = std::make_unique<T[]>(size);
  cudaMemcpy(host_pointer.get(), t, size * sizeof(T), cudaMemcpyDeviceToHost);
  auto h = host_pointer.get();

  // Now feel free to examine host_pointer (or through h)
  std::ofstream p(name, std::ios::binary);
  p.write((const char*)(h), size * sizeof(T));
  p.close();
}

template<class T, std::size_t D>
void debug_me_show_memory(md_view<T, D> t) {
  using _T = typename std::decay<T>::type;
  auto size = t.size();
  auto host_pointer = std::make_unique<_T[]>(size);
  cudaMemcpy(host_pointer.get(), t.data, size * sizeof(_T), cudaMemcpyDeviceToHost);
  auto h = host_pointer.get();

  // Now feel free to examine host_pointer (or through h)
  return;
}
