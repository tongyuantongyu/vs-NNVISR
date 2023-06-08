#pragma once

#include <cstdint>
#include "helper.h"

template<class T = int32_t>
struct hw {
  T h;
  T w;

  template<class U>
  constexpr util_attrs operator hw<U>() const noexcept {
    return {static_cast<U>(h), static_cast<U>(w)};
  }

  constexpr util_attrs hw<T> operator+(const hw<T>& o) const noexcept {
    return {h + o.h, w + o.w};
  }

  constexpr util_attrs hw<T> operator+(const T& o) const noexcept {
    return {h + o, w + o};
  }

  constexpr util_attrs hw<T>& operator+=(const hw<T>& o) noexcept {
    *this = {h + o.h, w + o.w};
    return *this;
  }

  constexpr util_attrs hw<T>& operator+=(const T& o) noexcept {
    *this = {h + o, w + o};
    return *this;
  }

  constexpr util_attrs hw<T> operator-(const hw<T>& o) const noexcept {
    return {h - o.h, w - o.w};
  }

  constexpr util_attrs hw<T> operator-(const T& o) const noexcept {
    return {h - o, w - o};
  }

  constexpr util_attrs hw<T> operator-() const noexcept {
    return {-h, -w};
  }
};
