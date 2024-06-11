#ifndef MDVIEW_H_
#define MDVIEW_H_

#include <cstdint>
#include <cstring>
#include <cassert>
#include <utility>
#include "helper.h"

typedef int64_t offset_t;

template<class T, std::size_t DIMS>
struct simple_array {
  T _d[DIMS] {};

  constexpr util_attrs T &operator[](std::size_t i) {
    return this->_d[i];
  }

  constexpr util_attrs const T &operator[](std::size_t i) const {
    return this->_d[i];
  }

  constexpr util_attrs bool operator==(const simple_array<T, DIMS>& o) const {
    for (std::size_t i = 0; i < DIMS; ++i) {
      if (this->_d[i] != o._d[i]) {
        return false;
      }
    }

    return true;
  }

  constexpr util_attrs bool operator!=(const simple_array<T, DIMS>& o) const {
    return !(*this == o);
  }

  constexpr util_attrs T *begin() {
    return this->_d + 0;
  }

  constexpr util_attrs T *end() {
    return this->_d + DIMS;
  }

  constexpr util_attrs const T *begin() const {
    return this->_d + 0;
  }

  constexpr util_attrs const T *end() const {
    return this->_d + DIMS;
  }

  constexpr util_attrs const T *cbegin() const {
    return begin();
  }

  constexpr util_attrs const T *cend() const {
    return end();
  }

  template<size_t Idx>
  constexpr util_attrs T &get() noexcept {
    return this->_d[Idx];
  }

  template<size_t Idx>
  constexpr util_attrs const T &get() const noexcept {
    return this->_d[Idx];
  }

  template<std::size_t begin, std::size_t count>
  constexpr util_attrs std::enable_if_t<((begin + count) <= DIMS), simple_array<T, count>> slice() const {
    return this->_slice<begin>(std::make_index_sequence<count> {});
  }

  template<size_t... Idx>
  constexpr util_attrs simple_array<T, sizeof...(Idx)> gather() const {
    return {(*this)[Idx]...};
  }

  template<class A, class... IdxT>
  constexpr util_attrs std::enable_if_t<sizeof...(IdxT) == DIMS> gather_from(A arr, IdxT... idx) {
    *this = {(static_cast<T>(arr[idx]))...};
  };

 private:
  template<std::size_t begin, std::size_t... I>
  constexpr util_attrs std::enable_if_t<(begin + sizeof...(I)) <= DIMS, simple_array<T, sizeof...(I)>>
  _slice(std::index_sequence<I...>) const {
    return {(*this)[begin + I]...};
  }
};

namespace std {
template<class T, size_t DIMS>
struct tuple_size<simple_array < T, DIMS>> : integral_constant<size_t, DIMS> {
};

template<size_t Idx, class T, size_t DIMS>
struct tuple_element<Idx, simple_array<T, DIMS>> {
  using type = T;
};
}

template<std::size_t DIMS = 1, typename = std::enable_if_t<(DIMS != 0)>>
struct stride_t : simple_array<offset_t, DIMS> {
  using _base = simple_array<offset_t, DIMS>;

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) <= DIMS, offset_t> offset(Tp... indexes) const {
    simple_array<offset_t, DIMS> offsets {static_cast<offset_t>(indexes)...};

    offset_t offset = 0;
    for (std::size_t i = 0; i < DIMS; ++i) {
      offset += (*this)[i] * offsets[i];
    }

    return offset;
  }

  constexpr util_attrs simple_array<offset_t, DIMS> indexes(offset_t offset) const {
    simple_array<offset_t, DIMS> indexes;
    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      assert((*this)[dim] != 0);
      indexes[dim] = offset / (*this)[dim];
      offset = offset % (*this)[dim];
    }

    return indexes;
  }

  template<std::size_t begin, std::size_t count>
  constexpr util_attrs stride_t<count> slice() const {
    return {_base::template slice<begin, count>()};
  }
};

template<class... T>
stride_t(T...) -> stride_t<sizeof...(T)>;

template<std::size_t DIMS = 1, typename = std::enable_if_t<(DIMS != 0)>>
struct shape_t : simple_array<offset_t, DIMS> {
  using _base = simple_array<offset_t, DIMS>;

  template<size_t SDIMS, typename = std::enable_if_t<SDIMS <= DIMS>>
  constexpr util_attrs offset_t offset(const shape_t<SDIMS> &offsets) const {
    offset_t offset = 0;
    for (std::size_t i = 0; i < DIMS; ++i) {
      offset = offset * (*this)[i];
      if (i < SDIMS) {
        offset += offsets[i];
      }
    }

    return offset;
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) <= DIMS, offset_t> offset(Tp... indexes) const {
    return offset<sizeof...(Tp)>({static_cast<offset_t>(indexes)...});
  }

  constexpr util_attrs simple_array<offset_t, DIMS> indexes(offset_t offset) const {
    simple_array<offset_t, DIMS> indexes;
    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      auto pos = DIMS - dim - 1;
      indexes[pos] = offset % (*this)[pos];
      offset /= (*this)[pos];
    }

    return indexes;
  }

  constexpr util_attrs offset_t count() const {
    offset_t size = 1;
    for (const auto &s: *this) {
      size *= s;
    }

    return size;
  }

  template<std::size_t begin, std::size_t count>
  constexpr util_attrs shape_t<count> slice() const {
    return {_base::template slice<begin, count>()};
  }

  template<std::size_t ...Idx>
  constexpr util_attrs shape_t<sizeof...(Idx)> gather() const {
    return {_base::template gather<Idx...>()};
  }

  constexpr util_attrs stride_t<DIMS> stride() const {
    stride_t<DIMS> stride;
    offset_t current = 1;

    for (std::size_t dim = 0; dim < DIMS; ++dim) {
      auto pos = DIMS - dim - 1;
      stride[pos] = current;
      current *= (*this)[pos];
    }

    return stride;
  }
};

template<class... T>
shape_t(T...) -> shape_t<sizeof...(T)>;

namespace std {
template<size_t DIMS>
struct tuple_size<shape_t < DIMS>> : integral_constant<size_t, DIMS> {
};

template<size_t Idx, size_t DIMS>
struct tuple_element<Idx, shape_t<DIMS>> {
  using type = offset_t;
};
}

template<class T_, std::size_t DIMS = 1, typename = std::enable_if_t<DIMS != 0>>
struct md_view;

template<class T_, std::size_t DIMS = 1, typename = std::enable_if_t<DIMS != 0>>
struct md_uview;

template<class T_, std::size_t DIMS>
constexpr md_uview<T_, DIMS> util_attrs to_uview(md_view<T_, DIMS>);

template<class T_, std::size_t DIMS>
constexpr md_view<T_, DIMS> util_attrs to_view(md_uview<T_, DIMS>);

template<class T_, std::size_t DIMS, typename>
struct md_view {
  using T = std::remove_reference_t<T_>;
  constexpr static std::size_t D = DIMS;

  T *data;
  shape_t<DIMS> shape;

  template<class CT = T, typename = std::enable_if_t<!std::is_const_v<CT>>>
  constexpr util_attrs operator md_view<const CT, DIMS>() const {
    return {data, shape};
  }

  constexpr util_attrs operator md_uview<T, DIMS>() const { return this->as_uview(); }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) < DIMS, md_view<T, DIMS - sizeof...(Tp)>>
  at(Tp... indexes) const noexcept {
    ptrdiff_t offset = shape.offset(indexes...);

    auto sub_span = md_view<T, DIMS - sizeof...(Tp)> {data + offset};
    for (std::size_t i = sizeof...(Tp); i < DIMS; ++i) {
      sub_span.shape[i - sizeof...(Tp)] = shape[i];
    }
    return sub_span;
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) == DIMS, T &> at(Tp... indexes) const noexcept {
    return data[shape.offset(indexes...)];
  }

template<class... Tp>
constexpr util_attrs std::enable_if_t<sizeof...(Tp) == DIMS, T *> ptr(Tp... indexes) const noexcept {
    return &at(indexes...);
}

  template<size_t SDIMS, typename = std::enable_if_t<SDIMS<DIMS>> constexpr util_attrs md_view<T, DIMS - SDIMS> at(
                             const shape_t<SDIMS> &offsets) const noexcept {
    ptrdiff_t offset = shape.offset(offsets);

    auto sub_span = md_view<T, DIMS - SDIMS> {data + offset};
    for (std::size_t i = SDIMS; i < DIMS; ++i) {
      sub_span.shape[i - SDIMS] = shape[i];
    }
    return sub_span;
  }

  template<size_t SDIMS, typename = std::enable_if_t<SDIMS == DIMS>>
  constexpr util_attrs T &at(const shape_t<SDIMS> &offsets) const noexcept {
    return data[shape.offset(offsets)];
  }

  template<std::size_t N_DIMS>
  constexpr util_attrs md_view<T, N_DIMS> reshape(shape_t<N_DIMS> new_shape) const {
    return {this->data, new_shape};
  }

  template<class T2>
  constexpr util_attrs md_view<T2, DIMS> reinterpret() const {
    shape_t new_shape = this->shape;
    new_shape[DIMS - 1] = new_shape[DIMS - 1] * sizeof(T) / sizeof(T2);
    assert(this->shape[DIMS - 1] * sizeof(T) == new_shape[DIMS - 1] * sizeof(T2));
    return {reinterpret_cast<T2>(this->data), new_shape};
  }

  template<class T2, std::size_t N_DIMS>
  constexpr util_attrs md_view<T2, DIMS> reinterpret(shape_t<N_DIMS> new_shape) const {
    return {reinterpret_cast<T2>(this->data), new_shape};
  }

  [[nodiscard]] constexpr util_attrs offset_t size() const noexcept {
    return this->shape.count();
  }

  constexpr util_attrs md_uview<T, DIMS> as_uview() const noexcept {
    return to_uview(*this);
  }
};

template<class T, class ...Tp>
md_view(T *t, Tp &&...shape)
-> md_view<std::enable_if_t<(std::is_convertible_v<Tp, offset_t> &&...), T>, sizeof...(shape)>;

template<class T, std::size_t DIMS>
md_view(T *t, offset_t (&&shape)[DIMS]) -> md_view<T, DIMS>;

template<class T, std::size_t DIMS>
md_view(T *t, shape_t<DIMS> shape) -> md_view<T, DIMS>;

template<class T_, std::size_t DIMS, typename>
struct md_uview {
  using T = std::remove_reference_t<T_>;
  constexpr static std::size_t D = DIMS;

  T *data;
  shape_t<DIMS> shape;
  stride_t<DIMS> stride;

  template<class CT = T, typename = std::enable_if_t<!std::is_const_v<CT>>>
  constexpr util_attrs operator md_uview<const CT, DIMS>() const {
    return {data, shape, stride};
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) < DIMS, md_uview<T, DIMS - sizeof...(Tp)>>
  at(Tp... indexes) const noexcept {
    ptrdiff_t offset = stride.offset(indexes...);

    auto sub_span = md_uview<T, DIMS - sizeof...(Tp)> {data + offset};
    for (std::size_t i = sizeof...(Tp); i < DIMS; ++i) {
      sub_span.shape[i - sizeof...(Tp)] = shape[i];
      sub_span.stride[i - sizeof...(Tp)] = stride[i];
    }
    return sub_span;
  }

  template<class... Tp>
  constexpr util_attrs std::enable_if_t<sizeof...(Tp) == DIMS, T &> at(Tp... indexes) const noexcept {
    return data[stride.offset(indexes...)];
  }

  [[nodiscard]] constexpr util_attrs offset_t size() const noexcept { return this->shape.count(); }

  template<std::size_t pos, typename = std::enable_if_t<pos >= 0 && pos<DIMS>> constexpr util_attrs md_uview<T, DIMS>
                                slice(offset_t begin = 0, offset_t end = 0) const {
    begin = begin < 0 ? begin + this->shape[pos] : begin;
    end = end <= 0 ? end + this->shape[pos] : end;
    assert(begin < end);

    md_uview result = *this;
    result.data += this->stride[pos] * begin;
    result.shape[pos] = end - begin;
    return result;
  }

  constexpr util_attrs bool is_contiguous() const {
    return this->shape.stride() == this->stride;
  }

  template<std::size_t N_DIMS>
  constexpr util_attrs md_uview<T, N_DIMS> reshape(shape_t<N_DIMS> new_shape) const {
    return {this->data, new_shape, this->stride};
  }

  template<class T2>
  constexpr util_attrs md_uview<T2, DIMS> reinterpret() const {
    shape_t new_shape = this->shape;
    new_shape[DIMS - 1] = new_shape[DIMS - 1] * sizeof(T) / sizeof(T2);
    assert(this->shape[DIMS - 1] * sizeof(T) == new_shape[DIMS - 1] * sizeof(T2));

    stride_t new_stride = this->stride;
    for (std::size_t i = 0; i < DIMS; ++i) {
      new_stride[i] = new_stride[i] * sizeof(T) / sizeof(T2);
      assert(this->stride[DIMS - 1] * sizeof(T) == new_stride[DIMS - 1] * sizeof(T2));
    }

    return {reinterpret_cast<T2>(this->data), new_shape, new_stride};
  }

  constexpr util_attrs md_view<T, DIMS> as_view() const noexcept {
    assert(this->is_contiguous());
    return to_view(*this);
  }
};

template<class T_, std::size_t DIMS>
constexpr md_uview<T_, DIMS> util_attrs to_uview(md_view<T_, DIMS> v) {
  return {
      v.data,
      v.shape,
      v.shape.stride()
  };
}

template<class T_, std::size_t DIMS>
constexpr md_view<T_, DIMS> util_attrs to_view(md_uview<T_, DIMS> uv) {
  return {
      uv.data,
      uv.shape
  };
}

template<class T, std::size_t DIMS>
void util_attrs copy(const md_view<T, DIMS> &dst, const md_view<const T, DIMS> &src) {
  assert(dst.shape == src.shape);
  std::memcpy(dst.data, src.data, dst.size() * sizeof(T));
}

template<class T, std::size_t DIMS>
void util_attrs copy(const md_view<T, DIMS> &dst, const md_view<T, DIMS> &src) {
  md_view<const T, DIMS> csrc = src;
  copy(dst, csrc);
}

template<class T, std::size_t DIMS>
void util_attrs copy_impl(const md_uview<T, DIMS> &dst, const md_uview<const T, DIMS> &src) {
  if (dst.at(0).is_contiguous() && src.at(0).is_contiguous()) {
    for (int i = 0; i < dst.shape[0]; ++i) {
      copy(dst.at(i).as_view(), src.at(i).as_view());
    }
  }
  else {
    for (int i = 0; i < dst.shape[0]; ++i) {
      copy_impl(dst.at(i), src.at(i));
    }
  }
}

template<class T>
void util_attrs copy_impl(const md_uview<T, 1> &dst, const md_uview<const T, 1> &src) {
  for (int i = 0; i < dst.shape[0]; ++i) {
    dst.at(i) = src.at(i);
  }
}

template<class T, std::size_t DIMS>
void util_attrs copy(const md_uview<T, DIMS> &dst, const md_uview<const T, DIMS> &src) {
  shape_t<DIMS> min_shape;
  for (std::size_t i = 0; i < DIMS; ++i) {
    min_shape[i] = std::min(dst.shape[i], src.shape[i]);
  }

  copy_impl(dst.reshape(min_shape), src.reshape(min_shape));
}

template<class T, std::size_t DIMS>
void util_attrs copy(const md_uview<T, DIMS> &dst, const md_uview<T, DIMS> &src) {
  md_uview<const T, DIMS> csrc = src;
  copy(dst, csrc);
}

#include <string>
#include <sstream>
#include <ios>
#include <iomanip>

template<std::size_t DIMS>
std::string describe(const shape_t<DIMS> &view) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < DIMS; ++i) {
    ss << view[i];
    if (i + 1 != DIMS) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

template<class T, std::size_t DIMS>
std::string describe(const md_view<T, DIMS> &view) {
  std::stringstream ss;
  ss << std::internal << std::hex << std::setw(16) << std::setfill('0') << (void*)(view.data);
  ss << "-" << std::setw(16) << (void*)((uint8_t*)(view.data) + view.size() * sizeof(T));
  ss << std::resetiosflags(ss.basefield);
  ss << "(" << view.size() * sizeof(T);
  ss << ", ";
  ss << describe(view.shape);
  ss << ")";
  return ss.str();
}

#endif //MDVIEW_H_