#pragma once

#include "PageAlignedAllocator.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

enum class DType : uint8_t { FP32, BF16 };

inline uint16_t float_to_bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return uint16_t(bits >> 16);
}

inline float bf16_to_float(uint16_t bf) {
  uint32_t bits = uint32_t(bf) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline size_t itemsize(DType d) { return d == DType::BF16 ? 2 : 4; }

struct Shape {
  size_t dims[5] = {};
  uint8_t ndim = 0;

  size_t operator[](size_t i) const { return dims[i]; }
  size_t& operator[](size_t i) { return dims[i]; }
  size_t size() const { return ndim; }
  size_t num_elements() const { size_t n = 1; for (uint8_t i = 0; i < ndim; ++i) n *= dims[i]; return n; }
  bool empty() const { return ndim == 0; }
  size_t back() const { return dims[ndim ? ndim - 1 : 0]; }

  bool operator==(const Shape& o) const {
    if (ndim != o.ndim) return false;
    for (uint8_t i = 0; i < ndim; ++i)
      if (dims[i] != o.dims[i]) return false;
    return true;
  }
  bool operator!=(const Shape& o) const { return !(*this == o); }

  bool operator==(const std::vector<size_t>& v) const {
    if (ndim != v.size()) return false;
    for (uint8_t i = 0; i < ndim; ++i) if (dims[i] != v[i]) return false;
    return true;
  }
  bool operator!=(const std::vector<size_t>& v) const { return !(*this == v); }
  friend bool operator==(const std::vector<size_t>& v, const Shape& s) { return s == v; }
  friend bool operator!=(const std::vector<size_t>& v, const Shape& s) { return !(v == s); }

  Shape& operator=(const std::vector<size_t>& v) { assign(v); return *this; }
  operator std::vector<size_t>() const { return std::vector<size_t>(dims, dims + ndim); }

  const size_t* begin() const { return dims; }
  size_t* begin() { return dims; }
  const size_t* end() const { return dims + ndim; }
  size_t* end() { return dims + ndim; }

  void assign(const size_t* d, uint8_t n) {
    ndim = n;
    for (uint8_t i = 0; i < n; ++i) dims[i] = d[i];
  }
  void assign(const std::vector<size_t>& v) { assign(v.data(), uint8_t(v.size())); }

  Shape() = default;
  Shape(const size_t* d, uint8_t n) { assign(d, n); }
  Shape(const std::vector<size_t>& v) { assign(v); }
  Shape(std::initializer_list<size_t> il) { assign(il.begin(), uint8_t(il.size())); }
};

class PagedBuffer {
  char* data_ = nullptr;
  size_t bytes_ = 0;
  size_t cap_bytes_ = 0;
  void* gpu_wrapper_ = nullptr;

public:
  PagedBuffer() = default;
  ~PagedBuffer() { release_gpu(); if (data_) std::free(data_); }

  PagedBuffer(PagedBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), cap_bytes_(other.cap_bytes_), gpu_wrapper_(other.gpu_wrapper_) {
    other.data_ = nullptr; other.bytes_ = 0; other.cap_bytes_ = 0; other.gpu_wrapper_ = nullptr;
    if (data_) metal_bridge::register_gpu_wrapper((float*)data_, &gpu_wrapper_);
  }
  PagedBuffer& operator=(PagedBuffer&& other) noexcept {
    if (this != &other) {
      release_gpu(); if (data_) std::free(data_);
      data_ = other.data_; bytes_ = other.bytes_; cap_bytes_ = other.cap_bytes_; gpu_wrapper_ = other.gpu_wrapper_;
      other.data_ = nullptr; other.bytes_ = 0; other.cap_bytes_ = 0; other.gpu_wrapper_ = nullptr;
      if (data_) metal_bridge::register_gpu_wrapper((float*)data_, &gpu_wrapper_);
    }
    return *this;
  }

  PagedBuffer(const PagedBuffer& other) {
    if (other.bytes_ > 0) { grow(other.bytes_); std::memcpy(data_, other.data_, other.bytes_); bytes_ = other.bytes_; }
  }
  PagedBuffer& operator=(const PagedBuffer& other) {
    if (this != &other) {
      release_gpu(); if (other.bytes_ > 0) { grow(other.bytes_); std::memcpy(data_, other.data_, other.bytes_); bytes_ = other.bytes_; }
      else clear();
    }
    return *this;
  }

  void* gpu_wrapper() const { return gpu_wrapper_; }
  void set_gpu_wrapper(void* w) { gpu_wrapper_ = w; }
  static void (*gpu_release_fn)(void*);
  void release_gpu() {
    if (gpu_wrapper_ && gpu_release_fn) { gpu_release_fn(gpu_wrapper_); gpu_wrapper_ = nullptr; }
    if (data_) metal_bridge::unregister_gpu_wrapper((float*)data_);
  }

  void* data() { return data_; }
  const void* data() const { return data_; }
  float* data_float() { return (float*)data_; }
  const float* data_float() const { return (const float*)data_; }
  size_t bytes() const { return bytes_; }
  bool empty() const { return bytes_ == 0; }
  void clear() { bytes_ = 0; }
  size_t num_elements(size_t elem_size) const { return elem_size > 0 ? bytes_ / elem_size : 0; }

  void resize_raw(size_t num_elements, size_t elem_size) {
    size_t need = num_elements * elem_size;
    if (need > cap_bytes_) grow(need);
    bytes_ = need;
  }

  void resize_fill_f32(size_t n, float val) {
    size_t need = n * sizeof(float);
    if (need > cap_bytes_) grow(need);
    std::fill(data_float(), data_float() + n, val);
    bytes_ = need;
  }

  void resize_store_f32(size_t n, const float* src) {
    size_t need = n * sizeof(float);
    if (need > cap_bytes_) grow(need);
    if (src) std::copy(src, src + n, data_float());
    bytes_ = need;
  }

  void reshape(size_t num_elements, size_t elem_size) {
    size_t need = num_elements * elem_size;
    if (need > cap_bytes_) grow(need);
    bytes_ = need;
  }

private:
  void grow(size_t need_bytes) {
    char* new_ptr;
    if (posix_memalign((void**)&new_ptr, 16384, need_bytes) != 0)
      throw std::bad_alloc();
    if (data_) {
      std::memcpy(new_ptr, data_, bytes_ < need_bytes ? bytes_ : need_bytes);
      metal_bridge::unregister_gpu_wrapper((float*)data_);
      std::free(data_);
    }
    data_ = new_ptr;
    cap_bytes_ = need_bytes;
    metal_bridge::register_gpu_wrapper((float*)data_, &gpu_wrapper_);
  }
};

class Tensor {
public:
  Tensor();
  explicit Tensor(const Shape &shape, DType dtype = DType::FP32);
  Tensor(const Shape &shape, float val, DType dtype = DType::FP32);
  Tensor(const Shape &shape, const std::vector<float> &data, DType dtype = DType::FP32);

  DType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  const size_t* strides_data() const { return strides_.data(); }
  std::vector<size_t> strides() const { return std::vector<size_t>(strides_.begin(), strides_.begin() + shape_.ndim); }
  size_t itemsize() const { return ::itemsize(dtype_); }
  size_t size() const { return shape_.num_elements(); }
  size_t num_elements() const { return shape_.num_elements(); }

  float* data() { return buf_ ? buf_->data_float() : nullptr; }
  const float* data() const { return buf_ ? buf_->data_float() : nullptr; }
  void* data_bf16() { return buf_ ? buf_->data() : nullptr; }
  const void* data_bf16() const { return buf_ ? buf_->data() : nullptr; }

  const void *raw_ptr() const { return buf_ ? buf_->data() : nullptr; }
  void *raw_ptr() { return buf_ ? buf_->data() : nullptr; }
  size_t raw_bytes() const { return buf_ ? buf_->bytes() : 0; }

  Tensor clone() const;
  Tensor to_dtype(DType target_dtype) const;

  void resize_storage(const Shape &new_shape) {
    size_t n = compute_size(new_shape);
    if (!buf_) buf_ = std::make_shared<PagedBuffer>();
    buf_->reshape(n, itemsize());
    shape_ = new_shape;
    compute_strides();
  }

  size_t get_index(const std::vector<size_t> &indices) const;

  float &operator()(size_t i);
  const float &operator()(size_t i) const;
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;
  float &operator()(size_t i, size_t j, size_t k);
  const float &operator()(size_t i, size_t j, size_t k) const;
  float &operator()(size_t i, size_t j, size_t k, size_t l);
  const float &operator()(size_t i, size_t j, size_t k, size_t l) const;
  float &operator()(const std::vector<size_t> &indices);
  const float &operator()(const std::vector<size_t> &indices) const;

  void fill(float val);
  void print(const std::string &name = "") const;

  Tensor add(const Tensor &other) const;
  Tensor mul(const Tensor &other) const;
  Tensor scale(float factor) const;
  void add_(const Tensor &other);
  void mul_(const Tensor &other);
  void scale_(float factor);

  Tensor matmul(const Tensor &other) const;
  Tensor transpose() const;
  Tensor reshape(const std::vector<size_t> &new_shape) const;

private:
  Shape shape_;
  std::array<size_t, 5> strides_{};
  std::shared_ptr<PagedBuffer> buf_;
  DType dtype_ = DType::FP32;

  static size_t compute_size(const Shape &s) {
    if (s.ndim == 0) return 0;
    size_t n = 1;
    for (uint8_t i = 0; i < s.ndim; ++i) n *= s.dims[i];
    return n;
  }
  void compute_strides();
  void ensure_buf(size_t n) {
    if (!buf_) buf_ = std::make_shared<PagedBuffer>();
    buf_->resize_raw(n, itemsize());
  }
};
