#pragma once

#include "PageAlignedAllocator.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>

enum class DType : uint8_t { FP32, BF16 };

// ── Stack-allocated shape (no heap alloc, no vector, no dynamic memory) ──
// LLM tensors have at most 5 dimensions (B, S, H, I, V).
struct Shape {
    size_t dims[5] = {};
    uint8_t ndim = 0;

    size_t operator[](size_t i) const { return dims[i]; }
    size_t& operator[](size_t i) { return dims[i]; }
    size_t size() const { return ndim; }
    bool empty() const { return ndim == 0; }
    size_t back() const { return dims[ndim - 1]; }

    // For existing APIs expecting std::vector<size_t>
    operator std::vector<size_t>() const {
        return std::vector<size_t>(dims, dims + ndim);
    }

    bool operator==(const Shape& o) const {
        if (ndim != o.ndim) return false;
        for (uint8_t i = 0; i < ndim; ++i)
            if (dims[i] != o.dims[i]) return false;
        return true;
    }
    bool operator!=(const Shape& o) const { return !(*this == o); }
    bool operator==(const std::vector<size_t>& v) const {
        if (ndim != v.size()) return false;
        for (uint8_t i = 0; i < ndim; ++i)
            if (dims[i] != v[i]) return false;
        return true;
    }
    bool operator!=(const std::vector<size_t>& v) const { return !(*this == v); }
    friend bool operator==(const std::vector<size_t>& v, const Shape& s) { return s == v; }
    friend bool operator!=(const std::vector<size_t>& v, const Shape& s) { return !(v == s); }

    const size_t* begin() const { return dims; }
    size_t* begin() { return dims; }
    const size_t* end() const { return dims + ndim; }
    size_t* end() { return dims + ndim; }

    void assign(const size_t* d, uint8_t n) {
        ndim = n;
        for (uint8_t i = 0; i < n; ++i) dims[i] = d[i];
    }
    void assign(const std::vector<size_t>& v) {
        assign(v.data(), (uint8_t)v.size());
    }

    // Constructors
    Shape() = default;
    Shape(const size_t* d, uint8_t n) { assign(d, n); }
    Shape(const std::vector<size_t>& v) { assign(v); }
    Shape(std::initializer_list<size_t> il) { assign(il.begin(), (uint8_t)il.size()); }
};

// ── Page-aligned buffer (replaces std::vector to eliminate value‑initialization) ──
class PagedBuffer {
  float* ptr_ = nullptr;
  size_t size_ = 0;
  size_t cap_ = 0;
public:
  PagedBuffer() = default;
  ~PagedBuffer() { if (ptr_) std::free(ptr_); }

  PagedBuffer(const PagedBuffer& other) : PagedBuffer() {
    if (other.size_ > 0) resize_store(other.size_, other.ptr_);
  }
  PagedBuffer& operator=(const PagedBuffer& other) {
    if (this != &other) {
      if (other.size_ > 0) resize_store(other.size_, other.ptr_);
      else clear();
    }
    return *this;
  }
  PagedBuffer(PagedBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), cap_(other.cap_) {
    other.ptr_ = nullptr; other.size_ = 0; other.cap_ = 0;
  }
  PagedBuffer& operator=(PagedBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) std::free(ptr_);
      ptr_ = other.ptr_; size_ = other.size_; cap_ = other.cap_;
      other.ptr_ = nullptr; other.size_ = 0; other.cap_ = 0;
    }
    return *this;
  }

  void resize_fill(size_t n, float val) {
    if (n > cap_) grow(n);
    if (val == 0.0f) { if (n > size_) std::fill(ptr_ + size_, ptr_ + n, 0.0f); }
    else { std::fill(ptr_, ptr_ + n, val); }
    size_ = n;
  }
  void resize_raw(size_t n) { if (n > cap_) grow(n); size_ = n; }
  void resize_store(size_t n, const float* src) {
    if (n > cap_) grow(n);
    if (src) std::copy(src, src + n, ptr_);
    size_ = n;
  }
  void reshape(size_t n) { if (n > cap_) grow(n); size_ = n; }

  float* data() { return ptr_; }
  const float* data() const { return ptr_; }
  float* begin() { return ptr_; }
  const float* begin() const { return ptr_; }
  float* end() { return ptr_ + size_; }
  const float* end() const { return ptr_ + size_; }
  size_t size() const { return size_; }
  size_t capacity() const { return cap_; }
  bool empty() const { return size_ == 0; }
  void clear() { size_ = 0; }

  float& operator[](size_t i) { return ptr_[i]; }
  const float& operator[](size_t i) const { return ptr_[i]; }

private:
  void grow(size_t n) {
    float* new_ptr;
    if (posix_memalign((void**)&new_ptr, 16384, n * sizeof(float)) != 0)
      throw std::bad_alloc();
    if (ptr_) { std::copy(ptr_, ptr_ + size_, new_ptr); std::free(ptr_); }
    ptr_ = new_ptr;
    cap_ = n;
  }
};

class Tensor {
public:
  Tensor();
  explicit Tensor(const Shape &shape, DType dtype = DType::FP32);
  Tensor(const Shape &shape, float val, DType dtype = DType::FP32);
  Tensor(const Shape &shape, const PagedBuffer &data, DType dtype = DType::FP32);
  Tensor(const Shape &shape, const std::vector<float> &data, DType dtype = DType::FP32);
  // Accept std::vector<size_t> via implicit conversion to Shape
  // (Shape has a constructor from std::vector<size_t> and from initializer_list)

  DType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  // strides as raw pointer (for internal use) and as vector view (for tests)
  const size_t* strides_data() const { return strides_.data(); }
  std::vector<size_t> strides() const { return std::vector<size_t>(strides_.begin(), strides_.begin() + shape_.ndim); }
  size_t size() const { return buf_ ? buf_->size() : 0; }

  // FP32 data access
  const PagedBuffer &data() const { return *buf_; }
  PagedBuffer &data() { return *buf_; }

  // Raw pointer for GPU dispatch
  const void *raw_ptr() const { return buf_ ? (const void*)buf_->data() : nullptr; }
  void *raw_ptr() { return buf_ ? (void*)buf_->data() : nullptr; }
  size_t raw_bytes() const { return buf_ ? buf_->size() * sizeof(float) : 0; }
  size_t num_elements() const { return buf_ ? buf_->size() : 0; }

  // ── Zero-copy view semantics ──
  // Tensor copies are shallow (shared_ptr refcount bump, no data copy).
  // Call clone() when an independent copy is required.
  Tensor clone() const {
    Tensor t(shape_);
    if (buf_) t.buf_->resize_store(buf_->size(), buf_->data());
    return t;
  }

  // ── Persistence API (no alloc, no zero if capacity suffices) ──
  void resize_storage(const Shape &new_shape) {
    size_t n = 1;
    for (uint8_t i = 0; i < new_shape.ndim; ++i) n *= new_shape.dims[i];
    if (!buf_) buf_ = std::make_shared<PagedBuffer>();
    buf_->reshape(n);
    shape_ = new_shape;
    compute_strides();
  }

  size_t get_index(const std::vector<size_t> &indices) const;

  float &operator()(size_t i) { return (*buf_)[i]; }
  const float &operator()(size_t i) const { return (*buf_)[i]; }
  float &operator()(size_t i, size_t j) { return (*buf_)[i * strides_[0] + j]; }
  const float &operator()(size_t i, size_t j) const { return (*buf_)[i * strides_[0] + j]; }
  float &operator()(size_t i, size_t j, size_t k) { return (*buf_)[i * strides_[0] + j * strides_[1] + k]; }
  const float &operator()(size_t i, size_t j, size_t k) const { return (*buf_)[i * strides_[0] + j * strides_[1] + k]; }
  float &operator()(size_t i, size_t j, size_t k, size_t l) { return (*buf_)[i * strides_[0] + j * strides_[1] + k * strides_[2] + l]; }
  const float &operator()(size_t i, size_t j, size_t k, size_t l) const { return (*buf_)[i * strides_[0] + j * strides_[1] + k * strides_[2] + l]; }
  // Vector-index operator (used by tests)
  float &operator()(const std::vector<size_t> &indices) { return (*buf_)[get_index(indices)]; }
  const float &operator()(const std::vector<size_t> &indices) const { return (*buf_)[get_index(indices)]; }

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
  std::shared_ptr<PagedBuffer> buf_;       // shared FP32 storage — zero-copy views
  DType dtype_ = DType::FP32;

  static size_t compute_size(const Shape &s) {
    if (s.ndim == 0) return 0;
    size_t n = 1;
    for (uint8_t i = 0; i < s.ndim; ++i) n *= s.dims[i];
    return n;
  }
  void compute_strides();
  // Ensure buf_ is allocated (lazy init for default-constructed Tensors)
  void ensure_buf(size_t n) {
    if (!buf_) buf_ = std::make_shared<PagedBuffer>();
    buf_->resize_raw(n);
  }
};
