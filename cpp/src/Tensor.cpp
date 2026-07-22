#include "Tensor.hpp"
#define ACCELERATE_NEW_LAPACK

void (*PagedBuffer::gpu_release_fn)(void*) = nullptr;

#include "gpu_kernel/MetalBridge.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

// ── Constructors ───────────────────────────────────────────────────────────
Tensor::Tensor() : shape_(), dtype_(DType::FP32) {}

Tensor::Tensor(const Shape &shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
  size_t n = compute_size(shape);
  if (n > 0) { buf_ = std::make_shared<PagedBuffer>(); buf_->resize_raw(n, itemsize()); }
  compute_strides();
}

Tensor::Tensor(const Shape &shape, float val, DType dtype)
    : shape_(shape), dtype_(dtype) {
  size_t n = compute_size(shape);
  if (n > 0) {
    buf_ = std::make_shared<PagedBuffer>();
    buf_->resize_fill_f32(n, val);
  }
  compute_strides();
}

Tensor::Tensor(const Shape &shape, const PagedBuffer &data, DType dtype)
    : shape_(shape), dtype_(dtype) {
  buf_ = std::make_shared<PagedBuffer>(data);
  compute_strides();
}

Tensor::Tensor(const Shape &shape, const std::vector<float> &data, DType dtype)
    : shape_(shape), dtype_(dtype) {
  buf_ = std::make_shared<PagedBuffer>();
  buf_->resize_store_f32(data.size(), data.data());
  compute_strides();
}

// ── Element access (FP32 path; BF16 access via raw_ptr for hot paths) ─────
float &Tensor::operator()(size_t i) { return buf_->data_float()[i]; }
const float &Tensor::operator()(size_t i) const { return buf_->data_float()[i]; }
float &Tensor::operator()(size_t i, size_t j) { return buf_->data_float()[i * strides_[0] + j]; }
const float &Tensor::operator()(size_t i, size_t j) const { return buf_->data_float()[i * strides_[0] + j]; }
float &Tensor::operator()(size_t i, size_t j, size_t k) { return buf_->data_float()[i * strides_[0] + j * strides_[1] + k]; }
const float &Tensor::operator()(size_t i, size_t j, size_t k) const { return buf_->data_float()[i * strides_[0] + j * strides_[1] + k]; }
float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) { return buf_->data_float()[i * strides_[0] + j * strides_[1] + k * strides_[2] + l]; }
const float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const { return buf_->data_float()[i * strides_[0] + j * strides_[1] + k * strides_[2] + l]; }
float &Tensor::operator()(const std::vector<size_t> &indices) { return buf_->data_float()[get_index(indices)]; }
const float &Tensor::operator()(const std::vector<size_t> &indices) const { return buf_->data_float()[get_index(indices)]; }

// ── Other methods ─────────────────────────────────────────────────────────
void Tensor::compute_strides() {
  if (shape_.ndim == 0) return;
  size_t stride = 1;
  for (int i = static_cast<int>(shape_.ndim) - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_.dims[i];
  }
}

size_t Tensor::get_index(const std::vector<size_t> &indices) const {
  if (indices.size() != shape_.ndim)
    throw std::invalid_argument("Dimensionality mismatch");
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_.dims[i])
      throw std::out_of_range("Index out of bounds");
    idx += indices[i] * strides_[i];
  }
  return idx;
}

void Tensor::fill(const float val) {
  if (!buf_) return;
  if (dtype_ == DType::BF16) {
    // For BF16, fill with the 16-bit truncated value
    uint16_t bf = float_to_bf16(val);
    uint16_t* ptr = (uint16_t*)buf_->data();
    size_t n = buf_->bytes() / 2;
    std::fill(ptr, ptr + n, bf);
  } else {
    if (val == 0.0f && buf_->size_float() > 0)
      buf_->resize_fill_f32(buf_->size_float(), 0.0f);
    else
      std::fill(buf_->data_float(), buf_->data_float() + buf_->size_float(), val);
  }
}

void Tensor::print(const std::string &name) const {
  if (!name.empty()) std::cout << name << " ";
  std::cout << "Tensor shape: [";
  for (size_t i = 0; i < shape_.ndim; ++i)
    std::cout << shape_.dims[i] << (i + 1 < shape_.ndim ? ", " : "");
  std::cout << "], dtype=" << (dtype_ == DType::BF16 ? "BF16" : "FP32")
            << ", size: " << size() << "\n";
  if (!buf_ || buf_->empty()) { std::cout << "  []\n"; return; }
  std::cout << "  ";
  size_t n = std::min(num_elements(), size_t(20));
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (dtype_ == DType::BF16) v = bf16_to_float(((uint16_t*)buf_->data())[i]);
    else v = buf_->data_float()[i];
    std::cout << v << " ";
  }
  if (num_elements() > 20) std::cout << "...";
  std::cout << "\n";
}

// ── Arithmetic (always works on FP32 data) ─────────────────────────────────
void Tensor::add_(const Tensor &other) {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch for in-place addition");
  // Both must be FP32 for vDSP. BF16 tensors should use GPU residual_add.
  size_t n = buf_->size_float();
  if (n > 262144 && metal_bridge::is_available()) {
    metal_bridge::residual_add(buf_->data_float(), other.buf_->data_float(), n);
    return;
  }
  vDSP_vadd(buf_->data_float(), 1, other.buf_->data_float(), 1,
            buf_->data_float(), 1, static_cast<vDSP_Length>(n));
}

void Tensor::mul_(const Tensor &other) {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch for in-place multiplication");
  vDSP_vmul(buf_->data_float(), 1, other.buf_->data_float(), 1,
            buf_->data_float(), 1,
            static_cast<vDSP_Length>(buf_->size_float()));
}

void Tensor::scale_(float factor) {
  vDSP_vsmul(buf_->data_float(), 1, &factor, buf_->data_float(), 1,
             static_cast<vDSP_Length>(buf_->size_float()));
}

Tensor Tensor::add(const Tensor &other) const {
  Tensor result = clone();
  result.add_(other);
  return result;
}

Tensor Tensor::mul(const Tensor &other) const {
  Tensor result = clone();
  result.mul_(other);
  return result;
}

Tensor Tensor::scale(float factor) const {
  Tensor result = clone();
  result.scale_(factor);
  return result;
}

// ── Deep copy ──────────────────────────────────────────────────────────────
Tensor Tensor::clone() const {
  Tensor t(shape_, dtype_);
  if (buf_ && buf_->bytes() > 0) {
    size_t n = num_elements();
    if (dtype_ == DType::BF16) {
      // Copy raw bytes — no conversion needed
      std::memcpy(t.buf_->data(), buf_->data(), buf_->bytes());
      t.buf_->reshape(n, 2);
    } else {
      t.buf_->resize_store_f32(n, buf_->data_float());
    }
  }
  return t;
}

// ── Matmul ─────────────────────────────────────────────────────────────────
Tensor Tensor::matmul(const Tensor &other) const {
  if (shape_.ndim < 2 || other.shape_.ndim < 2)
    throw std::invalid_argument("Matmul requires at least 2 dimensions");

  if (other.shape_.ndim == 2 && shape_.ndim != 2) {
    const size_t K = shape_.back();
    const size_t N = other.shape_[1];
    if (K != other.shape_[0])
      throw std::invalid_argument("Inner dimensions must match for projection");

    size_t M = 1;
    for (size_t i = 0; i + 1 < shape_.ndim; ++i) M *= shape_.dims[i];

    Shape result_shape = shape_;
    result_shape.dims[shape_.ndim - 1] = N;
    // Result inherits dtype from this tensor (will be BF16 if input is BF16)
    Tensor result(result_shape, 0.0f, dtype_);

    metal_bridge::gemm_bf16(buf_->data(), other.buf_->data(), result.buf_->data(),
                            M, N, K, false, false);
    return result;
  }

  if (shape_.ndim != other.shape_.ndim)
    throw std::invalid_argument("Batch size must match for matrix multiplication");

  size_t rank = shape_.ndim;
  for (size_t i = 0; i + 2 < rank; ++i)
    if (shape_.dims[i] != other.shape_.dims[i])
      throw std::invalid_argument("Batch dimension must match for matmul");

  if (shape_.dims[rank - 1] != other.shape_.dims[rank - 2])
    throw std::invalid_argument("Inner dimensions must match for matmul");

  Shape result_shape = shape_;
  result_shape.dims[rank - 1] = other.shape_.back();
  Tensor result(result_shape, 0.0f, dtype_);

  size_t num_batches = 1;
  for (size_t i = 0; i + 2 < rank; ++i) num_batches *= shape_.dims[i];

  const size_t M = shape_.dims[rank - 2];
  const size_t K = shape_.dims[rank - 1];
  const size_t N = other.shape_.back();

  const size_t elem = itemsize();
  const size_t batchA = M * K * elem, batchB = K * N * elem, batchC = M * N * elem;
  for (size_t b = 0; b < num_batches; ++b)
    metal_bridge::gemm_bf16(
        (const char*)buf_->data() + b * batchA,
        (const char*)other.buf_->data() + b * batchB,
        (char*)result.buf_->data() + b * batchC,
        M, N, K, false, false);

  return result;
}

Tensor Tensor::transpose() const {
  if (shape_.ndim != 2)
    throw std::invalid_argument("Transpose only for 2D tensors");
  size_t rows = shape_[0], cols = shape_[1];
  Tensor dest({cols, rows}, 0.0f);
  vDSP_mtrans(buf_->data_float(), 1, dest.buf_->data_float(), 1,
              static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
  return dest;
}

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const {
  size_t new_size = 1;
  for (auto d : new_shape) new_size *= d;
  if (new_size != num_elements())
    throw std::invalid_argument("Total size must match for reshape");
  Tensor t(new_shape);
  if (buf_) t.buf_->resize_store_f32(buf_->size_float(), buf_->data_float());
  return t;
}
