#include "Tensor.hpp"
#define ACCELERATE_NEW_LAPACK

void (*PagedBuffer::gpu_release_fn)(void*) = nullptr;

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

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
    if (dtype == DType::BF16) {
      buf_->resize_raw(n, 2);
      uint16_t bf = float_to_bf16(val);
      uint16_t* ptr = (uint16_t*)buf_->data();
      std::fill(ptr, ptr + n, bf);
    } else {
      buf_->resize_fill_f32(n, val);
    }
  }
  compute_strides();
}

Tensor::Tensor(const Shape &shape, const std::vector<float> &data, DType dtype)
    : shape_(shape), dtype_(dtype) {
  buf_ = std::make_shared<PagedBuffer>();
  size_t n = compute_size(shape);
  if (dtype == DType::BF16) {
    buf_->resize_raw(n, 2);
    uint16_t* dst = (uint16_t*)buf_->data();
    for (size_t i = 0; i < n && i < data.size(); ++i)
      dst[i] = float_to_bf16(data[i]);
  } else {
    buf_->resize_store_f32(data.size(), data.data());
  }
  compute_strides();
}

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

void Tensor::fill(float val) {
  if (!buf_) return;
  if (dtype_ == DType::BF16) {
    uint16_t bf = float_to_bf16(val);
    uint16_t* ptr = (uint16_t*)buf_->data();
    size_t n = buf_->bytes() / 2;
    std::fill(ptr, ptr + n, bf);
  } else {
    std::fill(buf_->data_float(), buf_->data_float() + buf_->bytes() / sizeof(float), val);
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
    if (dtype_ == DType::BF16) v = bf16_to_float(((const uint16_t*)buf_->data())[i]);
    else v = buf_->data_float()[i];
    std::cout << v << " ";
  }
  if (num_elements() > 20) std::cout << "...";
  std::cout << "\n";
}

void Tensor::add_(const Tensor &other) {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch for in-place addition");
  size_t n = buf_->bytes() / sizeof(float);
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
            static_cast<vDSP_Length>(buf_->bytes() / sizeof(float)));
}

void Tensor::scale_(float factor) {
  vDSP_vsmul(buf_->data_float(), 1, &factor, buf_->data_float(), 1,
             static_cast<vDSP_Length>(buf_->bytes() / sizeof(float)));
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

Tensor Tensor::clone() const {
  Tensor t(shape_, dtype_);
  if (buf_ && buf_->bytes() > 0) {
    size_t n = num_elements();
    if (dtype_ == DType::BF16) {
      std::memcpy(t.buf_->data(), buf_->data(), buf_->bytes());
      t.buf_->reshape(n, 2);
    } else {
      t.buf_->resize_store_f32(n, buf_->data_float());
    }
  }
  return t;
}

Tensor Tensor::to_dtype(DType target_dtype) const {
  if (dtype_ == target_dtype) return *this;
  Tensor res(shape_, target_dtype);
  size_t n = num_elements();
  if (buf_ && buf_->bytes() > 0) {
    if (target_dtype == DType::BF16) {
      uint16_t *dst = (uint16_t*)res.buf_->data();
      const float *src = buf_->data_float();
      for (size_t i = 0; i < n; ++i) dst[i] = float_to_bf16(src[i]);
    } else {
      float *dst = res.buf_->data_float();
      const uint16_t *src = (const uint16_t*)buf_->data();
      for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_float(src[i]);
    }
  }
  return res;
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (shape_.ndim < 2 || other.shape_.ndim < 2)
    throw std::invalid_argument("Matmul requires at least 2 dimensions");

  bool run_as_bf16 = (dtype_ == DType::BF16 || other.dtype_ == DType::BF16);
  DType result_dtype = run_as_bf16 ? DType::BF16 : dtype_;

  size_t rank = shape_.ndim;
  Shape result_shape = shape_;
  result_shape.dims[rank - 1] = other.shape_.back();

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (!use_gpu) {
    Tensor a_fp32 = to_dtype(DType::FP32);
    Tensor b_fp32 = other.to_dtype(DType::FP32);
    Tensor res_fp32(result_shape, 0.0f, DType::FP32);

    if (other.shape().ndim == 2 && shape().ndim != 2) {
      const size_t K = shape().back();
      const size_t N = other.shape()[1];
      size_t M = 1;
      for (size_t i = 0; i + 1 < shape().ndim; ++i) M *= shape().dims[i];

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  (int)M, (int)N, (int)K,
                  1.0f, a_fp32.data(), (int)K,
                  b_fp32.data(), (int)N,
                  0.0f, res_fp32.data(), (int)N);
    } else {
      size_t num_batches = 1;
      for (size_t i = 0; i + 2 < rank; ++i) num_batches *= a_fp32.shape().dims[i];

      const size_t M = a_fp32.shape().dims[rank - 2];
      const size_t K = a_fp32.shape().dims[rank - 1];
      const size_t N = b_fp32.shape().back();

      const size_t batchA = M * K, batchB = K * N, batchC = M * N;

      for (size_t b_idx = 0; b_idx < num_batches; ++b_idx) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)M, (int)N, (int)K,
                    1.0f, a_fp32.data() + b_idx * batchA, (int)K,
                    b_fp32.data() + b_idx * batchB, (int)N,
                    0.0f, res_fp32.data() + b_idx * batchC, (int)N);
      }
    }
    return res_fp32.to_dtype(result_dtype);
  }

  Tensor a = run_as_bf16 ? to_dtype(DType::BF16) : *this;
  Tensor b = run_as_bf16 ? other.to_dtype(DType::BF16) : other;

  if (b.shape_.ndim == 2 && a.shape_.ndim != 2) {
    const size_t K = a.shape_.back();
    const size_t N = b.shape_[1];
    if (K != b.shape_[0])
      throw std::invalid_argument("Inner dimensions must match for projection");

    size_t M = 1;
    for (size_t i = 0; i + 1 < a.shape_.ndim; ++i) M *= a.shape_.dims[i];

    Shape res_shape_proj = a.shape_;
    res_shape_proj.dims[a.shape_.ndim - 1] = N;
    Tensor result(res_shape_proj, 0.0f, result_dtype);

    metal_bridge::gemm_bf16(a.buf_->data(), b.buf_->data(), result.buf_->data(),
                            M, N, K, false, false);
    return run_as_bf16 ? result.to_dtype(dtype_) : result;
  }

  if (a.shape_.ndim != b.shape_.ndim)
    throw std::invalid_argument("Batch size must match for matrix multiplication");

  for (size_t i = 0; i + 2 < rank; ++i)
    if (a.shape_.dims[i] != b.shape_.dims[i])
      throw std::invalid_argument("Batch dimension must match for matmul");

  if (a.shape_.dims[rank - 1] != b.shape_.dims[rank - 2])
    throw std::invalid_argument("Inner dimensions must match for matmul");

  Shape res_shape_batched = a.shape_;
  res_shape_batched.dims[rank - 1] = b.shape_.back();
  Tensor result(res_shape_batched, 0.0f, result_dtype);

  size_t num_batches = 1;
  for (size_t i = 0; i + 2 < rank; ++i) num_batches *= a.shape_.dims[i];

  const size_t M = a.shape_.dims[rank - 2];
  const size_t K = a.shape_.dims[rank - 1];
  const size_t N = b.shape_.back();

  const size_t elem = ::itemsize(result_dtype);
  const size_t batchA = M * K * elem, batchB = K * N * elem, batchC = M * N * elem;
  for (size_t b_idx = 0; b_idx < num_batches; ++b_idx)
    metal_bridge::gemm_bf16(
        (const char*)a.buf_->data() + b_idx * batchA,
        (const char*)b.buf_->data() + b_idx * batchB,
        (char*)result.buf_->data() + b_idx * batchC,
        M, N, K, false, false);

  return run_as_bf16 ? result.to_dtype(dtype_) : result;
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
  if (buf_) t.buf_->resize_store_f32(buf_->bytes() / sizeof(float), buf_->data_float());
  return t;
}
