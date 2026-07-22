// ==============================================================================
// TECHNICAL SPECIFICATION: ADAMW OPTIMIZER STEP (FUSED GPU KERNEL)
// ==============================================================================
//
// WHAT: Performs the complete AdamW parameter update in a single GPU pass.
//       Updates first moment (m), second moment (v), applies weight decay,
//       and updates the parameter — all fused into one kernel launch.
//
// WHY: Running 6 separate element-wise operations as individual kernel launches
//      would waste GPU launch overhead. Fusing them into one kernel reads/writes
//      each memory location exactly once, maximizing memory bandwidth utilization.
//
// THREAD GRID: 1D dispatch — one thread per parameter element.
//
// ALGORITHM (per element j):
//   1. m[j] = beta1 * m[j] + (1 - beta1) * grad[j]
//   2. v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j]
//   3. param[j] -= lr * weight_decay * param[j]    (decoupled weight decay)
//   4. m_hat = m[j] / bias_correction1
//   5. v_hat = v[j] / bias_correction2
//   6. param[j] -= lr * m_hat / (sqrt(v_hat) + eps)
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

struct AdamWParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
    uint  n;
};

kernel void adamw_step(
    device bfloat*      param [[buffer(0)]],
    device const bfloat* grad  [[buffer(1)]],
    device float*       m     [[buffer(2)]],
    device float*       v     [[buffer(3)]],
    constant AdamWParams& p   [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    // WHAT: Bounds check to prevent out-of-range GPU threads from executing.
    if (idx >= p.n) return;

    float g = grad[idx];

    // WHAT: Update first moment (exponential moving average of gradients).
    // WHY: Tracks the direction of recent gradients for momentum-based updates.
    m[idx] = p.beta1 * m[idx] + (1.0f - p.beta1) * g;

    // WHAT: Update second moment (exponential moving average of squared gradients).
    // WHY: Tracks the magnitude of recent gradients for adaptive learning rate scaling.
    v[idx] = p.beta2 * v[idx] + (1.0f - p.beta2) * g * g;

    // WHAT: Apply decoupled weight decay directly to the parameter.
    // WHY: AdamW separates weight decay from gradient-based updates to prevent
    //      the adaptive learning rate from interfering with regularization.
    if (p.weight_decay > 0.0f) {
        param[idx] = (bfloat)((float)param[idx] - p.lr * p.weight_decay * (float)param[idx]);
    }

    float m_hat = m[idx] / p.bias_correction1;
    float v_hat = v[idx] / p.bias_correction2;
    param[idx] = (bfloat)((float)param[idx] - p.lr * m_hat / (sqrt(v_hat) + p.eps));
}
