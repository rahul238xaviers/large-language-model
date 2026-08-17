// ==============================================================================
// KERNEL: embedding_forward
// WHAT: Performs token embedding lookup on GPU.
//       For every output position (b, s, d), reads the token ID at (b, s)
//       and copies the corresponding embedding row from the weight table.
// WHY:  Replaces a CPU triple-nested loop (batch × seq × hidden_dim = 33.5M
//       scalar assignments) that previously ran before any GPU work began.
// INPUT LAYOUT:
//   token_ids      : uint32[B * S]               — flattened token IDs
//   embedding_table: float[vocab_size * hidden_dim] — row-major embedding matrix
//   output (h)     : float[B * S * hidden_dim]   — flattened hidden states
// THREAD LAYOUT:
//   1D grid of (B * S * hidden_dim) threads.
//   Each thread owns exactly one float element in the output.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

kernel void embedding_forward(
    device const uint*  token_ids       [[buffer(0)]],  // WHAT: Flat token ID array [B*S]
    device const bfloat* embedding_table [[buffer(1)]],  // WHAT: Weight matrix [vocab_size, hidden_dim]
    device bfloat*       output          [[buffer(2)]],  // WHAT: Output hidden states [B*S, hidden_dim]
    constant uint&      hidden_dim      [[buffer(3)]],  // WHAT: Embedding / hidden dimension H
    constant uint&      total_tokens    [[buffer(4)]],  // WHAT: B * S (total token positions)
    constant uint&      vocab_size      [[buffer(5)]],  // WHAT: Embedding table row count
    uint gid [[thread_position_in_grid]]
) {
    // WHAT: Total elements = B * S * hidden_dim. Guard threads past this.
    // WHY:  Threadgroup size is a power of 2; last group may overshoot.
    if (gid >= total_tokens * hidden_dim) return;

    // WHAT: Decompose flat index into (token_position, dim).
    // WHY:  Allows each thread to independently look up its token ID and
    //       offset into the embedding table — no inter-thread communication.
    uint token_pos = gid / hidden_dim;  // which (b, s) position
    uint d         = gid % hidden_dim;  // which embedding dimension

    // WHAT: Bounds check on token_pos.
    if (token_pos >= total_tokens) return;

    // WHAT: Read the integer token ID for this position.
    uint token_id = token_ids[token_pos];

    // WHAT: Copy the d-th dimension of the token's embedding row to the output.
    // WHY:  row = token_id * hidden_dim; element = row + d.
    // Guard: an out-of-range token id would index past the table (GPU fault).
    if (token_id >= vocab_size) {
      output[gid] = 0;
      return;
    }
    output[gid] = embedding_table[token_id * hidden_dim + d];
}
