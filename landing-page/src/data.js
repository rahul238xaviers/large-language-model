export const curriculumData = [
  {
    phase: "Data Pipeline",
    description: "Before the model can learn, raw text must be converted into a mathematical format it can process.",
    topics: [
      {
        id: "01",
        title: "Text Corpus → Tokenization",
        subtitle: "Converting words to integer IDs",
        description: "Language models don't read text; they read numbers. We use Byte-Pair Encoding (BPE) to compress common words and subwords into unique integer IDs.",
        why: "Without tokenization, the model would have to learn character by character, which is highly inefficient and loses semantic meaning.",
        cppLink: "cpp/src/DataIngestion.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/data.py"
      },
      {
        id: "02",
        title: "Batching",
        subtitle: "Building the 2D input grid [batch_size, seq_len]",
        description: "To maximize GPU utilization, we group multiple sequences of tokens into a single batch matrix.",
        why: "GPUs are designed for massive parallel matrix operations. Processing one sequence at a time would leave 99% of the GPU idle.",
        cppLink: "cpp/src/DataIngestion.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/data.py"
      }
    ]
  },
  {
    phase: "Forward Pass (Inside the Transformer)",
    description: "The core engine where tokens attend to each other to build contextual understanding.",
    topics: [
      {
        id: "03",
        title: "Token Embedding Lookup",
        subtitle: "2D [batch, seq_len] → 3D [batch, seq_len, hidden_dim]",
        description: "Each token ID is mapped to a high-dimensional vector. Words with similar meanings will eventually have vectors pointing in similar directions.",
        why: "Integers have no inherent relationship (ID 5 isn't 'closer' to ID 6 than ID 1000). Vectors allow the model to represent complex semantic relationships in space.",
        cppLink: "cpp/src/gpu_kernel/embedding_forward.metal",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "04",
        title: "RMSNorm",
        subtitle: "Normalize each token's embedding vector",
        description: "Root Mean Square Normalization stabilizes the network by keeping the variance of activations in check.",
        why: "Without normalization, activations can explode or vanish as they pass through dozens of layers, completely destroying the learning process.",
        cppLink: "cpp/src/RMSNorm.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "05",
        title: "Q, K, V Projections",
        subtitle: "Token-local matrix multiplication",
        description: "Each token creates a Query (what I'm looking for), a Key (what I contain), and a Value (what I will pass on).",
        why: "This is the fundamental mechanism of self-attention. It allows the network to dynamically route information based on context rather than fixed weights.",
        cppLink: "cpp/src/Attention.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "07",
        title: "RoPE (Rotary Position Embedding)",
        subtitle: "Injecting relative position information",
        description: "Rotates the Query and Key vectors in space based on their position in the sequence.",
        why: "Standard attention has no concept of order. RoPE mathematically encodes that 'word A is 3 steps away from word B' by rotating their vectors relative to each other.",
        cppLink: "cpp/src/Positional.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "09",
        title: "Attention Scores & Output",
        subtitle: "Scaled dot product of Q · K^T",
        description: "Queries dot-product with Keys to find matches, scaled down to prevent saturation, masked to prevent looking into the future, and softmaxed to create probabilities. Finally, it's multiplied by Values.",
        why: "This is where 'context' is actually built. The word 'bank' figures out if it's sitting next to 'river' or 'money'.",
        cppLink: "cpp/src/gpu_kernel/flash_attn_fwd.metal",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "15",
        title: "Feed-Forward Network (SwiGLU)",
        subtitle: "Non-linear feature processing",
        description: "A two-layer network applied to every token individually. It uses a Swish-Gated Linear Unit to determine which features to activate and pass forward.",
        why: "Attention routes information between words; the FFN processes that information to build complex representations (like 'is this a sarcastic sentence?').",
        cppLink: "cpp/src/gpu_kernel/swiglu_forward.metal",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      }
    ]
  },
  {
    phase: "Output & Loss",
    description: "Evaluating the model's predictions against reality.",
    topics: [
      {
        id: "21",
        title: "Language Model Head",
        subtitle: "Projecting back to vocabulary size",
        description: "A massive linear layer that maps the final hidden states back to a vector the size of the entire vocabulary.",
        why: "We need raw scores (logits) for every possible word in the dictionary to determine what the model thinks should come next.",
        cppLink: "cpp/src/Transformer.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/model.py"
      },
      {
        id: "23",
        title: "Cross-Entropy Loss",
        subtitle: "Measuring how wrong the prediction was",
        description: "Compares the model's predicted probability distribution with the actual next word in the training data.",
        why: "This single scalar value is the 'ground truth' error signal. It drives the entire learning process in the backward pass.",
        cppLink: "cpp/src/Loss.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/train.py"
      }
    ]
  },
  {
    phase: "Backward Pass & Parameter Update",
    description: "The learning phase: updating weights to reduce future errors.",
    topics: [
      {
        id: "24",
        title: "Gradients (Chain Rule)",
        subtitle: "Flowing the error backward",
        description: "Calculating the partial derivative of the loss with respect to every single parameter in the network, in reverse order.",
        why: "Tells us exactly which direction (and by how much) we need to adjust every weight matrix to make the model slightly more accurate next time.",
        cppLink: "cpp/src/gpu_kernel/gemm_backward.metal",
        pythonLink: "python/apple-silicon/src/pre-training/train.py"
      },
      {
        id: "36",
        title: "AdamW Optimizer Step",
        subtitle: "Updating the weights",
        description: "Applies the calculated gradients to update the weights, using momentum (keeping track of past gradients) and weight decay (preventing weights from growing too large).",
        why: "Standard gradient descent is too slow and gets stuck easily. AdamW navigates the complex, multi-billion dimensional loss landscape efficiently.",
        cppLink: "cpp/src/Optimizer.cpp",
        pythonLink: "python/apple-silicon/src/pre-training/train.py"
      }
    ]
  }
];
