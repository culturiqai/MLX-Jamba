# Hybrid Mamba Implementation for MLX

An educational implementation of Hybrid Mamba architecture combining Mamba (state-space models) with Transformer attention mechanisms for MLX framework.

## ⚠️ Important Notice

This is an **educational/research implementation** with significant performance limitations. It is not optimized for production use.

## Overview

This repository contains two hybrid architectures:
1. **Simple Hybrid**: Alternating Mamba and Attention layers with learnable state bridges
2. **True Hybrid**: Parallel processing of both paths with complexity-based routing (experimental)

## Features

### ✅ Working Features
- Complete Mamba block implementation with selective SSM
- Multi-head attention with causal masking
- Configurable hybrid layer placement
- Basic text generation
- RMSNorm and proper weight initialization
- Extensive testing suite

### ⚠️ Partially Implemented
- KV cache structure (defined but not utilized in generation)
- Chunked attention (basic implementation)
- Cross-modal attention mechanisms
- Adaptive routing (runs both paths regardless)

### ❌ Not Implemented
- Top-p sampling (returns original logits)
- Gradient checkpointing integration
- Optimized generation with KV cache
- Vectorized scan operations

## Installation

```bash
# Requires MLX framework
pip install mlx
pip install numpy

# Clone repository
git clone <repository-url>
cd hybrid-mamba-mlx
```

## Quick Start

```python
from hybrid_mamba import HybridMambaConfig, HybridMambaModel

# Configure model
config = HybridMambaConfig(
    d_model=512,
    n_layers=12,
    attention_layers=[3, 6, 9],  # Attention at layers 3, 6, 9
    vocab_size=32000
)

# Create model
model = HybridMambaModel(config, use_true_hybrid=False)

# Generate text
input_ids = mx.array([[1, 2, 3, 4, 5]])
output = model.generate(input_ids, max_new_tokens=50)
```

## Architecture Details

### Simple Hybrid (Recommended)
- Alternates between Mamba and Attention layers
- Learnable state bridges transfer information between layer types
- More efficient than true hybrid

### True Hybrid (Experimental)
- Processes input through both Mamba and Attention simultaneously
- Uses complexity analysis for routing decisions
- Currently inefficient due to parallel execution

## Performance Considerations

### ⚠️ Known Limitations

1. **Sequential Scan**: The core Mamba scan uses a Python loop, making it significantly slower than optimized implementations
2. **No KV Caching**: Despite being implemented, KV cache is not used in generation
3. **Sampling**: Top-k sampling uses nested loops; top-p is unimplemented
4. **Memory Usage**: No optimization for long sequences
5. **True Hybrid Overhead**: Runs both paths always, doubling computation

### Benchmark Results
```
Model Size: 512d, 12 layers
- Forward pass: ~0.5s for 32 tokens (batch=2)
- Generation: ~2s for 50 tokens (no KV cache benefit)
- Memory: ~200MB for small model
```

## Code Structure

```
hybrid_mamba.py
├── Configuration (HybridMambaConfig)
├── Core Components
│   ├── SelectiveSSM (State Space Model)
│   ├── CausalConv1d
│   ├── MambaBlock
│   └── MultiHeadAttention
├── Hybrid Components  
│   ├── HybridStateBridge
│   ├── ComplexityAnalyzer
│   └── TrueHybridLayer
├── Main Model (HybridMambaModel)
└── Tests (embedded - should be separated)
```

## Usage Examples

### Basic Generation
```python
# Simple generation
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50  # Note: top_p not implemented
)
```

### Routing Analysis (True Hybrid only)
```python
# Analyze routing decisions
analysis = model.get_routing_analysis(input_ids)
print(f"Mamba usage: {analysis['average_usage']['mamba']:.1%}")
print(f"Attention usage: {analysis['average_usage']['attention']:.1%}")
```

## Development Status

This implementation prioritizes clarity and educational value over performance. Key areas needing improvement:

1. Vectorize all sequential operations
2. Implement KV caching in generation
3. Complete sampling methods
4. Separate tests into proper test files
5. Optimize memory usage
6. Implement conditional routing (not parallel)

## Contributing

Contributions welcome, especially for:
- Performance optimizations
- Completing unimplemented features
- Adding proper benchmarks
- Improving documentation

## License

MIT

## Citation

If you use this code for research, please cite:
```bibtex
Aditya Tiwari
```

## Acknowledgments

Based on the Mamba architecture paper and MLX framework examples.

---

**Note**: This is an educational implementation. For production use, consider optimized alternatives or the official Mamba implementation.
