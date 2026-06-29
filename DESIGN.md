# AQUA Platform — Design Notes (Phase 1–2)

This document records architectural decisions for the autodiff engine, mask-aware optimization, and performance characteristics observed while building Phases 1 and 2.

---

## 1. What does the engine compute as "the gradient of a masked weight", and why is that the right choice?

A **masked weight** is a parameter whose boolean `mask` entry is `False`. In the forward pass the weight value is treated as zero (or will be, once pruning is integrated in the Linear layer). In reverse mode, the engine must answer: *if this connection is dead, how much should we move the weight?*

**Answer: the gradient is exactly `0.0` at every masked index.**

### Where this is enforced

**`engine/tensor.py` — `Tensor.backward()`**

After topological ordering, each node's accumulated gradient is multiplied by its mask *before* propagating to parents:

```python
if node.mask is not None:
    node.grad = node.grad * node.mask
```

Masked entries therefore never send non-zero signal into the rest of the graph, and never receive a non-zero `.grad` value themselves. This matches the calculus: if a weight is hard-constrained to zero, it is not a free variable—the partial derivative of the loss with respect to that weight is undefined as an update direction, and the correct optimization treatment is to exclude it.

**`optim/adam.py` — `Adam.step()`**

Even with zero gradients, standard Adam would still apply **historical** first-moment (`m`) and second-moment (`v`) estimates built from earlier, pre-pruning steps. A previously active weight could have large stored momentum; once pruned, gradient becomes zero but `m` and `v` remain non-zero, so the bias-corrected update

\[
\theta \leftarrow \theta - \alpha \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)
\]

can push \(\theta\) away from zero. This is the **zombie weight** failure mode.

Our optimizer therefore:

1. Multiplies `m` and `v` by `param.mask` after each moment update, permanently zeroing optimizer state at dead indices.
2. Re-applies `param.data *= param.mask` after the parameter update so numerical drift cannot resurrect pruned weights.

### Why this is the right choice

| Approach | Problem |
|----------|---------|
| Standard Adam (no masking) | Momentum resurrects dead weights; sparsity is violated. |
| Mask only `param.data` | Optimizer state still updates; weights creep back non-zero. |
| Mask grad + mask `m`, `v`, and `data` | Dead weights stay at 0 with zero optimizer memory—consistent sparsity. |

For pruning, **importance scores** (Phase 3) also rely on `weight * gradient`. Zeroing the gradient at masked sites ensures pruned parameters do not pollute saliency rankings or trigger spurious un-pruning.

---

## 2. Where does the autodiff engine bottleneck, and how would you optimize it?

The current engine is a **pure Python reverse-mode interpreter** over a dynamically built DAG. Each forward op allocates a new `Tensor`, stores a closure for `_backward`, and links `parents`. `backward()` runs a fresh topological sort on every loss evaluation.

### Primary bottlenecks

1. **Python interpreter overhead**
   Every arithmetic op dispatches through dunder methods, `ops.py` wrappers, and closure creation. For a small MLP this dominates wall time compared to the underlying NumPy BLAS calls.

2. **Per-step topological sort**
   `_topological_sort` walks the full graph with recursive DFS and `id()`-based visited sets. Training re-traverses an isomorphic graph structure every iteration instead of reusing a cached execution plan.

3. **NumPy allocation churn**
   `_accumulate_grad` does `tensor.grad = tensor.grad + grad`, allocating a new array each accumulation. Reduction ops (`unbroadcast`, `_expand_grad_for_reduction`) frequently call `.copy()` to materialize broadcast gradients. Adam adds further full-sized `m` and `v` buffers per parameter.

4. **No kernel fusion**
   Sequences like `matmul → add → relu` launch separate kernels (and separate graph nodes) where a fused CPU/GPU kernel would touch memory once.

5. **Float64 everywhere**
   Correct for gradient checking and numerical stability, but roughly 2× memory bandwidth vs. float32 at production scale.

### Production optimization roadmap

| Technique | Benefit |
|-----------|---------|
| **Static graph compilation** | Trace or script the model once; emit a fixed topo order and fused backward schedule; eliminate per-op Python dispatch. |
| **Gradient buffer reuse** | Pre-allocate `.grad` arrays and use in-place `+=`; pool temporaries across steps. |
| **Sparse / structured ops** | When masks are fixed, store CSR/CSC weight blocks and skip dead multiply-adds analytically (see `evaluate/cost.py`). |
| **C++/CUDA extension** | Move hot paths (`matmul`, `relu`, cross-entropy backward) to compiled code; keep Python for orchestration. |
| **Mixed precision** | float32 forward + float32/bfloat16 grads with loss scaling; keep float64 only in debug/grad-check builds. |
| **Mini-batch vectorization at graph level** | Avoid rebuilding subgraph metadata when only batch data pointers change. |

For the Digits proof-of-concept, these costs are acceptable. For million-parameter sparse training, **graph compilation plus sparse linear algebra** would be the first lever—Python autodiff validates correctness; compiled sparse kernels deliver throughput.

---

*Subsequent phases will extend this document with pruning schedule rationale, saliency vs. magnitude trade-offs, and analytical FLOP accounting.*
