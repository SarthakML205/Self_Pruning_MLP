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

## 3. Derive your importance criterion and explain why it approximates the loss change from removing a connection

We prune unstructured weights by ranking them with a **saliency score** derived from a first-order Taylor expansion of the training loss.

### Setup

Let \(\mathcal{L}(w)\) be the mini-batch loss and \(w\) a scalar weight. Suppose the weight is **removed** by forcing it to zero. Define the perturbation:

\[
\Delta w = 0 - w = -w
\]

A first-order Taylor expansion of \(\mathcal{L}\) around the current \(w\) gives:

\[
\mathcal{L}(w + \Delta w) \approx \mathcal{L}(w) + \frac{\partial \mathcal{L}}{\partial w}\,\Delta w
\]

The **change in loss** from removing the connection is therefore:

\[
\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial w}\,(-w) = -\frac{\partial \mathcal{L}}{\partial w}\, w
\]

The **magnitude** of the predicted impact is:

\[
\left|\Delta \mathcal{L}\right| \approx \left|\frac{\partial \mathcal{L}}{\partial w}\, w\right|
\]

This is exactly the **Taylor saliency** score implemented in `SaliencyCriterion`:

\[
\text{score}(w) = \left| w \cdot \frac{\partial \mathcal{L}}{\partial w} \right|
\]

### Execution order matters

The score uses `.grad` from the **current** backward pass. In `train/trainer.py` the batch loop is:

1. `model.zero_grad()`
2. forward + `loss.backward()`
3. `pruner.step()` — scores computed here
4. `optimizer.step()`

If `optimizer.zero_grad()` ran before pruning, saliency would read stale or zero gradients and degenerate toward magnitude-only behavior.

### Why this beats magnitude pruning

| Criterion | Score | What it measures |
|-----------|-------|------------------|
| **Magnitude** | \(\|w\|\) | Static size of the weight |
| **Saliency (Taylor)** | \(\|w \cdot \nabla_w \mathcal{L}\|\) | Predicted loss increase if \(w \to 0\) |

A weight can be **small** yet sit on a steep loss slope (high gradient)—removing it hurts accuracy. Conversely, a **large** weight on a flat region (gradient \(\approx 0\)) contributes little to the current loss and is safe to prune.

Magnitude pruning ignores the local loss landscape and often prunes weights that are still functionally important. Taylor saliency aligns pruning decisions with **immediate sensitivity** of \(\mathcal{L}\), which is why `SaliencyCriterion` is the default in Phase 3.

### Schedule interaction (Zhu & Gupta cubic ramp)

Sparsity follows:

\[
s(t) = s_{\text{target}} + (s_{\text{initial}} - s_{\text{target}})\left(1 - \frac{t}{T}\right)^3
\]

Early steps prune rapidly while the network has surplus capacity; later steps slow down as remaining weights become scarce and each removal is higher risk. The `Pruner` applies a **global** `np.percentile` threshold across all layers so the live budget is met exactly at each schedule step.

---

*Phase 4 will extend this document with analytical FLOP accounting and Pareto sweep results.*
