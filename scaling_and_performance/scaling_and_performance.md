# LLM & ML Performance Engineering

---

## Table of Contents

1. [Metrics & Definitions](#1-metrics--definitions)
2. [Benchmarking & Load Testing](#2-benchmarking--load-testing)
3. [GPU Fundamentals](#3-gpu-fundamentals)
4. [LLM Inference Metrics](#4-llm-inference-metrics)
5. [Quantization & Optimization](#5-quantization--optimization)
6. [KV Cache](#6-kv-cache)
7. [Batching & Scheduling](#7-batching--scheduling)
8. [ML Serving Architecture](#8-ml-serving-architecture)
9. [Multi-GPU Scaling](#8-multi-gpu-scaling)
10. [Autoscaling LLM Services](#9-autoscaling-llm-services)
11. [Cost Optimization for LLM Inference](#10-cost-optimization-for-llm-inference)
12. [Self-Hosted Vector Database Performance](#11-self-hosted-vector-database-performance)
13. [Vector Database-Specific Questions](#12-vector-database-specific-questions)
14. [Embedding Performance](#13-embedding-performance)
15. [RAG Pipeline Performance](#14-rag-pipeline-performance)
16. [Chat-Retrieval Speed](#15-chat-retrieval-speed)
17. [Caching Strategies](#16-caching-strategies)
18. [Streaming Performance](#17-streaming-performance)
19. [Concurrency Management](#18-concurrency-management)
20. [Rate Limiting and QoS](#19-rate-limiting-and-qos)
21. [Load Balancing](#20-load-balancing)
22. [Observability and Monitoring](#21-observability-and-monitoring)

---

## 1. Metrics & Definitions

### What is the difference between latency, throughput, and concurrency?

- **Latency** = time to complete one request (ms). Think of it as the speed for a single car on a highway.
- **Throughput** = how many requests you can complete per second (RPS or tokens/sec). Think of it as the total number of cars passing through per hour.
- **Concurrency** = how many requests are being handled simultaneously at any moment. Like the number of lanes open on the highway.

> 💡 **Interview tip:** Low latency ≠ high throughput. You can have a fast single-response system that's terrible at handling 1000 simultaneous users.

---

### What is P50, P95, P99 latency?

These are percentiles of your latency distribution over many requests.

| Percentile | Meaning |
|---|---|
| P50 | Median. 50% of requests completed faster than this. |
| P95 | 95% of requests completed faster than this. Your "typical worst case." |
| P99 | 99% of requests completed faster than this. Tail latency — the pain your slowest users feel. |

```
Example: P50=100ms, P95=800ms, P99=2000ms
→ most users are fine, but 1 in 100 waits 2 seconds
```

> 💡 **Averages hide the pain. Always report percentiles in interviews.**

---

### Why is tail latency more important than average latency?

Average latency hides outliers. In large-scale systems, "fan-out" calls make tail latency compound.

If a page makes 100 microservice calls, the page latency = max latency of all 100 calls. Even a 1% slow tail means almost every page load hits that slow path.

```
Classic example:
P99 = 2s with 100 dependencies
→ 63% of page views will hit at least one slow service
```

> 💡 Always optimize tails, not averages.

---

### What is SLO, SLI, SLA?

| Term | Full Name | Meaning |
|---|---|---|
| SLI | Service Level Indicator | The metric you actually measure. e.g., "P99 latency = 250ms" |
| SLO | Service Level Objective | Your internal target. e.g., "P99 must be ≤ 500ms, 99.9% of the time" |
| SLA | Service Level Agreement | The contract with customers. SLOs with consequences if violated (refunds, penalties) |

```
SLI = what you measure → SLO = your goal → SLA = the legal promise
```

---

### What is Little's Law and how does it apply?

**Formula:** `L = λ × W`

| Variable | Meaning |
|---|---|
| L | Average number of requests in the system (concurrency) |
| λ | Arrival rate (RPS) |
| W | Average time spent per request (latency in seconds) |

```
Example:
100 RPS × 0.2s latency = 20 requests in-flight at any moment
```

> 💡 Use this to size thread pools, connection pools, and estimate server capacity.

---

### What is utilization and saturation?

- **Utilization** = % of time a resource is busy. GPU at 80% utilization = busy 80% of the time.
- **Saturation** = when a resource is at 100% and requests start queuing. This is where latency spikes.

> 💡 A system at 70% utilization feels smooth. At 90%+ it starts queuing and latency degrades non-linearly.

---

### What is queueing delay?

Time a request spends waiting in a queue **before** processing starts. When arrival rate > processing rate, requests queue up and wait.

Queueing delay grows **exponentially** as utilization approaches 100% — this is why you don't run systems at full capacity.

```
Total latency = service time + queueing delay
```

---

### What is the difference between CPU-bound and I/O-bound workloads?

- **CPU-bound:** Bottleneck is computation. More CPUs = faster. Example: model inference, matrix multiplication.
- **I/O-bound:** Bottleneck is waiting for disk, network, DB. CPU sits idle. Example: feature lookups, reading logs.

> 💡 LLM decoding is **memory-bandwidth bound** (a third category) — the GPU cores wait on memory reads, not computation.

---

### How do you measure tokens per second and RPS?

```
Tokens/sec = total output tokens generated / total time (seconds)
RPS        = total requests completed / total time (seconds)

Tokens/sec/GPU = total tokens / (time × num_GPUs)  ← normalize for hardware comparison
```

---

## 2. Benchmarking & Load Testing

### How do you benchmark an ML service?

Run controlled experiments with controlled variables:

1. Fix input shape/size (e.g., prompt length, batch size)
2. Measure TTFT, TPOT, throughput, GPU utilization
3. Sweep variables one at a time (batch size, concurrency, model precision)
4. **Warm up** the service before measuring (JIT, cache cold start)

> 💡 **Tools:** `locust`, `k6`, `wrk`, custom Python scripts. For LLMs: `vllm benchmark_serving.py`, `llmperf`.

---

### How do you generate realistic traffic patterns?

- Replay production logs with actual prompt lengths and timing
- Use Poisson distribution for random arrival rates
- Include realistic prompt length distributions (short/medium/long mix)
- Model time-of-day traffic curves (ramp up, peak, cooldown)

---

### How do you test burst traffic?

Gradually ramp load then spike it 5–10x suddenly. Observe:

- **Queue depth** — does it drain or grow indefinitely?
- **Latency degradation** during burst
- Whether autoscaling kicks in fast enough
- **Error rates** — do requests get rejected or time out?

---

### How do you isolate bottlenecks?

Profile each stage independently:

| Stage | Tool | Bottleneck Signal |
|---|---|---|
| GPU compute | `nvidia-smi`, `nsight` | SM utilization near 100% |
| CPU | `htop`, `perf` | CPU near 100%, GPU idle |
| Memory bandwidth | `nsight compute` | Low arithmetic intensity |
| Network | `iperf`, tracing | High p99 on data transfer |

> 💡 **Rule:** Find where utilization is highest AND latency is highest. That's your bottleneck.

---

## 3. GPU Fundamentals

### CUDA cores vs Tensor cores

| Type | Purpose | Speed |
|---|---|---|
| CUDA cores | General-purpose parallel processing. Handle any float/int operation. | Baseline |
| Tensor cores | Specialized for matrix multiply-accumulate (MMA). 4x4 or 8x8 matrix ops in one clock. | 10–50x faster for DL |

> 💡 LLM inference uses Tensor cores heavily for attention and feed-forward layers.

---

### What is VRAM and memory bandwidth?

- **VRAM:** GPU's on-chip memory (HBM). Stores model weights, KV cache, activations.
  - A100 = 80 GB, H100 = 80 GB
- **Memory bandwidth:** Speed of reading/writing VRAM.
  - A100 = 2 TB/s, H100 = 3.35 TB/s

> 💡 LLM decoding is **memory-bandwidth bound**. More bandwidth = more tokens/sec during decoding.

---

### What is PCIe bottleneck and NVLink?

- **PCIe:** Bus connecting GPU to CPU. ~64 GB/s. Bottleneck when moving data CPU→GPU frequently.
- **NVLink:** GPU-to-GPU interconnect. 600 GB/s on H100. Allows fast weight sharing across GPUs for tensor parallelism.

> 💡 Multi-GPU tensor parallelism **needs NVLink**. PCIe is too slow for weight synchronization across GPUs.

---

### How do you estimate GPU memory required for a model?

**Weight memory:**

```
FP16 weights (GB) = (parameters × 2 bytes) / 1e9

Examples:
  7B model  × 2 bytes = 14 GB   → fits in 1× A100 80GB
  70B model × 2 bytes = 140 GB  → needs 2× A100 80GB
  70B INT4  × 0.5 bytes = 35 GB → fits in 1× A100 80GB
```

**Total VRAM:**

```
Total = weights + KV cache + activations + overhead (~1.2x buffer)
```

---

### What contributes to VRAM usage?

| Component | Description | Nature |
|---|---|---|
| Model weights | Biggest component | Fixed per model |
| KV cache | Grows with batch size × sequence length | Dynamic |
| Activations | Intermediate layer outputs during forward pass | Dynamic |
| CUDA overhead | ~1–2 GB fixed per GPU | Fixed |

> 💡 In serving, **KV cache is often the limiting factor** — not weights. Bigger batches = more KV cache = OOM.

---

### What is SM utilization and occupancy?

- **SM (Streaming Multiprocessor):** GPU's compute units. A100 has 108 SMs.
- **SM utilization:** % of time SMs are doing actual work (vs idle/stalled).
- **Occupancy:** How many warps (thread groups) are active per SM vs maximum possible. Higher = better latency hiding.

---

### What is NVLink, and how does it affect multi-GPU training/inference?

NVLink is NVIDIA's high-bandwidth GPU interconnect (600 GB/s on H100 vs PCIe's ~64 GB/s). It enables:

- Fast all-reduce for gradient synchronization in training
- Tensor parallelism in inference (split weight matrices across GPUs, communicate activations each layer)
- Without NVLink, inter-GPU communication becomes the bottleneck for large models

---

## 4. LLM Inference Metrics

### What is TTFT, TPOT, and end-to-end latency?

| Metric | Full Name | What it measures |
|---|---|---|
| TTFT | Time To First Token | Time from request arrival to first token returned. Dominated by prompt prefill. |
| TPOT | Time Per Output Token | Time to generate each subsequent token. Dominated by memory bandwidth. |
| ITL | Inter-Token Latency | Same as TPOT in practice. |
| E2E | End-to-End Latency | Total user wait time. |

```
E2E = TTFT + (TPOT × num_output_tokens)

Example:
500ms TTFT + 30ms/token × 100 tokens = 3.5s total
```

---

### Why is prompt ingestion (prefill) expensive?

During prefill, the model computes attention over **all tokens in the prompt**. Attention is **O(n²)** in sequence length — doubling prompt length = 4x compute.

This phase is **compute-bound** (unlike decoding which is memory-bound).

> 💡 A 10k token prompt can take several seconds to prefill. This is why long-context models have high TTFT.

---

### What is the complexity of attention and why does it matter?

```
Standard attention: O(n²) time and space in sequence length n

For n = 1,000 tokens  → 1M attention pairs
For n = 10,000 tokens → 100M attention pairs (100x more compute)
```

This is why extending context length is hard and why FlashAttention, sliding window attention, and linear attention alternatives exist.

---

### How does FlashAttention improve performance?

Standard attention writes intermediate matrices (Q×K, softmax) to HBM (slow GPU memory) and reads them back. FlashAttention **fuses these operations** and keeps everything in fast SRAM (on-chip cache).

| Aspect | Standard Attention | FlashAttention |
|---|---|---|
| Speed | Baseline | ~3x faster |
| Memory for attention matrices | O(n²) HBM | O(n) SRAM |
| Accuracy | Reference | Exact same result |

> 💡 FlashAttention is the single biggest practical speedup in LLM inference. All major frameworks use it.

---

### What is Paged Attention?

Normally, KV cache is pre-allocated as one large contiguous block per sequence — wasting memory due to fragmentation and variable lengths.

**PagedAttention** (from vLLM) manages KV cache in fixed-size "pages" like OS virtual memory:

- Pages allocated on-demand
- Pages can be non-contiguous in physical memory
- Pages shared across sequences (prefix sharing, beam search)

**Result:** Near-zero memory waste, up to **24x higher throughput** vs naive serving.

---

### Why is decoding memory-bandwidth bound?

During decoding, you generate **one token at a time**. Each token generation:

1. Loads entire model weights from HBM → GPU cores
2. Does a tiny matrix multiply (1 token × weight matrix)
3. Produces one token
4. Repeats

The compute is trivial. The bottleneck is how fast you can **read all those weights from memory** every single token.

```
Arithmetic Intensity = FLOPs / bytes_read

Decoding has very low arithmetic intensity → memory bandwidth limited
```

> 💡 This is why batching helps! With a batch of 32, you load weights once and serve 32 users, spreading the memory cost.

---

### How does speculative decoding work?

Use a small cheap "draft" model to predict N tokens ahead. Verify all N tokens in parallel with the large model in a single forward pass.

```
Step 1: Draft model generates 5 tokens speculatively (fast, cheap)
Step 2: Target model verifies all 5 in one parallel forward pass
Step 3: Accept tokens up to first mismatch, discard rest
Step 4: Repeat
```

**Result:** 2–3x speedup with **no quality loss** (mathematically equivalent to greedy/sampling from target model).

> 💡 Works best when draft model has high acceptance rate (similar output distribution to target).

---

### What is prefix caching / prompt caching?

If many requests share the same prefix (e.g., system prompt), cache the KV entries for that prefix. Subsequent requests **reuse** those cached KV values instead of recomputing.

- Reduces TTFT significantly for common system prompts
- Reduces compute for repeated prefixes
- Supported by Anthropic, OpenAI, vLLM, SGLang

```
Example:
System prompt = 2000 tokens
Without prefix cache: 2000 tokens prefilled every request
With prefix cache:    0 tokens prefilled (cache hit) → 50-80% TTFT reduction
```

---

### What is Medusa decoding?

An alternative to speculative decoding. Instead of a separate draft model, add multiple "Medusa heads" on top of the base model — each head predicts tokens at different future positions simultaneously.

All heads run in the **same forward pass**, no separate draft model needed. Simpler to deploy. Typically **2–3x speedup**.

---

## 5. Quantization & Optimization

### What are FP32, FP16, BF16, INT8, INT4?

| Format | Bits | Bytes/param | Notes |
|---|---|---|---|
| FP32 | 32 | 4 | Training standard. Full precision. |
| FP16 | 16 | 2 | ~2x faster on tensor cores. Can have overflow issues. |
| BF16 | 16 | 2 | Same exponent range as FP32. Best for modern GPU training/inference. |
| INT8 | 8 | 1 | ~4x smaller than FP32. Small quality drop. Good for inference. |
| INT4 | 4 | 0.5 | Fits huge models on fewer GPUs. Noticeable quality drop on some tasks. |

> 💡 **Production rule:** BF16 for standard serving. INT4/INT8 when GPU memory is the hard constraint.

---

### What is GPTQ, AWQ, GGUF?

| Method | What it does | Best for |
|---|---|---|
| **GPTQ** | Post-training quantization that minimizes error layer by layer using calibration data | GPU serving at INT4 |
| **AWQ** | Identifies high-activation weights and preserves them at higher precision. Often better than GPTQ at INT4 | GPU serving at INT4 |
| **GGUF** | File format for quantized models (used by llama.cpp). Supports CPU or mixed CPU+GPU | Local / edge inference |

---

### What are vLLM, TGI, TensorRT-LLM, SGLang, llama.cpp?

| Framework | Key strength | Use case |
|---|---|---|
| **vLLM** | PagedAttention, best continuous batching | Production GPU serving |
| **TGI** | HuggingFace ecosystem, good autoscaling hooks | Production, HF models |
| **TensorRT-LLM** | NVIDIA compiler, fused kernels, highest throughput | NVIDIA GPU, max performance |
| **SGLang** | Radix attention, best for branching / agent workloads | Agents, multi-turn, complex programs |
| **llama.cpp** | CPU inference + quantization support | Local, edge, CPU deployment |

---

### What is operator fusion?

Combining multiple GPU operations into one kernel to avoid reading/writing intermediate results to slow HBM memory.

```
Without fusion:
LayerNorm → [write to HBM] → GELU → [write to HBM] → Linear → [write to HBM]
= 3 HBM reads + 3 HBM writes

With fusion:
LayerNorm + GELU + Linear in one kernel, all in fast SRAM
= 1 HBM read + 1 HBM write
```

> 💡 FlashAttention is the most famous example. TensorRT-LLM does this extensively across all layer types.

---

### What is TensorRT-LLM?

NVIDIA's production inference framework that:
- Compiles LLM graphs with layer fusion and kernel optimization
- Generates fast CUDA kernels specific to your GPU and model shape
- Supports FP8, INT8, INT4 quantization with calibration
- Achieves the highest raw throughput on NVIDIA GPUs

Trade-off: complex setup, longer compilation time, less flexible than vLLM.

---

## 6. KV Cache

### What is the KV cache and why does it dominate memory?

During autoregressive decoding, the model computes Key (K) and Value (V) matrices for every past token in context. Instead of recomputing these every step, we cache them.

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element

Llama-3 70B, FP16, batch=32, seq=4096:
= 2 × 80 × 8 × 128 × 4096 × 32 × 2 bytes ≈ 160 GB (!)
```

> 💡 This is why batch size and context length are the primary levers for memory management in LLM serving.

---

### How do you estimate KV cache size per token?

```
KV size per token = 2 × num_layers × (num_kv_heads × head_dim) × bytes_per_element

Llama-3 8B example:
32 layers, 8 KV heads, 128 head_dim, FP16 (2 bytes)
= 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB per token

For 4096 token context × 10 concurrent users:
= 128 KB × 4096 × 10 ≈ 5 GB KV cache
```

---

### What is paged KV cache vs standard KV cache?

| Approach | How it works | Problem |
|---|---|---|
| **Standard** | Pre-allocate `max_seq_len × max_batch_size` contiguous memory | Massive waste — most sequences shorter than max. High fragmentation. |
| **Paged (vLLM)** | Allocate in fixed "pages" (e.g., 16 tokens each). Pages assigned on demand, can be non-contiguous. | Near-zero waste. Enables prefix sharing. |

> 💡 PagedAttention lets vLLM run **3–4x higher batch sizes** than naive implementations on the same GPU.

---

### What is prefix sharing / KV cache reuse?

If multiple requests share the same prefix (system prompt, few-shot examples, RAG context), compute and cache the KV for that prefix once and **share those pages** across all requests.

- Called "radix attention" in SGLang, "prefix caching" in vLLM
- Reduces both TTFT and memory for workloads with common prefixes
- Works especially well for chatbots with fixed system prompts

---

### How do long conversations impact KV memory?

Each additional turn adds tokens to context. KV cache grows **linearly** with context length.

```
50-turn chat × 200 tokens/turn = 10,000 tokens of KV to store and attend over
```

**Strategies to control growth:**

1. **Truncate** oldest turns when context limit is reached
2. **Summarize** old context and replace raw turns with summary
3. **Sliding window attention** — only attend to last N tokens
4. **KV eviction** — evict least-recently-used cache entries
5. **Cap max context** — enforce a hard limit per session

---

### How does cache eviction work?

When KV cache is full, evict (discard) pages from sequences with lowest priority:

- **LRU (Least Recently Used):** Evict sequences not recently decoded
- **Beam search eviction:** Drop lower-scoring beams first
- **Priority-based:** Keep premium user sessions, evict free tier first
- **Preemption:** Swap KV pages to CPU RAM temporarily (slower but recoverable)

---

## 7. Batching & Scheduling

### What is static, dynamic, and continuous batching?

| Type | How it works | Problem |
|---|---|---|
| **Static batching** | Wait until exactly N requests arrive, process all together | Slow requests block entire batch. GPU idles when sequences finish unevenly. |
| **Dynamic batching** | Wait up to T ms or N requests, whichever first | Still has head-of-line blocking. |
| **Continuous batching** | New requests join mid-generation. Finished slots immediately filled by new requests. | Near-optimal GPU utilization. Used by vLLM, TGI. |

> 💡 Continuous batching is the key innovation that made production LLM serving practical. Without it, GPUs are mostly idle.

---

### Why is continuous batching better for LLMs specifically?

LLM outputs are **variable length** — some requests generate 10 tokens, some 1000. Static batching forces everyone to wait for the longest sequence.

With continuous batching:
- Short requests don't get blocked behind long ones
- GPU utilization stays consistently high
- Enables much higher effective throughput

```
Throughput improvement: typically 5–23x over static batching
(Source: Orca paper, 2022)
```

---

### How does batching affect latency vs throughput?

```
Larger batch → higher throughput, higher per-request latency
Smaller batch → lower latency, lower throughput, more GPU waste
```

This is the fundamental **latency-throughput tradeoff**. Tune based on your SLO:

- Strict latency SLO (P99 < 500ms) → keep batch size small
- Cost/throughput optimization → push batch size up until latency SLO is hit

---

### How do you tune max batch size?

Max batch size is constrained by **GPU memory** (specifically, KV cache).

```
Step 1: Load model onto GPU, note remaining VRAM
Step 2: Estimate KV cache per request:
         KV_per_request = max_seq_len × kv_size_per_token
Step 3: Max batch = remaining VRAM / KV_per_request
Step 4: Benchmark latency at different batch sizes
Step 5: Find the "knee" — where throughput gains flatten but latency spikes
```

---

### How do you avoid head-of-line blocking?

- **Continuous batching** — short requests don't wait for long ones to finish
- **Max generation limits** — cap output tokens to prevent runaway long generations
- **Priority queues** — route short/simple requests to dedicated workers
- **Preemption** — pause low-priority sequences to free slots for high-priority ones
- **Disaggregated prefill/decode** — separate servers for prompt processing and token generation

---

### What is fair scheduling and how do you prioritize premium users?

**Strategies for premium user prioritization:**

1. **Dedicated GPU pools** — no sharing with free tier
2. **Weighted fair scheduling** — premium users get higher queue weights
3. **Preemption** — scheduler pauses lower-priority in-flight requests; KV state saved to CPU memory temporarily
4. **Separate rate limits** — different RPS caps and max queue depths per tier
5. **SLO-based scheduling** — track each user's SLO, prioritize those most at risk of violating it

---

## 8. ML Serving Architecture

### How do you horizontally scale ML inference services?

| Strategy | When to use |
|---|---|
| Stateless replicas + load balancer | For models with no session state |
| Session affinity (sticky routing) | For stateful workloads (chat with KV cache) |
| Tensor parallelism (split model across GPUs) | When model doesn't fit on one GPU |
| Pipeline parallelism (split layers across nodes) | Very large models, multi-node |
| Autoscaling on GPU utilization or queue depth | Variable traffic patterns |

> 💡 LLM serving is tricky to scale horizontally because **KV cache is stateful**. New approaches: KV cache disaggregation, prefill/decode separation.

---

### What is tensor parallelism vs pipeline parallelism?

| Type | How it works | Best for |
|---|---|---|
| **Tensor parallelism** | Split each weight matrix across GPUs. All GPUs participate in every layer. Low latency. | Same-node GPUs with NVLink |
| **Pipeline parallelism** | Split layers across GPUs. GPU 1 does layers 1–10, GPU 2 does 11–20. Has pipeline bubbles. | Multi-node deployments |

```
Practical rule:
70B on 4 GPUs (same node) → tensor parallelism
70B+ across nodes         → tensor + pipeline parallelism combined
```

---

### What is the difference between synchronous and asynchronous inference?

| Type | How it works | Best for |
|---|---|---|
| **Synchronous** | Client blocks until full response arrives | Simple request-response APIs |
| **Asynchronous** | Client gets request ID immediately. Uses polling, webhook, or streaming. | Long LLM generations, streaming |

> 💡 All good LLM APIs use **streaming (SSE or WebSocket)** — users see tokens as they generate. This dramatically reduces perceived latency even if total generation time is unchanged.

---

### How do you manage model versioning with zero downtime?

| Strategy | How it works |
|---|---|
| **Blue-green deployment** | Run v1 and v2 simultaneously, shift traffic gradually |
| **Canary release** | Send 1% of traffic to new model, monitor, gradually increase |
| **Shadow mode** | New model runs on all traffic but responses discarded; compare outputs offline |
| **Feature flags** | Control rollout per user/cohort programmatically |

---

### How do you cache predictions?

**Exact match caching:** Store (input hash → output) in Redis/Memcached. Works for deterministic queries (temperature=0) with repeated prompts.

**Semantic caching:** Embed the query and find nearest-neighbor cached responses. Useful for similar but not identical queries.

**Trade-offs:**
- Cache hit rate depends on query diversity
- Staleness risk if model updates
- Storage cost for large outputs

---

### When should you use ONNX?

ONNX (Open Neural Network Exchange) exports models to a common format for inference runtime optimization.

**Use ONNX when:**
- Serving traditional ML/DL models (BERT, ResNet, XGBoost)
- You need cross-platform portability (ONNX Runtime supports CPU, GPU, mobile, edge)
- Using ONNX Runtime for graph optimization on CPU

**Don't use ONNX for:**
- LLM serving — vLLM/TensorRT-LLM are better choices

---

### What metrics should trigger autoscaling?

| Metric | Threshold | Action |
|---|---|---|
| GPU utilization | > 80% sustained | Scale out |
| Request queue depth | > N requests | Scale out |
| P95 latency | > SLO threshold | Scale out |
| Tokens/sec per GPU | Degrading | Investigate + scale |
| GPU utilization | < 20% sustained | Scale in (cost savings) |

> 💡 Queue depth is often the most responsive signal — it rises before latency visibly degrades.

---

## Quick Reference Cheat Sheet

### Memory Estimation

```
Model weights (FP16) = params × 2 bytes
Model weights (INT4)  = params × 0.5 bytes

KV per token = 2 × layers × kv_heads × head_dim × 2 bytes

Total VRAM = weights + (KV per token × max_seq × batch_size) × 1.2
```

### Latency Decomposition

```
E2E latency = TTFT + (TPOT × output_tokens)
TTFT        = prefill time (compute-bound, O(n²))
TPOT        = decode time (memory-bandwidth-bound)
```

### Little's Law

```
Concurrency = RPS × Latency(sec)
```

### Batch Size Upper Bound

```
Max batch ≈ free_VRAM / (max_seq_len × kv_size_per_token)
```

---
# FAANG ML Engineer Interview Prep — Topics 8 to 14

> **Audience:** 5+ years ML/AI engineer preparing for FAANG system design interviews.
> **Goal:** Simple, clear answers you can explain confidently to an interviewer.

---

## 8. Multi-GPU Scaling

### Parallelism Strategies

**Q: What is data parallelism?**

You split the training data across multiple GPUs, but each GPU holds a full copy of the model. Each GPU processes a different mini-batch, computes gradients, then all GPUs sync their gradients using all-reduce.

- Simple and works great when the model fits in one GPU.
- Think of it like 4 chefs cooking the same recipe independently, then averaging their feedback.

---

**Q: What is tensor parallelism?**

You split individual layers (matrices) across GPUs. For example, a big weight matrix gets split column-wise across 4 GPUs. Each GPU processes its shard, then results are combined.

- Used when a single layer is too large to fit in one GPU's memory.
- Megatron-LM pioneered this for transformers.

---

**Q: What is pipeline parallelism?**

You split the model layer-by-layer across GPUs. GPU1 holds layers 1–8, GPU2 holds layers 9–16, etc. Data flows like an assembly line.

- The challenge is pipeline bubbles (idle GPUs waiting).
- GPipe and PipeDream solve this with micro-batching.

---

**Q: What is expert parallelism?**

Used in Mixture-of-Experts (MoE) models. Each GPU hosts a different set of "experts" (sub-networks). A router decides which expert handles each token.

- GPUs only activate a subset of experts per token.
- You get huge model capacity without proportionally huge compute.

---

**Q: What is context parallelism?**

You split the input sequence length across GPUs. For very long contexts (128K+ tokens), attention computation is split so each GPU processes a slice of the sequence.

- Ring attention is a common implementation.
- Needed when context is too long for one GPU's memory.

---

### Design Decisions

**Q: When do you use tensor parallelism?**

When a single layer (like a huge attention head or FFN) doesn't fit in one GPU's memory. Also when you need tight latency and can afford fast interconnects like NVLink.

- It adds communication overhead on every forward pass.
- Only use it when layer size demands it.

---

**Q: What are the communication costs?**

| Strategy | Communication Pattern | Frequency |
|---|---|---|
| Data parallelism | All-reduce on gradients | Once per backward pass |
| Tensor parallelism | All-reduce on activations | Every layer (very frequent) |
| Pipeline parallelism | Point-to-point between stages | Once per micro-batch |

Tensor parallelism has the highest bandwidth demand — needs NVLink or InfiniBand to not bottleneck.

---

**Q: How does NCCL work?**

NCCL (NVIDIA Collective Communications Library) handles GPU-to-GPU communication efficiently.

- Auto-selects the best algorithm (ring, tree, broadcast) based on topology.
- Uses NVLink for intra-node and InfiniBand/RoCE for inter-node.
- Think of it as the networking layer for GPU clusters.

---

**Q: How does NVLink impact performance?**

NVLink is NVIDIA's high-speed GPU interconnect — up to 600 GB/s bidirectional in H100 systems, vs PCIe at ~64 GB/s.

- This ~10x bandwidth difference is critical for tensor parallelism.
- Without NVLink, tensor parallelism is often slower than not using it at all.

---

**Q: How do you reduce all-reduce overhead?**

1. **Gradient compression** — use FP16 or quantized gradients.
2. **Gradient bucketing** — accumulate multiple gradients and send one large all-reduce (PyTorch DDP does this automatically).
3. **Overlap communication with computation** — start all-reduce for early layers while later layers are still computing backward.
4. **ZeRO optimizer (DeepSpeed)** — shard optimizer states across GPUs to reduce per-GPU memory and communication volume.

---

## 9. Autoscaling LLM Services

**Q: What metrics should drive autoscaling?**

- **Primary:** Request queue length and queue wait time (best signal).
- **Secondary:** GPU utilization — tricky because high utilization is fine unless the queue is also growing.
- **Also track:** Time-to-first-token (TTFT), tokens-per-second, requests-in-flight per replica.

---

**Q: How do you scale based on queue length?**

Set a target queue depth per replica (e.g., max 5 pending requests). If queue > 5 × num_replicas, spin up new replicas.

- Use a PID controller or KEDA (Kubernetes Event-Driven Autoscaler) with a queue-length metric source.
- Scale-down is lagged by 2–5 min to avoid flapping.

---

**Q: How do you predict traffic spikes?**

1. Time-series forecasting on historical traffic (business hours, Monday spikes, etc.).
2. Pre-scale before known events (releases, marketing campaigns).
3. Use leading indicators — like API key creation rates or frontend page loads that precede LLM calls.
4. Tools: Prophet, or simple scheduled cron-based scaling rules.

---

**Q: How do you handle GPU cold starts?**

GPU cold start (loading model weights) takes 30–180 seconds. Mitigate by:

1. Keeping minimum replicas > 0.
2. Pre-loading model weights into CPU RAM so GPU loading is faster.
3. Using spot instance pools with pre-loaded AMIs/images.
4. Storing weights on fast NVMe or RDMA storage, not S3.

---

**Q: How do you maintain model warm pools?**

Keep a small pool of "warm" replicas (model loaded, ready to serve) beyond your current load.

- Scale this pool based on your P99 spike multiplier.
- Cost: you pay for idle GPU time.
- Trade-off: pay ~10–20% extra compute to eliminate cold-start latency entirely.
- Warm pool size formula: `max_burst / avg_throughput_per_replica`

---

**Q: How do you balance cost and latency?**

Use a two-tier fleet:

| Tier | Instance Type | Latency SLO | Cost |
|---|---|---|---|
| Baseline | On-demand / Reserved | P99 TTFT < 500ms | Higher |
| Burst | Spot instances | P99 TTFT < 5s | 60–80% cheaper |

Route latency-sensitive requests to on-demand. Batch/async jobs go to spot.

---

## 10. Cost Optimization for LLM Inference

**Q: How do you calculate cost per million tokens?**

```
cost_per_M_tokens = (GPU_hourly_cost × inference_time_per_token × 1,000,000) / batch_size
```

**Example:**
- H100 at $4/hr, model generates 50 tok/sec at batch=1 → $22/M tokens
- Same GPU with batch=16 at 400 tok/sec → $2.8/M tokens
- **Batching is your biggest cost lever.**

---

**Q: How do you compare GPUs economically?**

Use **tokens-per-dollar** as your metric, not raw TFLOPS.

| GPU | $/hr | Tok/sec | Tokens per $0.01 |
|---|---|---|---|
| A100 | $2 | 200 | 100 |
| H100 | $4 | 500 | 125 |

H100 wins economically despite 2x price. Always benchmark your specific model — memory bandwidth often matters more than compute for inference.

---

**Q: When is a smaller model better?**

When:
1. Task quality threshold is met by the smaller model (test this first).
2. Latency is critical — smaller models are 2–5x faster.
3. You're doing high-volume, simple tasks (classification, extraction).

> Rule of thumb: Use the smallest model that meets your quality bar. A 7B model at 10% of the cost that's 90% as good is usually the right call.

---

**Q: How does quantization reduce cost?**

Quantization reduces weight precision: FP16 → INT8 → INT4.

- INT8 roughly halves memory → fit a bigger batch or larger model on fewer GPUs.
- Throughput improves because memory bandwidth is the bottleneck for inference.
- Cost reduction: ~30–50% with INT8, ~50–70% with INT4.
- Quality degradation: typically 1–5%.
- Tools: GPTQ, AWQ for LLMs.

---

**Q: How does prompt compression reduce cost?**

Shorter prompts = fewer input tokens = less prefill time and lower cost. Techniques:

1. LLMLingua or similar to compress long system prompts.
2. Truncate retrieved context to only relevant chunks.
3. Summarize long conversation history instead of sending raw.

For long prompts (10K+), prefill compute adds up significantly.

---

**Q: How do cache hits reduce cost?**

| Cache Type | What Is Cached | Savings |
|---|---|---|
| KV-cache prefix | System prompt KV states (vllm, TRT-LLM) | 20–40% prefill compute |
| Result cache | Full response for identical prompts (Redis) | 100% compute for repeat queries |
| Embedding cache | Pre-computed doc/query embeddings | 100% embedding compute for known inputs |

Each cache layer can cut 20–60% of redundant compute.

---

**Q: How do you optimize idle GPU utilization?**

1. **Continuous batching** — don't wait for a full batch; keep adding requests as slots free up (vllm does this by default).
2. **Background jobs** — run indexing, reranking, or evaluation tasks on idle GPU capacity.
3. **Multi-tenant serving** — share GPUs across multiple model variants with time-slicing.
4. **Right-size your fleet** — use autoscaling so you're not paying for idle replicas overnight.

---

## 11. Self-Hosted Vector Database Performance

### Fundamentals

**Q: What is approximate nearest neighbor (ANN)?**

Instead of comparing your query against every vector (exact search), ANN finds neighbors that are "close enough" using smart indexing.

- Trade a small drop in recall for huge speed gains.
- At 100M vectors: exact search takes seconds; ANN takes milliseconds.
- FAISS, HNSW, IVF are all ANN methods.

---

**Q: What is exact search?**

Brute-force: compare query against every single vector.

- 100% recall, but O(N × d) cost — scales terribly.
- Only practical for < 100K vectors.
- Used as a ground truth baseline to measure ANN recall.

---

**Q: What is HNSW?**

Hierarchical Navigable Small World — a graph-based ANN index.

- Builds a layered graph where top layers are sparse (fast navigation), bottom layers are dense (precision).
- Query: enter at top layer, greedily traverse to nearest neighbors.
- Best recall/latency trade-off for in-memory search.
- RAM-heavy: needs ~100–200 bytes per vector extra for graph edges.

---

**Q: What is IVF?**

Inverted File Index. Clusters vectors into N centroids (like k-means).

- At query time, only search the top-k nearest clusters, not all vectors.
- Parameter `nprobe` = how many clusters to search (higher = better recall, slower).
- Good for large datasets where you can't afford HNSW's RAM overhead.

---

**Q: What is PQ (Product Quantization)?**

Compresses vectors by splitting them into sub-vectors and quantizing each to a codebook.

- Reduces memory 8–64x. E.g., a 768-dim float32 vector (3KB) → ~96 bytes with PQ.
- Recall drops slightly.
- Usually combined with IVF (IVF-PQ) for large-scale search with limited RAM.

---

**Q: What is OPQ?**

Optimized Product Quantization — a rotation applied to vectors before PQ to make sub-vectors more independent.

- Gives 5–15% better recall at the same compression ratio.
- Adds a preprocessing step.
- Use when PQ recall is insufficient for your quality bar.

---

**Q: What is DiskANN?**

An ANN algorithm designed to store the index on SSD, not RAM.

- Uses a Vamana graph structure optimized for sequential disk reads.
- Enables billion-scale search with low RAM (cache only a fraction of the index).
- Latency: 2–10ms (vs sub-ms for in-memory HNSW).
- Much lower cost per GB than in-memory approaches.

---

### Key Metrics

**Q: What is recall@k?**

The fraction of true top-k nearest neighbors your ANN index actually returns.

- `recall@10 = 0.95` means your ANN finds 9.5 of the true 10 nearest neighbors on average.
- For most production RAG, `recall@10 > 0.90` is acceptable.
- Measured against brute-force (exact search) ground truth.

---

**Q: What is search latency?**

Time from query vector to returned results.

- Target: < 10ms P99 for interactive use, < 100ms for batch.
- Affected by: index type, `nprobe`, `ef_search`, number of results, metadata filters, and hardware.

---

**Q: What is indexing throughput?**

How many vectors/sec you can insert into the index.

- HNSW indexing is slow (~1K–10K vec/sec).
- IVF-PQ is faster to build but needs a training phase upfront.
- Critical when you need real-time or near-real-time index updates.

---

**Q: What is memory footprint per vector?**

| Format | Bytes per vector (768-dim) |
|---|---|
| Raw float32 | 768 × 4 = 3,072 bytes (~3KB) |
| HNSW (with graph) | ~3KB + 100–200 bytes graph overhead |
| IVF-PQ compressed | 32–128 bytes |

For 100M × 768-dim vectors: raw = ~300GB, PQ-compressed ≈ 3–10GB.

---

**Q: What is QPS?**

Queries per second — throughput of your vector search system.

- A well-tuned HNSW on a 32-core CPU can handle 1K–5K QPS for 1M vectors.
- Scale horizontally by sharding or replicating the index across QueryNodes.

---

### Capacity Planning

**Q: How do you estimate RAM for 100M embeddings?**

```
RAM = num_vectors × dim × 4 bytes (float32)
    + HNSW graph overhead (~100 bytes/vec)

Example: 100M × 768 × 4 = 307 GB raw
         + 100M × 100  = 10 GB graph
         Total ≈ 317 GB
```

With PQ compression (8 bytes/vec): 100M × 8 = ~0.8 GB for quantized index (but raw vectors still needed for re-scoring).

Practical planning number: **300–400 GB RAM for raw in-memory HNSW at 100M × 768d**.

---

**Q: How do embedding dimensions affect storage?**

Linear relationship: 2x dimensions = 2x storage.

| Model | Dimensions | Storage for 100M vectors |
|---|---|---|
| MiniLM | 384 | ~150 GB |
| BGE-base, E5-base | 768 | ~300 GB |
| OpenAI ada-002 | 1536 | ~600 GB |

Lower-dim models are cheaper to store and search, with modest quality trade-off.

---

**Q: How do metadata filters affect performance?**

- **Post-filtering** (search then filter): wastes ANN effort if filter is highly selective.
- **Pre-filtering** (filter then search): risks missing good results.
- **Best approach:** Use indexed payload fields alongside the vector index. Qdrant and Milvus push filters into the ANN graph traversal.
- Highly selective filters (< 1% pass rate) can increase latency 5–20x without proper indexing.

---

## 12. Vector Database-Specific Questions

### Milvus

**Q: How does Milvus scale horizontally?**

Milvus separates storage, query, and indexing into independent services.

- Scale **QueryNodes** (search workers) independently from **DataNodes** (ingestion workers).
- Shared object storage (S3/MinIO) decouples compute from data.
- Add more QueryNodes for higher QPS; add more DataNodes for higher insert throughput.

---

**Q: What roles do QueryNode and DataNode play?**

| Node | Responsibility |
|---|---|
| DataNode | Ingestion, WAL, segment compaction, writes to object store |
| QueryNode | Loads segments into memory, serves search requests |

They scale independently — e.g., 2 DataNodes + 20 QueryNodes for a read-heavy workload.

---

**Q: How does Milvus handle sharding?**

- Collections are split into shards (default: 2).
- Each shard maps to a DML channel (message queue).
- Shards distribute write traffic across DataNodes.
- For very high write throughput, increase shard count.

---

**Q: How do you tune HNSW parameters in Milvus?**

| Parameter | Range | Effect |
|---|---|---|
| `M` | 8–64 | Higher = better recall, more RAM, slower build |
| `ef_construction` | 100–500 | Higher = better recall at build time, slower indexing |
| `ef_search` (query) | 50–500 | Controls recall vs latency at query time |

Recommended starting point: `M=16`, `ef_construction=200`, then tune `ef_search` until you hit your recall target.

---

### FAISS

**Q: How do FAISS indexes differ?**

| Index | Best For | Trade-off |
|---|---|---|
| `Flat` | Small datasets, exact search | Highest recall, O(N) cost |
| `IVF_Flat` | Large datasets, limited RAM | Fast ANN, no compression |
| `IVF_PQ` | Very large datasets, low RAM | Most memory-efficient, slight recall drop |
| `HNSW_Flat` | Best recall/latency trade-off | High RAM usage |

Decision rule: small data → `Flat`; large RAM → `HNSW`; limited RAM → `IVF_PQ`.

---

**Q: When should you use GPU FAISS?**

GPU FAISS (`faiss-gpu`) is 10–50x faster than CPU for IVF indexes.

- **Best for:** Offline batch search, re-ranking candidate pools, maximum throughput scenarios.
- **Not ideal for:** Interactive low-latency search — CPU HNSW often has better P99 latency due to GPU overhead per query.

---

## 13. Embedding Performance

**Q: How long does embedding generation take?**

| Hardware | Throughput | Latency per query |
|---|---|---|
| A100 GPU | ~1,000–2,000 chunks/sec | ~5–10ms |
| T4 GPU | ~300–600 chunks/sec | ~10–20ms |
| CPU | ~10–50 chunks/sec | ~50–200ms |

Batch size assumed: 32 chunks at 512 tokens each. For real-time RAG, query embedding takes ~5–20ms on GPU — acceptable. Indexing millions of docs is an offline batch job.

---

**Q: How does batch size affect embedding throughput?**

Throughput roughly doubles from batch=1 to batch=32, then plateaus.

- Sweet spot: **batch=32–64** for most embedding models.
- batch=256 may OOM depending on model and GPU.
- Use dynamic batching to group incoming requests without adding latency.

---

**Q: How do you cache embeddings?**

| Cache Type | Strategy | When To Use |
|---|---|---|
| Document embeddings | Compute once, store in vector DB | Always — never recompute unless doc changes |
| Query embeddings | Hash query → Redis with TTL (e.g., 1 hour) | When repeated identical queries are common |
| Semantic cache | Embed query, check for semantically similar past queries | When near-duplicate queries are frequent |

---

**Q: How do you choose embedding dimension?**

| Model | Dimensions | Use Case |
|---|---|---|
| MiniLM, e5-small | 384 | Storage-constrained, ~80% quality |
| BGE-base, E5-base | 768 | General English RAG sweet spot |
| BGE-large, E5-large | 1024 | Higher quality, more cost |
| Multilingual models | 768–1024 | Non-English or mixed-language |

> Don't choose based on dimension alone — benchmark on your domain using MTEB scores.

---

**Q: How do you update embeddings incrementally?**

1. **Append** new vectors to the index without rebuilding (HNSW supports this).
2. Track `doc_id → vector_id` mapping to handle updates: delete old, insert new.
3. For IVF indexes: rebuild centroids periodically (or use a freshness threshold) to maintain recall quality.
4. For large updates (> 10% of index), a background rebuild may be needed.

---

## 14. RAG Pipeline Performance

### Pipeline Latency Budget

**Q: What is the latency budget for each RAG stage?**

Target: **P50 end-to-end < 2 seconds** for interactive use.

| Stage | Target Latency |
|---|---|
| Query preprocessing (cleaning, expansion) | < 10ms |
| Embedding generation | 10–30ms |
| Vector search (in-memory HNSW) | 5–20ms |
| Reranking (cross-encoder, top-20) | 50–200ms |
| Prompt assembly | < 5ms |
| LLM generation (time-to-first-token) | 200–500ms |
| **Total (first token to user)** | **~300–800ms** |

> Parallelize embedding + vector search where possible. Reranking and LLM generation are the dominant latency drivers.

---

### Retrieval Optimization

**Q: How do you reduce retrieval latency?**

1. Use in-memory HNSW instead of disk-based index.
2. Reduce `nprobe` / `ef_search` with acceptable recall trade-off.
3. Pre-filter by metadata before vector search to reduce the candidate set.
4. Shard index across QueryNodes and aggregate results.
5. Cache frequent query embeddings.
6. Use smaller embedding models (384d vs 768d = ~2x faster search).

---

**Q: How do you tune top-k?**

Retrieve top-k candidates from vector DB, then rerank to top-n for the LLM context.

- General rule: `k = 3–5x` your final context window size.
- Example: want 5 chunks in LLM context → retrieve top-20, rerank to top-5.
- Higher k = better recall but slower reranking.
- Profile: if reranking 20 docs takes 150ms but improves answer quality by 10%, it's likely worth it.

---

**Q: How do metadata filters affect speed?**

- Well-indexed metadata filter: adds < 1ms.
- Unindexed string-match filter: can add 50–500ms.
- **Always index fields you filter on** (Qdrant and Milvus support indexed payload fields).
- Filters like "only last 30 days" or "only tenant X" reduce search space but hurt performance if done naively.

---

**Q: How do hybrid retrieval methods affect latency?**

Hybrid = dense (vector) + sparse (BM25/keyword) retrieval, merged with RRF or weighted fusion.

- Adds one extra sparse search pass: ~5–20ms on Elasticsearch.
- Result fusion is O(k log k) — negligible overhead.
- **Total overhead: ~20–30ms.**
- Worth it for domains with exact-match needs (product codes, names, IDs) where dense retrieval alone misses.

---

### Reranking

**Q: What is the latency cost of cross-encoders?**

Cross-encoders (BGE-Reranker, Cohere Rerank) score each (query, doc) pair jointly.

| Hardware | Latency (top-20 candidates, 512 tokens each) |
|---|---|
| GPU | 80–150ms |
| CPU | 300–800ms |

This is often the second biggest latency source in a RAG pipeline after LLM generation. Always run reranking on GPU if latency matters.

---

**Q: When should reranking be skipped?**

Skip reranking when:
1. Latency budget is tight (< 500ms total) and quality delta is small.
2. Retrieved chunks are already highly precise (narrow, clean corpus).
3. Using a high-quality large embedding model (large E5 or GTE) with high baseline recall.

> Test: run A/B with and without reranking on your domain. If quality improvement < 5% and latency cost > 150ms, skip it.

---

**Q: How do you batch reranking requests?**

Group multiple users' top-k results into one cross-encoder batch call.

- Example: 10 users each need reranking of top-20 docs → batch all 200 (query, doc) pairs in one GPU forward pass.
- GPU utilization: from ~20% to ~80%.
- Adds slight queue wait time but massively improves throughput.
- Implementation: async queue with a max batch wait of 20–50ms.

---
# FAANG LLM Systems Design — Interview Prep (Sections 15–21)

> **Audience:** AI/ML Engineer (5 YOE) preparing for FAANG interviews
> **Topics:** Chat Retrieval · Caching · Streaming · Concurrency · Rate Limiting · Load Balancing · Observability

---

## 15. Chat-Retrieval Speed

### Q: How do you optimize multi-turn retrieval?

**TL;DR:** Don't retrieve from scratch each turn. Reuse, merge, and query-compress.

In a multi-turn chat, re-running full retrieval every turn is expensive and often redundant. Optimize by:

- **Query compression:** Instead of sending the full conversation to the retriever, first run it through a small LLM to generate a single compressed query. E.g., "User asked about X, then Y → retrieve docs about XY intersection."
- **Delta retrieval:** Only retrieve if the new user turn introduces genuinely new intent or entities.
- **Context window reuse:** If retrieved docs from turn 2 are still relevant at turn 4, don't throw them away — carry them forward.
- **Result merging:** Merge new retrieval results with prior ones using dedup + relevance score.

> **Interviewer tip:** Say "I separate retrieval into a *necessity check* step and a *query reformulation* step before hitting the vector DB."

---

### Q: How do you summarize conversation history?

**TL;DR:** Sliding window + periodic LLM summarization keeps context lean.

Long conversation history inflates token count and degrades retrieval quality. Common approaches:

- **Sliding window:** Keep only the last N turns verbatim, drop older ones.
- **Summarization buffer:** When history exceeds a threshold (e.g., 10 turns), run a background LLM call: *"Summarize these 10 turns in 3 sentences."* Use the summary as compressed history.
- **Entity memory:** Extract key entities/facts from old turns (user name, prior preferences) and store them separately as structured state — not raw text.
- **Hybrid:** Maintain a rolling summary + last 3 turns verbatim. The verbatim recent context ensures coherence; the summary provides background.

> **LangChain reference:** `ConversationSummaryBufferMemory` does exactly this pattern.

---

### Q: How do you reuse prior retrieval results?

**TL;DR:** Cache retrieval results keyed on compressed query. Re-score instead of re-retrieve.

- **Session-level retrieval cache:** Store `{query_hash → [doc_ids, scores]}` in Redis per session.
- **Re-ranking over cached docs:** When a new turn is related, re-score the cached docs against the new query using a cross-encoder reranker — much cheaper than hitting the vector DB again.
- **Append-only result pool:** Maintain a pool of retrieved docs for the session. New turns add to the pool; the LLM reads the top-k from the pool.

> **Key insight:** Retrieval is the expensive step. Reranking over cached results is ~10x cheaper than full ANN search.

---

### Q: How do you cache frequent queries?

**TL;DR:** Semantic cache: embed the query → check nearest neighbor in cache store.

Two flavors of query caching:

- **Exact cache:** Hash the query string, store result in Redis. Works only for repeated identical queries (e.g., FAQ bots).
- **Semantic cache:** Embed the query, store embedding + result. On a new query, compute embedding, find nearest neighbor in the cache. If cosine similarity > threshold (e.g., 0.95), serve cached result. Tools: `GPTCache`, `LangChain SemanticCache`.

**TTL matters:** Set TTL based on how frequently your docs change.
- Static knowledge base → 24h TTL
- Live data → 5–10 min TTL

> **Gotcha:** Semantic cache can return wrong results if threshold is too low. Monitor cache hit *quality*, not just hit rate.

---

### Q: How do you avoid retrieving on every turn?

**TL;DR:** Classify each turn — is retrieval actually needed?

Not every user message needs new docs. You add a lightweight *retrieval gate*:

- **Intent classifier:** A small model (BERT-size or even regex rules) classifies: *needs retrieval* vs *conversational* vs *clarification*.
- **Entity change detection:** If the user's new turn doesn't introduce new entities/topics vs the prior turn, skip retrieval.
- **LLM self-check (cheap):** Ask the LLM with a 1-sentence prompt: *"Does answering this require new information? Yes/No."* Use a fast small model (GPT-3.5 equivalent) for this gate.

> **Business impact:** In a 10-turn conversation, you might only need retrieval on 3–4 turns. This can cut vector DB costs by 60%.

---

### Q: How do you detect when retrieval is necessary?

**TL;DR:** Signal detection: new entity, factual gap, or explicit knowledge request.

Signals that indicate "go retrieve":

- **New named entity:** User introduces a product name, person, date, or place not in current context.
- **Factual question pattern:** "What is...", "How does X work", "When did..." → high retrieval likelihood.
- **LLM uncertainty signal:** If model's self-confidence (via logprobs or a calibration head) is low, trigger retrieval.
- **Missing context signal:** Check if the current context window already contains enough info to answer. Use a small embedding similarity check between query and current context.

> **Architecture pattern:** Think of this as a *retrieval router* — a fast, cheap classifier that gates the expensive vector search.

---

## 16. Caching Strategies

### LLM Caching

#### Q: What is semantic caching?

**TL;DR:** Match queries by meaning (embedding similarity), not exact text.

Semantic caching stores LLM responses keyed on the semantic meaning of prompts, not exact text.

1. Embed incoming query → search cache store for nearest neighbor.
2. If similarity > threshold → return cached response.
3. Handles paraphrases: "What's the capital of France?" and "France's capital?" return the same cached answer.

**Stack:** Query embedding (text-embedding-ada) → FAISS/Redis vector search → cache hit check.

**Tools:** `GPTCache`, `LangChain SemanticCache`

---

#### Q: What is exact prompt caching?

**TL;DR:** Hash the full prompt string → cache key → return stored response.

Exact prompt caching: hash the full prompt (including system message + user message) → check if hash exists in Redis → return cached LLM response if yes.

- Great for: FAQ chatbots, templated queries, static system prompts.
- Cache key: `SHA256(system_prompt + user_message)`
- Near-zero latency on hit (Redis read ~0.1ms vs LLM inference ~500ms).

> **Limitation:** Even a single character change creates a cache miss. Use semantic caching for fuzzy matching.

---

#### Q: What is prefix cache?

**TL;DR:** Cache the KV-attention values of a repeated prompt prefix to skip recomputation.

Prefix caching is a GPU-level optimization. Computing attention over the prompt (prefill) is expensive.

- If many requests share the same system prompt (e.g., long RAG context or instructions), the KV cache values for that prefix are computed **once** and reused.
- Subsequent requests sharing the prefix skip the prefill phase for it → big latency win.
- Supported in: `vLLM` (automatic prefix caching), `SGLang`, `TensorRT-LLM`.

**Example:** If your system prompt is 1000 tokens and shared across 1000 users, you compute its KV values once and reuse.

> **Limitation:** Cache lives in VRAM. High VRAM pressure can evict prefix cache entries.

---

#### Q: How do cache hit rates impact cost?

**TL;DR:** Every cache hit = zero LLM tokens billed. Even 30% hit rate = significant savings.

LLM pricing is per-token. A cache hit means you serve the response without calling the LLM.

- **Input token savings:** If cache hit rate = 40%, you're paying for only 60% of input tokens.
- **Output token savings:** Cached responses skip output generation entirely (the most expensive part).
- **Latency savings:** P99 latency drops from 2s → 5ms on cache hits.

**Formula:**
```
Cost saved = hit_rate × avg_tokens_per_request × token_price × total_requests
```

> **Interviewer note:** Always mention you'd instrument cache hit rate as a key metric on your dashboard — it directly maps to infra cost.

---

### Retrieval Caching

#### Q: How do you cache embeddings?

**TL;DR:** Store pre-computed embeddings in a KV store; never re-embed the same text twice.

- **Key:** `hash(text_chunk)` — SHA256 or MD5 of the raw text
- **Value:** Float array (the embedding vector), serialized as bytes or JSON
- **Store:** Redis with persistence, or a simple PostgreSQL table with `text_hash → embedding` column

At indexing time: check cache before calling embedding API. At query time: same — hash the query text, check if embedding exists.

> **Benefit:** Embedding API calls cost money and add 20–100ms. For a 10M doc corpus, you compute embeddings once.

---

#### Q: How do you cache vector search results?

**TL;DR:** Cache ANN search results keyed on query embedding (or its hash).

ANN search in Pinecone/Weaviate/Qdrant takes 10–200ms. For repeated queries, cache the results.

- **Key:** Hash of the query embedding vector (quantize if needed for consistency)
- **Value:** List of doc IDs + scores from last ANN search
- **TTL:** Based on index update frequency. If index is updated hourly, set TTL = 1 hour.

**Practical pattern:** Store in Redis sorted sets — doc_id as member, score as rank. Easy to retrieve top-k.

> **Warning:** Stale cache can serve outdated docs if your index updates frequently. Use short TTL or invalidate on index update.

---

#### Q: How do you cache reranker outputs?

**TL;DR:** Cache `(query, doc_id) → reranker score` pairs. Cross-encoder inference is expensive.

Rerankers (cross-encoder models) score each (query, doc) pair — this is O(k) inference calls per request, where k = candidate docs (typically 20–100).

- **Key:** `hash(query + doc_id)`
- **Value:** Relevance score (float)
- **TTL:** Long TTL if docs don't change (e.g., 24h). Short TTL for live data.

> **Why it matters:** Cross-encoder inference is 10–50x slower than embedding lookup. Caching even 20% of reranker calls gives measurable latency wins.
>
> **Challenge:** Query paraphrases won't hit the same cache key. Combine with semantic query normalization upstream.

---

### Application Caching

#### Q: When should you use Redis vs in-memory datastore?

**TL;DR:** Redis = shared, persistent, multi-process. In-memory = single process, ultra-fast, ephemeral.

| Scenario | Use |
|---|---|
| Multiple service replicas sharing cache | Redis |
| Need persistence across restarts | Redis |
| Single process, lowest possible latency | In-memory (Python dict / LRU cache) |
| Cache size fits comfortably in RAM | In-memory |
| Request-level cache within one API call | In-memory |

**Common pattern — two-tier cache:**
- L1: In-memory (sub-ms) for hot items
- L2: Redis (~1ms) for warm items
- L3: Vector DB (50–200ms) authoritative source

> **FAANG answer:** "In production with multiple replicas, I always use Redis. In-memory only as L1 or for single-node dev setups."

---

#### Q: How do TTLs affect consistency?

**TL;DR:** Short TTL = fresh data but more misses. Long TTL = faster but potentially stale.

| TTL Length | Cache Miss Rate | Staleness Risk |
|---|---|---|
| Seconds–minutes | High | Low |
| Hours–days | Low | High |

**Invalidation strategies:**

- **Lazy invalidation:** Let TTL expire naturally. Simple but potentially stale.
- **Event-based invalidation:** When docs update, explicitly delete their cache keys. Complex but fresh. Use a message queue (Kafka) to trigger cache purge.
- **Stale-while-revalidate:** Serve stale cache immediately, refresh asynchronously. Best UX + freshness balance.

> **Rule of thumb:** TTL should be ≤ acceptable staleness for your use case.

---

## 17. Streaming Performance

### Q: How does token streaming improve UX?

**TL;DR:** First token appears fast. User sees progress. Perceived wait time drops dramatically.

Without streaming: user waits 3–5s staring at a spinner, then the full response appears at once.

With streaming: first token appears in ~200ms (TTFT). User sees text generating word-by-word.

- **Perceived latency:** Even if total generation time is the same, streaming feels 3–5x faster because user gets feedback immediately.
- **User behavior:** Users start reading while text generates. Dropout/frustration rate drops significantly.
- **TTFT (Time to First Token):** The primary UX metric for streaming. Optimize this aggressively — get it under 300ms.

> **Analogy:** Like a restaurant that brings bread while you wait vs one that brings everything after 30 min.

---

### Q: How do you implement server-sent events (SSE)?

**TL;DR:** One-way HTTP stream from server to client. Simple, HTTP-native, great for LLM token streaming.

SSE uses a regular HTTP connection where the server keeps the response open and sends events as data becomes available.

**Server side (FastAPI):**
```python
from fastapi.responses import StreamingResponse

async def stream_llm(prompt: str):
    async for token in llm.astream(prompt):
        yield f"data: {token}\n\n"

return StreamingResponse(stream_llm(prompt), media_type="text/event-stream")
```

**Client side (JavaScript):**
```javascript
const es = new EventSource("/stream");
es.onmessage = (e) => appendToken(e.data);
```

| SSE Pros | SSE Cons |
|---|---|
| HTTP/1.1 compatible | One-way only (server → client) |
| Auto-reconnect built into EventSource | Not suited for bidirectional use |
| Works through most proxies/CDNs | |

---

### Q: How do WebSockets compare to SSE?

**TL;DR:** WebSockets = bidirectional, persistent, complex. SSE = one-way, simpler, HTTP-native.

| Feature | SSE | WebSocket |
|---|---|---|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | TCP (upgraded) |
| Auto-reconnect | Built-in | Manual |
| CDN/proxy support | Good | Varies |
| Complexity | Low | Higher |

**For LLM streaming:** SSE is almost always the right choice. LLM responses are one-way (server → client). WebSockets add complexity for no benefit here.

**Use WebSocket for LLMs when:** Client needs to interrupt generation mid-stream (cancel), or for real-time voice applications with bidirectional audio.

---

### Q: How do you measure perceived latency?

**TL;DR:** Track TTFT and inter-token latency, not just total generation time.

| Metric | Definition | Target |
|---|---|---|
| **TTFT** | Time from request sent → first token received | P95 < 300ms |
| **TPOT** | Time per output token (inter-token latency) | < 50ms/token |
| **E2E Latency** | Total time for complete response | Depends on use case |

**Real User Monitoring (RUM):** Instrument the client to record timestamps:
- `request_sent`
- `first_token_received`
- `last_token_received`

Report percentiles: P50, P95, P99. Use `performance.now()` in the browser or custom client-side beacons.

---

### Q: How do you handle stream interruptions?

**TL;DR:** Detect disconnect → cancel server-side generation → implement client-side retry with backoff.

- **Server-side cancellation:** When client disconnects, detect via `request.is_disconnected()` (FastAPI/Starlette) or WebSocket close event. Cancel the running generation to free GPU resources — don't let orphaned inferences run.
- **Client-side retry:** SSE's `EventSource` auto-reconnects with exponential backoff. For WebSockets, implement manual retry with backoff.
- **Resumable streams:** Assign a stream ID. Store generated tokens server-side (short TTL). On reconnect, client sends stream ID and last received token index → server replays from that point.
- **Graceful degradation:** If streaming fails after N retries, fall back to non-streaming (wait for full response).

> **Key point:** Always cancel server-side generation on disconnect. Orphaned inference wastes GPU memory and throughput.

---

## 18. Concurrency Management

### Q: How many concurrent chats can one GPU support?

**TL;DR:** Depends on model size, KV cache per session, and batch size. Typically 10–200 for LLMs.

No fixed number — it depends on:

- **Model size:** 7B model on 80GB A100 leaves ~50GB for KV cache. 70B model leaves almost none.
- **Sequence length:** Longer context = more KV cache per session.
- **Batch size:** vLLM's PagedAttention lets you batch efficiently by dynamically sharing memory.

**Rough estimate:** With a 7B model on one A100 (80GB), using vLLM with PagedAttention:
- At 2K context → 50–200 concurrent sessions
- At 32K context → 10–30 concurrent sessions

**Bottleneck order:** VRAM (KV cache) → Compute (tokens/sec) → Network bandwidth

> **Interview answer:** "I'd profile the KV cache footprint per session and model VRAM usage, then use vLLM's continuous batching to maximize GPU utilization."

---

### Q: How do you estimate KV cache usage per session?

**TL;DR:** `KV cache size = 2 × layers × heads × head_dim × seq_len × dtype_bytes`

KV cache stores key and value tensors for every transformer layer, for every token in the sequence.

```
KV_cache_bytes =
  2 (K + V)
  × num_layers
  × num_heads
  × head_dim
  × seq_len
  × bytes_per_element  (2 for fp16)
```

**Example — LLaMA-2 7B** (32 layers, 32 heads, head_dim=128, seq_len=4096, fp16):
```
= 2 × 32 × 32 × 128 × 4096 × 2 = ~2 GB per session
```

**Practical implication:** At 2GB/session on an 80GB GPU with 7B model weights (~14GB), you have ~66GB left → ~33 concurrent sessions at 4K context.

> **vLLM's PagedAttention** addresses fragmentation by allocating KV cache in pages (like OS virtual memory).

---

### Q: How do you enforce max tokens per user?

**TL;DR:** Token counting middleware at the API gateway. Block or queue when budget exceeded.

Token budgets are enforced at multiple levels:

- **API gateway layer:** Count tokens in the incoming request using a fast tokenizer (`tiktoken`). Reject if over quota before hitting the model.
- **Prompt truncation:** If input context exceeds limit, truncate oldest messages from conversation history first (keep system prompt + recent turns).
- **Max new tokens param:** Always set `max_new_tokens` or `max_tokens` in model config. Hard limit on output length.
- **Per-user counters:** Track rolling token usage per user in Redis. Increment on each request. Return 429 when budget hit.

```
[Request] → [Token counter middleware] → [Rate limiter] → [Model inference]
```

---

### Q: How do you queue requests fairly?

**TL;DR:** Priority queue with fairness: per-user FIFO, avoid starvation, tier-aware prioritization.

A naive FIFO queue lets heavy users monopolize the GPU. Fair queuing alternatives:

- **Weighted Fair Queuing (WFQ):** Each user gets a weight. GPU time allocated proportionally.
- **Per-user FIFO queues:** Separate queue per user. Round-robin across user queues. Ensures every user gets a turn.
- **Priority tiers:** Enterprise > Pro > Free. Enterprise requests jump the queue but fair queuing within each tier.
- **Head-of-line blocking prevention:** Large requests shouldn't block small ones. Break very long requests into chunks or put them in a "low priority" lane.

**Tools:** Celery with priority queues, custom Redis-backed priority queue, or vLLM's built-in scheduler.

---

### Q: How do you avoid out-of-memory errors?

**TL;DR:** Predict memory before accepting request. Reject or queue if insufficient VRAM.

OOM on GPU crashes the entire serving process. Prevention > recovery.

- **Pre-flight memory check:** Before running inference, estimate required KV cache (from input length) and check available VRAM. Reject with 503 if not enough.
- **vLLM PagedAttention:** Eliminates KV cache fragmentation. Allocates pages dynamically, enables much higher utilization without OOM.
- **Request queue with memory budget:** Admit requests only up to X% VRAM capacity. Queue the rest.
- **Graceful eviction:** If VRAM pressure rises, evict KV caches of idle/long-waiting sessions to swap (host RAM).
- **Monitoring + alerts:** Alert at 85% VRAM utilization. Horizontal scale-out before hitting 95%.

> **vLLM's approach:** Treats VRAM like OS virtual memory — pages in/out as needed. The modern standard for production LLM serving.

---

## 19. Rate Limiting and QoS

### Q: How do you implement token-based quotas?

**TL;DR:** Count tokens per request, accumulate per user/org in Redis, enforce hard limits.

Token-based quotas are more accurate than request-count quotas for LLMs (different requests have wildly different token counts).

- **Count input + output tokens:** Use `tiktoken` for fast tokenization before inference. Count output tokens from model response metadata.
- **Redis counter:** `INCRBY user:{id}:tokens:daily {token_count}` with TTL set to end-of-day. Atomic increment is critical to avoid race conditions.
- **Quota tiers:** Free = 100K tokens/day, Pro = 10M/day, Enterprise = unlimited.
- **Soft vs hard limits:**
  - Soft limit (90%) → warn user
  - Hard limit (100%) → reject with 429 + `Retry-After` header

> **Important:** Count tokens before and after. Pre-flight uses estimated output tokens (from `max_tokens` setting) for reservation, then adjust after actual response.

---

### Q: What is a token bucket algorithm?

**TL;DR:** Tokens fill at a constant rate. Each request consumes tokens. Allows bursts up to bucket capacity.

The token bucket is the standard rate limiting algorithm:

- Imagine a bucket that holds max N tokens (e.g., 100 requests/min = bucket of 100).
- Tokens refill at a constant rate (e.g., 100 tokens per minute = ~1.67/sec).
- Each request consumes 1 token. If bucket is empty → reject or queue.
- Bucket allows **bursts** up to capacity.

```python
# Redis-based token bucket
tokens = redis.get(f"bucket:{user_id}")
if tokens > 0:
    redis.decr(f"bucket:{user_id}")
    process_request()
else:
    return 429, "Rate limit exceeded"
# Refill via background job or EXPIRE-based logic
```

| Algorithm | Burst Allowed | Use Case |
|---|---|---|
| Token Bucket | Yes (up to capacity) | LLM APIs, user-friendly |
| Leaky Bucket | No (smooth output) | Network traffic shaping |

---

### Q: How do you prioritize enterprise customers?

**TL;DR:** Separate queues per tier + preemption. Enterprise gets dedicated headroom.

- **Dedicated queue lanes:** Enterprise, Pro, and Free each have their own request queue. Scheduler processes Enterprise first.
- **Reserved capacity:** Allocate X% of GPU capacity as enterprise-reserved. E.g., 40% enterprise / 40% Pro / 20% Free.
- **Preemption:** If enterprise queue is backed up and Free tier is using reserved capacity → throttle Free requests.
- **SLA guarantees:** Enterprise SLA = P99 TTFT < 500ms. Monitor per-tier SLA compliance. Alert if violated.
- **Dedicated instances:** For top-tier enterprise, spin up dedicated model replicas (hard isolation).

> **Interview tip:** Mention that prioritization must be transparent — log which tier a request was in and whether it was delayed.

---

### Q: How do you enforce per-model limits?

**TL;DR:** Model-specific rate limiters. GPT-4 class models get tighter limits than smaller models.

Different models have vastly different resource costs. A 70B model call costs ~10x vs a 7B model call.

- **Model-scoped quotas:** Separate Redis counters per model per user: `user:{id}:model:gpt4:tokens` vs `user:{id}:model:gpt35:tokens`.
- **Compute unit pricing:** Normalize costs to "compute units" — 1 GPT-4 token = 10 compute units, 1 GPT-3.5 token = 1 compute unit. Set quota in compute units.
- **Model routing + fallback:** If user is near their GPT-4 limit, optionally route to a smaller model (with user consent or auto-fallback flag).

> Per-model limits also protect against model-specific VRAM oversubscription on specific GPU fleets.

---

### Q: How do you degrade gracefully?

**TL;DR:** Return useful fallback, not hard failure. Reduce quality, not availability.

Graceful degradation means the system stays useful under stress, just at reduced quality.

- **Fallback to smaller model:** If 70B model is overloaded, route to 13B or 7B model with a latency/quality tradeoff disclosure.
- **Disable retrieval:** If vector DB is down or overloaded, serve LLM-only response with caveat: *"Answer may not include latest information."*
- **Reduce max tokens:** Under load, set `max_new_tokens=512` instead of 2048. Faster, cheaper, but shorter answers.
- **Return cached stale response:** Serve a slightly stale cached answer rather than timing out.
- **Inform the user:** *"We're experiencing high load. Your response may be shorter than usual."* Transparency beats silent degradation.

> **Circuit breaker pattern:** If error rate for a service > X%, open the circuit and go to fallback automatically.

---

## 20. Load Balancing

### Q: How do you route requests across GPUs?

**TL;DR:** Least-loaded routing based on current queue depth and VRAM availability.

Simple round-robin ignores GPU state. Smarter strategies:

- **Least connections (queue depth):** Route to the GPU/replica with the fewest pending requests. Standard in NGINX, Envoy.
- **Least VRAM pressure:** Custom load balancer that queries each GPU's current VRAM usage via metrics endpoint. Route to GPU with most free VRAM.
- **Weighted routing:** Different GPU types (A100 vs V100) get different weights. A100 handles more requests proportionally.
- **Consistent hashing:** Hash request ID → same replica. Useful when prefix cache is warm on a specific GPU.

**Tools:** Custom sidecar in Envoy/Istio, NGINX Plus, or application-level routing in the inference gateway (e.g., LiteLLM proxy, custom FastAPI router).

---

### Q: How do you account for current KV cache load?

**TL;DR:** Expose VRAM and KV cache utilization as a health metric. Route away from hot GPUs.

KV cache is the binding constraint for LLM concurrency. Load balancer must be KV-cache-aware.

- **Each GPU exposes** a `/metrics` endpoint with `kv_cache_used_blocks`, `kv_cache_free_blocks`, `num_running_requests`.
- **Load balancer queries metrics:** Before routing, query each replica's KV cache utilization. Route to replica with most free KV cache blocks.
- **vLLM exposes this natively:** `vllm_cache_config_info`, `vllm_gpu_cache_usage_perc` metrics in Prometheus format.

> **Why it matters:** A GPU might have low CPU/network load but be at 95% KV cache capacity. Standard CPU-based load balancers miss this entirely.
>
> **Interview answer:** "I'd build a custom Envoy filter or sidecar that reads vLLM's Prometheus metrics and uses KV cache free blocks as the routing weight."

---

### Q: How do you implement sticky sessions?

**TL;DR:** Route same user to same GPU replica to keep their KV cache warm.

Sticky sessions ensure a user's requests consistently go to the same replica, so the KV prefix cache is already warm.

- **Session affinity:** Hash user_id or session_id → replica index: `replica = hash(session_id) % num_replicas`
- **Cookie-based:** Load balancer sets a cookie with the assigned replica ID on first request.
- **Consistent hashing ring:** Handles replica addition/removal gracefully — only 1/N sessions reroute when a replica is added.

**Trade-off:** Sticky sessions reduce load balancing effectiveness — one hot user can overload one replica.

> **Hybrid approach:** Use sticky sessions with overflow — if assigned replica is at >80% KV capacity, break stickiness and route to a less loaded one (accept cold cache).

---

### Q: How do you route by model size?

**TL;DR:** Different GPU fleets for different model sizes. Route based on requested model.

A 7B and a 70B model don't share the same hardware efficiently.

- **Fleet segmentation:** 7B models on A10G fleet, 70B models on multi-A100 fleet. Each fleet is a separate target group.
- **Model registry:** Central config maps `model_id → fleet_id`. Load balancer reads model from request header/body, looks up fleet, routes there.
- **Autoscaling per fleet:** Each fleet has its own HPA based on queue depth. 70B fleet scales independently of 7B fleet.

```python
if model == "gpt4":     route → A100_fleet
elif model == "gpt35":  route → A10G_fleet
else:                   route → T4_fleet  # small/default
```

---

### Q: How do you detect unhealthy replicas?

**TL;DR:** Active health checks + passive monitoring. Remove replicas that miss SLA.

Unhealthy replicas come from OOM crashes, runaway inference, or network issues.

- **Active health checks:** Load balancer pings `/health` every N seconds. Replica returns 200 if ready, 503 if not.
- **Passive monitoring:** Track error rate per replica. If >5% of requests return 5xx → mark unhealthy, remove from pool.
- **VRAM watchdog:** Alert + remove replica if VRAM utilization >95% for >60s (likely stuck inference).
- **Liveness vs readiness (Kubernetes):**
  - Liveness probe: Is process alive? OOM-crashed pod fails liveness → restarted automatically.
  - Readiness probe: Is it ready to serve? Traffic only returns after GPU memory is freed and model is loaded.
- **Circuit breaker:** Don't immediately remove; trip circuit breaker first. Allow recovery before hard removal.

---

## 21. Observability and Monitoring

### Metrics Reference

| Metric | What It Measures | Target / Alert Threshold |
|---|---|---|
| **GPU Utilization** | % of SM time active | Target: 70–90%. <50% = underloaded. 100% = bottleneck. |
| **VRAM Usage** | GB used / total GPU memory | Alert at 85%. OOM risk above 95%. KV cache dominates. |
| **SM Occupancy** | Active warps / max warps | Low = memory-bound. Use Nsight to diagnose. |
| **Queue Depth** | Pending requests per replica | Spike = overload. Sustained high = need scale-out. |
| **TTFT** | Time to first token | P95 target < 500ms |
| **Tokens/sec** | Generation throughput | Higher = more efficient batching. Model-dependent. |
| **Cache Hit Rate** | % requests served from cache | Target >30% for FAQ systems. Directly maps to cost. |
| **Vector DB QPS** | Queries/sec to vector store | Saturation → scale DB replicas |
| **Recall@k** | % relevant docs in top-k results | Offline eval metric. Monitor over time for drift. |
| **Error Rate** | % of 5xx / timeouts | Alert if >1%. SLA typically 99.9% uptime. |

---

### Q: What should be on an LLM serving dashboard?

**TL;DR:** Four sections: GPU health, serving quality (SLA), cost efficiency, retrieval quality.

A well-designed LLM dashboard has four swim lanes:

**1. Infrastructure Health**
- GPU utilization, VRAM usage, SM occupancy
- Node CPU/memory, queue depth per replica
- Alert when any metric goes out of bounds

**2. Serving Quality (SLA)**
- TTFT P50/P95/P99
- Tokens/sec, E2E latency per model
- Error rate by error type (OOM, timeout, rate limit)

**3. Cost Efficiency**
- Cache hit rate (semantic + exact)
- Cost per request (token_count × price)
- Tokens wasted (truncated or max_tokens hit)
- Requests served from cache vs model

**4. RAG / Retrieval Quality**
- Vector DB QPS and latency
- Retrieval cache hit rate
- Reranker latency
- Recall@k from offline eval runs
- Answer quality score (human or automated eval)

**Tools:** Prometheus + Grafana for metrics, Jaeger/Honeycomb for traces, LangSmith/Langfuse for LLM-specific tracing.

> **Interview format answer:** "I'd organize the dashboard around four questions: Is the GPU healthy? Are users getting fast responses? Are we spending efficiently? Is retrieval accurate?"

---

### Q: How do you correlate latency spikes with GPU memory?

**TL;DR:** Overlay VRAM usage time-series with P95 latency. Spikes co-occur with high VRAM pressure.

Latency spikes in LLM serving are often caused by VRAM pressure forcing KV cache eviction or batch size reduction.

- **Time-series overlay:** In Grafana, plot `p95_latency` and `vram_usage_%` on the same chart. Look for temporal correlation.
- **Event annotation:** Annotate the chart with events — new model deployments, traffic spikes, batch size changes.
- **vLLM KV cache eviction metric:** `vllm_cache_evictions_total` — spikes in evictions → KV cache pressure → latency spikes.
- **Request-level trace:** For a specific slow request, look at its trace: how much time was in prefill (input processing) vs decode (output generation)? High prefill time → long input or KV cache pressure.

**Action:** If correlation is confirmed:
1. Scale VRAM (bigger GPU or more replicas)
2. Reduce max_sequence_length
3. Implement KV cache-aware load balancing to spread pressure

---

### Q: How do you trace RAG pipelines?

**TL;DR:** Instrument each stage (embed → retrieve → rerank → generate) with span IDs using OpenTelemetry.

RAG has 4+ stages. A slow response could be failing at any of them. Distributed tracing makes this visible.

**Key spans to instrument with OpenTelemetry:**

| Span | Key Attributes to Record |
|---|---|
| `embed_query` | duration, embedding model used, input length |
| `vector_search` | duration, num results, top score, index name |
| `rerank` | duration, num candidates, top score after rerank |
| `llm_generate` | TTFT, total tokens, model name, finish reason |

**Sample trace code:**
```python
from opentelemetry import trace

tracer = trace.get_tracer("rag_pipeline")

with tracer.start_as_current_span("embed_query") as span:
    embedding = embed(query)
    span.set_attribute("model", "text-embedding-ada-002")
    span.set_attribute("input_length", len(query))

with tracer.start_as_current_span("vector_search") as span:
    results = vector_db.search(embedding, top_k=20)
    span.set_attribute("num_results", len(results))
    span.set_attribute("top_score", results[0].score)
```

**Trace backends:**
- General: Jaeger, Honeycomb, Datadog APM
- LLM-specific: LangSmith, Langfuse, Arize Phoenix

> **Goal:** For any user complaint ("slow response", "wrong answer"), pull the trace and see exactly which stage caused the issue. Root cause in minutes, not hours.

---

## Quick Reference Cheat Sheet

### Key Numbers to Remember

| Thing | Number |
|---|---|
| Redis read latency | ~0.1ms |
| LLM inference latency (no streaming) | 500ms–5s |
| TTFT target | P95 < 300ms |
| TPOT target | < 50ms/token |
| VRAM alert threshold | 85% |
| Semantic cache similarity threshold | ~0.95 cosine |
| LLaMA-2 7B KV cache @ 4K context | ~2 GB/session |
| Reranker vs embedding lookup speed | 10–50x slower |

### Tools Cheat Sheet

| Need | Tool |
|---|---|
| LLM serving / batching | vLLM, TensorRT-LLM, SGLang |
| Semantic caching | GPTCache, LangChain SemanticCache |
| Vector DB | Pinecone, Weaviate, Qdrant, pgvector |
| Metrics | Prometheus + Grafana |
| Distributed tracing | OpenTelemetry + Jaeger |
| LLM-specific tracing | LangSmith, Langfuse, Arize Phoenix |
| Rate limiting | Redis token bucket, Celery |
| Token counting | tiktoken |
| Load balancing | NGINX, Envoy, LiteLLM proxy |
| Streaming | SSE (FastAPI StreamingResponse) |

---
