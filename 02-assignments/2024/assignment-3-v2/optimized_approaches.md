That **$3\times$ speedup** (from $\sim 1000\text{ s}$ down to $352\text{ s}$ at iteration 5,000, and $5,734 \rightarrow 13,978\text{ words/sec}$) is a huge result for just the data pipeline!

Here is the prioritized list of remaining optimizations, what they do, and their estimated speedup on A100:

---

### 1. Automatic Mixed Precision (BF16 Autocast)
* **What it is**: The A100 has dedicated Tensor Cores for BFloat16/FP16 matrix multiplications that are up to $3\text{--}4\times$ faster than standard FP32 and use half the memory bandwidth.
* **How to implement**: Wrap the model forward pass and loss in `run.py` with:
  ```python
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      example_losses = -model(src_padded, src_lengths, tgt_padded)
  ```
  *(BF16 requires no GradScaler on A100 because it shares the dynamic range of FP32).*
* **Estimated Additional Gain**: **$\sim 1.5\times - 2.0\times$** ($14\text{k} \rightarrow \sim 22\text{k}-28\text{k}\text{ words/sec}$).

---

### 2. Batch Size Scaling ($32 \rightarrow 64$ or $128$)
* **What it is**: An A100 has 108 Streaming Multiprocessors (SMs). With `batch_size=32` and `hidden_size=768`, the GPU is under-saturated (most CUDA cores sit idle waiting for matrix multiplication tasks). 
* **How to implement**: Change `--batch-size=128` (or `64`) in `run.sh`, while slightly increasing the learning rate (e.g. $5\times 10^{-4} \rightarrow 1\times 10^{-3}$) with a brief linear warm-up.
* **Estimated Additional Gain**: **$\sim 1.3\times - 1.8\times$**.

---

### 3. Fused Cross-Entropy Loss (`nn.CrossEntropyLoss(ignore_index=pad_id)`)
* **What it is**: In `nmt_model.py:90-98`, the model currently projects to vocabulary, runs `F.log_softmax` across the entire vocabulary ($21\text{k}$ entries), allocates a large 3D tensor in VRAM, and runs `torch.gather` + manual masks.
* **How to implement**: Have the model return raw unnormalized logits `(tgt_len, b, vocab_size)` and let PyTorch's native `nn.CrossEntropyLoss(ignore_index=pad_id)` compute the loss in a single fused CUDA kernel.
* **Estimated Additional Gain**: **$\sim 1.1\times - 1.2\times$** (plus $30\text{--}40\%$ lower peak VRAM).

---

### 4. Length-Based Batch Bucketing (`BucketBatchSampler`)
* **What it is**: Sentences in the dataset vary in length from 5 to 70 words. When a short 5-word sentence is paired with a 65-word sentence in the same batch, the model wastes $90\%$ of its computation calculating attention over `<pad>` tokens.
* **How to implement**: Group sentences with similar source lengths into contiguous buckets in `dataset.py` so each batch has minimal padding.
* **Estimated Additional Gain**: **$\sim 1.25\times - 1.4\times$**.

---

### 5. `torch.compile(model)` (PyTorch 2.x Kernel Fusion)
* **What it is**: The Luong attention decoder runs an explicit step-by-step loop for every target time step ($t=1\dots T$). `torch.compile` traces this loop and generates a fused Triton kernel, eliminating individual CUDA kernel launch overheads.
* **How to implement**: In `run.py`, after `model = model.to(device)`, add `model = torch.compile(model)`.
* **Estimated Additional Gain**: **$\sim 1.2\times - 1.5\times$**.

---

### Summary Overview

| Step | Optimization | Estimated Incremental Gain | Estimated Cumulative Speed | Time for 1 Full Epoch ($200\text{K}$) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Baseline (Original CS224N on T4) | $1.0\times$ | $\sim 5,700\text{ w/s}$ | $\sim 1,000\text{ s}$ ($\sim 17\text{ min}$) |
| **1** | **Dataset + DataLoader (Current)** | **$3.0\times$** | **$\sim 14,000\text{ w/s}$** | **$\sim 350\text{ s}$ ($\sim 5.8\text{ min}$)** |
| **2** | + Automatic Mixed Precision (BF16) | $\sim 1.8\times$ | $\sim 25,000\text{ w/s}$ | $\sim 190\text{ s}$ ($\sim 3.2\text{ min}$) |
| **3** | + Batch Size ($128$) & Bucketing | $\sim 1.5\times$ | $\sim 37,000\text{ w/s}$ | $\sim 130\text{ s}$ ($\sim 2.1\text{ min}$) |
| **4** | + Fused Loss & `torch.compile` | $\sim 1.3\times$ | $\sim 48,000\text{ w/s}$ | $\sim 100\text{ s}$ ($\sim 1.6\text{ min}$) |

When you are ready, **AMP (BF16)** is the cleanest and easiest next step to implement.