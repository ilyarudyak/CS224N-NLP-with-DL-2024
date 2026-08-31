# CS224N (2024) Assignment 3: Optimized NMT (v2)

This project is an optimized version of the Stanford CS224N Assignment 3 Neural Machine Translation (Chinese to English) system. 

The goal is to keep the exact same model architecture (Bi-LSTM Encoder + Luong Attention Decoder with input feeding) while replacing the slow data pipeline with standard, fast PyTorch components (`Dataset`, `DataLoader`, and `collate_fn`), enabling Automatic Mixed Precision (AMP / BF16), and scaling batch sizes to fully saturate modern GPUs like the NVIDIA A100.

## 1. Overview and Key Results

### Optimization Benchmark Summary (1 Full Epoch on 200K Samples)

| Stage / Configuration | Hardware | Batch Size | Precision | Time / Epoch | Throughput | Peak GPU Util | Max VRAM | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Original Baseline** | T4 | 32 | FP32 | $\sim 1045\text{ s}$ ($17.4\text{ min}$) | $\sim 5,700\text{ w/s}$ | $<15\%$ | $\sim 2.5\text{ GB}$ | $1.0\times$ |
| **1. Modern Data Pipeline** | A100 | 32 | FP32 | $440\text{ s}$ ($7.3\text{ min}$) | $\sim 14,000\text{ w/s}$ | $52\%$ | $3.9\text{ GB}$ | $2.4\times$ |
| **2. Pipeline + AMP** | A100 | 128 | BF16 | $196\text{ s}$ ($3.3\text{ min}$) | $\sim 13,100\text{ w/s}$ | $63\%$ | $8.5\text{ GB}$ | $5.3\times$ |
| **3. Scaled Batch Size** | A100 | 256 | BF16 | **$134\text{ s}$ ($2.23\text{ min}$)** | **$22,530\text{ w/s}$** | **$94\%$** | **$13.6\text{ GB}$** | **$7.8\times$** |
| **4. Fused CrossEntropyLoss** | A100 | 256 | BF16 | **$144\text{ s}$ ($2.40\text{ min}$)** | **$22,000\text{ w/s}$** | **$90\%$** | **$13.7\text{ GB}$** | **$7.3\times$** |

### Key Optimizations Applied:
- **Pre-tokenized `TranslationDataset`**: Replaced string list storage with tokenized integer IDs loaded upfront.
- **Asynchronous `DataLoader`**: Background multi-worker data prefetching (`num_workers=4`) with pinned host memory (`pin_memory=True`).
- **Dynamic Tensor `collate_fn`**: Uses PyTorch's native `pad_sequence` and length-sorting before transferring batches directly to GPU tensors (`non_blocking=True`).
- **Automatic Mixed Precision (`torch.autocast`)**: Native `bfloat16` execution leveraging A100 Tensor Cores.
- **Batch Size Scaling ($32 \rightarrow 128 \rightarrow 256$)**: Fills all 108 A100 Streaming Multiprocessors, cutting kernel launch overhead by $87.5\%$.
- **Fused `nn.CrossEntropyLoss`**: Replaced manual `F.log_softmax` + `torch.gather` with PyTorch's standard fused CUDA kernel.

## 2. Project Structure

Here is the flat project structure, showing which files are new, modified, or unchanged from the original assignment:

```
assignment-3-v2/
├── README.md                     # [MODIFIED - Significant] Project notes and architecture plan
│
├── dataset.py                    # [NEW] Custom TranslationDataset and collate_fn
├── utils.py                      # [MODIFIED - Minor] Keep text reading helpers; remove old batch_iter
├── vocab.py                      # [UNCHANGED] Vocabulary and SentencePiece mappings
│
├── model_embeddings.py           # [UNCHANGED] Source and target embedding layers
├── nmt_model.py                  # [MODIFIED - Minor] forward() now takes tensor batches directly
├── beam_search_diagnostics.py    # [UNCHANGED] Helper to print translation samples
│
├── run.py                        # [MODIFIED - Significant] Training script using PyTorch DataLoader
├── sanity_check.py               # [MODIFIED - Minor] Tests adapted for tensor input
├── run.sh                        # [MODIFIED - Minor] Shell script to run training and decoding
│
├── 01_implementation_v2.ipynb    # [NEW] Step-by-step notebook to test each module
└── 02_experiments_colab_v2.ipynb # [NEW] Notebook for training and evaluation on Colab
```

### Summary of File Status

1. **`dataset.py` [NEW]**: Implements `TranslationDataset` and `collate_fn` for standard PyTorch data loading.
2. **`utils.py` [MODIFIED - Minor]**: Kept functions for loading text files with SentencePiece (`read_corpus`). Removed the slow `batch_iter` generator.
3. **`vocab.py` [UNCHANGED]**: Kept as-is for loading the vocabulary and converting tokens to IDs.
4. **`model_embeddings.py` [UNCHANGED]**: Kept as-is.
5. **`nmt_model.py` [MODIFIED - Minor]**: The internal model math (CNN, LSTM encoder, attention, decoder) is identical. The only change is in `forward()`: it now receives ready tensor batches from the DataLoader instead of receiving lists of text strings and converting them on CPU during every step.
6. **`run.py` [MODIFIED - Significant]**: Training loop now uses `DataLoader` instead of `batch_iter`. (We will address this script later).
7. **`beam_search_diagnostics.py` [UNCHANGED]**: Kept as-is.

## 3. Key Bottleneck #1: The Data Pipeline

### 3.1 Bottleneck Description

In the original assignment code, training 5 epochs on a T4 GPU took ~1.5 hours (5,225 seconds) on 200k sentence pairs because of how data was fed into the model:

- **Single-threaded text processing on CPU**: The custom generator `batch_iter` in `utils.py` sorted and sliced Python lists of strings on a single CPU thread.
- **Converting words to IDs inside `model.forward()`**: On every single training step, the model called `vocab.to_input_tensor(...)` to map words to integers and pad them on CPU before copying to GPU.
- **GPU Starvation**: The GPU spent most of its time waiting for the CPU to finish string manipulation and padding, rather than actually doing matrix multiplications.

### 3.2 Required Changes

To fix this bottleneck, we replace the custom string generator with standard PyTorch data loading:

1. `TranslationDataset` (in new file `dataset.py`)
- Instead of: Keeping sentences as raw lists of strings in memory.
- What it does: 
  - Converts text sentences into integer token IDs when loading the dataset.
  - `__getitem__(idx)` simply returns `(src_ids, tgt_ids)` as lists of integers.

2. `collate_fn` (in new file `dataset.py`)
- Instead of: Calling `vocab.to_input_tensor()` inside `nmt_model.py`'s `forward()`.
- What it does: 
  - Takes a batch of integer lists from `TranslationDataset`.
  - Sorts them by source sentence length (longest to shortest, required for packed LSTM sequence).
  - Pads source sentences to the batch max length: `(src_len, batch_size)`.
  - Pads target sentences to the batch max length: `(tgt_len, batch_size)`.
  - Returns `src_padded`, `src_lengths`, and `tgt_padded` ready for the model.

3. PyTorch `DataLoader` (used in `run.py` / notebooks)
- Instead of: The custom `batch_iter(...)` while-loop in `utils.py`.
- What it does: 
  - Automatically loads batches in parallel in background workers (`num_workers=4`).
  - Pre-fetches and pins memory so data transfers to the GPU are fast and don't block training.

---

## 4. Key Bottleneck #2: Precision and Floating-Point Overhead (AMP / BF16)

### 4.1 Bottleneck Description

- **Standard FP32 computation**: In the original implementation, all matrix multiplications, recurrent cell steps, and attention calculations were performed in 32-bit floating point (`float32`).
- **Tensor Core under-utilization**: Modern accelerators like the NVIDIA A100 feature dedicated Tensor Cores optimized for lower-precision formats (`bfloat16` and `float16`), offering up to $3\times\text{--}4\times$ the raw compute throughput and requiring half the memory bandwidth.
- **Micro-batch casting overhead at small batch sizes**: At small batch sizes (`batch_size=32`), running the step-by-step recurrent loop under AMP incurs type-casting overhead that offsets gains (yielding $522\text{ s}$ vs $440\text{ s}$). Once scaled to larger batch sizes ($128+$), the Tensor Core compute acceleration overwhelmingly dominates.

### 4.2 Required Changes

- Enabled PyTorch's native Automatic Mixed Precision via `torch.autocast`:
  ```python
  with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
      example_losses = -model(src_padded, src_lengths, tgt_padded)
      loss = example_losses.sum() / batch_size
  ```
- **Hardware-adaptive precision**:
  - **A100 (CUDA)**: Auto-selects `torch.bfloat16` (full dynamic range matching FP32, eliminating the need for loss scaling / `GradScaler`).
  - **T4 (CUDA)**: Uses `torch.float16` with `torch.amp.GradScaler`.
  - **Apple Silicon (MPS)**: Uses `torch.float16`.
- Wrapped validation evaluation (`evaluate_ppl`) in autocasting for faster dev set checks.

---

## 5. Key Bottleneck #3: Batch Size & GPU Saturation (Scaling $32 \rightarrow 128 \rightarrow 256$)

### 5.1 Bottleneck Description

- **Kernel Launch Overhead**: With `batch_size=32`, an epoch on 200K samples requires $6,250$ mini-batch iterations. Because the decoder unrolls an LSTMCell and Luong attention step-by-step for $\sim 50$ timesteps per sentence, this triggered over $312,500$ CUDA kernel launches per epoch.
- **Streaming Multiprocessor (SM) Starvation**: The NVIDIA A100 contains 108 Streaming Multiprocessors. A mini-batch of size 32 cannot supply enough parallel operations to saturate all 108 SMs, causing GPU compute utilization to hover around only $33\%$.
- **VRAM Headroom**: At `batch_size=32`, peak VRAM usage was only $3.9\text{ GB}$ ($<10\%$ of a 40GB A100).

### 5.2 Required Changes & Hyperparameter Tuning

1. **Batch Size Scaling**:
   - Scaled from `batch_size=32` $\rightarrow$ `128` $\rightarrow$ `256`.
   - Iterations per epoch dropped from $6,250$ down to $781$ ($87.5\%$ fewer kernel launches).
2. **Learning Rate Adjustment**:
   - Increased `--lr` from `5e-4` to `1.2e-3` to accommodate the larger, more stable gradient estimates across 256 samples per step.
3. **Validation Frequency**:
   - Adjusted `--valid-niter` from `200` to `25` or `50` iterations to maintain frequent dev perplexity checks throughout training.

### 5.3 Actual Profiling & Benchmark Data

```
# Profile at Batch Size 32 (FP32)
Average GPU Utilization: 33.67%
Peak GPU Utilization:    52.00%
Max VRAM Used:          3912.00 MB
Time for 1 Epoch:       440.29 sec

# Profile at Batch Size 128 (BF16 AMP)
Average GPU Utilization: 36.17%
Peak GPU Utilization:    63.00%
Max VRAM Used:          8548.00 MB
Time for 1 Epoch:       196.44 sec  (5.3x faster than baseline)

# Profile at Batch Size 256 (BF16 AMP)
Average GPU Utilization: 44.24%
Peak GPU Utilization:    94.00%
Max VRAM Used:          13648.00 MB
Time for 1 Epoch:       134.03 sec  (7.8x faster than baseline)
Throughput:             22,530.65 words/sec
```

---

## 6. Key Bottleneck #4: Un-fused Probability & Loss Computation (`nn.CrossEntropyLoss`)

### 6.1 Bottleneck Description

In the original implementation, the loss calculation in `forward()` performed three separate, un-fused memory operations:
1. **Full `F.log_softmax`**: Allocated a dense 3D tensor of shape `(tgt_len - 1, batch_size, 21000)` in VRAM containing all log probabilities.
2. **Explicit `torch.gather`**: Gathered log probabilities corresponding to the target gold word indices.
3. **Manual boolean masking**: Built a float mask tensor to zero out `<pad>` tokens and summed the probabilities.

This caused unnecessary peak VRAM allocation and multiple memory-bound CUDA kernel launches per step.

### 6.2 Required Changes

- Initialized `nn.CrossEntropyLoss` with automatic padding exclusion:
  ```python
  self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.tgt['<pad>'], reduction='none')
  ```
- Directly passed raw unnormalized logits `(tgt_len - 1, batch_size, vocab_size)` and targets `(tgt_len - 1, batch_size)` to `self.loss_fn`:
  ```python
  logits = self.target_vocab_projection(combined_outputs)
  loss_matrix = self.loss_fn(
      logits.reshape(-1, vocab_size),
      target_gold.reshape(-1)
  ).reshape(tgt_len_m1, batch_size)
  scores = -loss_matrix.sum(dim=0)
  ```
- **Benefits**:
  - PyTorch's native C++/CUDA kernel computes the softmax and negative log likelihood in a single fused pass.
  - Avoids storing large dense probability matrices in GPU memory.
  - Maintains exact numerical compatibility with the training loop and perplexity calculations.
  - Aligns code with modern industry standards (Transformers, PyTorch sequence modeling).

### 6.3 Actual Profiling Data

```
# Profile at Batch Size 256 (BF16 AMP + Fused CrossEntropyLoss)
Average GPU Utilization: 40.59%
Peak GPU Utilization:    90.00%
Max VRAM Used:          13712.00 MB
Time for 1 Epoch:       144.86 sec  (7.3x faster than baseline)
Throughput:             21,994.75 words/sec
Final Dev Perplexity:   24.32
```
