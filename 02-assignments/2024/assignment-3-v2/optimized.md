# CS224N (2024) Assignment 3: Optimized NMT (v2)

This project is an optimized version of the Stanford CS224N Assignment 3 Neural Machine Translation (Chinese to English) system. 

The goal is to keep the exact same model architecture (Bi-LSTM Encoder + Luong Attention Decoder with input feeding) while replacing the slow data pipeline with standard, fast PyTorch components (`Dataset`, `DataLoader`, and `collate_fn`).

## 1. Overview and Key Bottlenecks


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

### 3.3. Next Steps

- Implement `dataset.py` with `TranslationDataset` and `collate_fn`, and test them in a small notebook cell to make sure batch shapes match what `NMT.forward()` expects.
