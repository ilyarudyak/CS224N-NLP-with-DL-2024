# Step 1.  Loading Data Files

-  `zh_en_data` : data files that contain plain text pairs in English (`train.en`) and Chinese (`train.zh`). The two files have matching lines. The Chinese file is the source language. The English file is the target language. Files contain 200K samples each.

```bash
(env) zh_en_data  (main) $ cat train.en | wc -l
  200000
(env) zh_en_data  (main) $ cat train.zh | wc -l
  200000
```

- The files are read by `read_corpus()` from `utils.py` in `train()` from `run.py`. It is defined in `utils.py`. It reads the plain text files and returns a list of tokenized sentences `List[List[str]]` using `spm.SentencePieceProcessor()`: `subword_tokens = sp.encode_as_pieces(line)`. 

```python
train_data_src = read_corpus(args['--train-src'], 
							source='src',
							vocab_size=21000) 
							
train_data_tgt = read_corpus(args['--train-tgt'], 
							source='tgt', 
							vocab_size=8000)
```
```
['<s>', '▁he', '▁was', '▁seen', '▁wear', 'ing', '▁a', '▁white', '▁', 'jacket', ',', '▁a', '▁dark', '-', 'coloured', '▁t', '-', 'shi', 'rt', '▁and', '▁p', 'ants', '.', '</s>']
```

- We then convert source and target list of sentences into list of tuples, where each tuple contains a pair of sentences.

```python
# Convert train and dev data into list of tuples (src_sent, tgt_sent)
train_data = list(zip(train_data_src, train_data_tgt)) # List[Tuple[List[str], List[str]]]
```
```
(['▁', '事发', '时', '身穿', '白色', ...], ['<s>', '▁he', '▁was', '▁seen', '▁wear', ...])
```

### New Approach

- We implement a standard PyTorch `TranslationDataset(Dataset)` in `dataset.py`.
- When loading the data files, we convert each sentence into a list of integer token IDs (`List[int]`) instead of storing string pieces:
  - Source sentences: list of token IDs (`List[int]`).
  - Target sentences: list of token IDs (`List[int]`) with `<s>` and `</s>` IDs added.
- `TranslationDataset.__getitem__(idx)` directly returns the integer pair `(src_ids, tgt_ids)`.

## Step 2. Loading Vocabulary

- We load vocabulary from a file `vocab.json`. We do *not* train the tokenizer, we use a pre-trained vocabulary. In `train` we use: `vocab = Vocab.load(args['--vocab'])`.

- We then supply `vocab` when creating our model:

```python
model = NMT(embed_size=1024,
			hidden_size=768,
			dropout_rate=float(args['--dropout']),
			vocab=vocab
			)
```

### New Approach

- **Unchanged**: We still load `vocab = Vocab.load(...)` from `vocab.json`.
- We pass `vocab` to `TranslationDataset` (to convert tokens to IDs during initialization) and `pad_id` to our collate function and model.

## Step 3. Loading Batches

- First of all, we do *not* use a standard PyTorch pipeline with Dataset and loaders. The manual batching is in `batch_iter` from `utils.py`. 

- We call in `run.py` using `train_data` as list of tokenized sentences from the Step 1. In other words, we still deal with list of tokenized sentences.

```python
for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
	...
```

- `batch_iter` is defined in `utils.py`. It yields batches of tokenized sentences without any padding.

### New Approach

- We replace `batch_iter` with a standard PyTorch `DataLoader(dataset, batch_size=..., shuffle=True, collate_fn=collate_fn, ...)`.
- DataLoader parameters can be configured via CLI flags (e.g. `--num-workers=<int>`, `--pin-memory`):
  - **Local / MPS / CPU**: `num_workers=0`, `pin_memory=False`
  - **Colab T4 (2 vCPU)**: `num_workers=2`, `pin_memory=True`
  - **Colab A100 (12 vCPU)**: `num_workers=4`, `pin_memory=True`, `persistent_workers=True`

## Step 4. Converting to tensors

- Our batches are created by `batch_iter` and contain a list of tokenized sentences. `forward` method of our model receives this list and converts it to tensors using `to_input_tensor` from `VocabEntry` class. This method converts the list of sentences into the list of IDs, pad them using `pad_sents` and converts it to a tensor.

### New Approach

- Handled automatically inside `collate_fn` in `dataset.py` before batches reach `model.forward()`:
  1. Sorts the batch of integer pairs by source length in descending order (needed for `pack_padded_sequence`).
  2. Pads sequences using PyTorch's native `pad_sequence(..., padding_value=pad_id)`.
  3. Returns `src_padded` `(src_len, b)`, `src_lengths` `List[int]`, and `tgt_padded` `(tgt_len, b)`.
- `model.forward(src_padded, src_lengths, tgt_padded)` receives ready tensor batches directly without any string-to-ID conversion during training.
