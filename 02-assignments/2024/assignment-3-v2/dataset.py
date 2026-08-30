#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py: PyTorch Dataset, CollateFn, and DataLoader creation for Neural Machine Translation.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm


class TranslationDataset(Dataset):
    """
    Dataset for Neural Machine Translation that pre-tokenizes and converts
    source and target text sentences into integer token IDs during initialization.
    """

    def __init__(
        self,
        src_file_path: Union[str, Path],
        tgt_file_path: Union[str, Path],
        src_vocab,
        tgt_vocab,
        src_sp_model: Union[str, Path] = "src.model",
        tgt_sp_model: Union[str, Path] = "tgt.model",
        max_samples: Optional[int] = None,
    ):
        """
        @param src_file_path (str or Path): Path to the source language text file.
        @param tgt_file_path (str or Path): Path to the target language text file.
        @param src_vocab (VocabEntry): Vocabulary object for the source language.
        @param tgt_vocab (VocabEntry): Vocabulary object for the target language.
        @param src_sp_model (str or Path): Path to source SentencePiece model file.
        @param tgt_sp_model (str or Path): Path to target SentencePiece model file.
        @param max_samples (int, optional): Maximum number of samples to load (for debugging/testing).
        """
        super().__init__()
        self.src_file_path = str(src_file_path)
        self.tgt_file_path = str(tgt_file_path)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Initialize SentencePiece processors
        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load(str(src_sp_model))

        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load(str(tgt_sp_model))

        # Pre-tokenize all data into lists of integer IDs
        self.src_data: List[List[int]] = []
        self.tgt_data: List[List[int]] = []
        self._load_and_encode_data(max_samples)

    def _load_and_encode_data(self, max_samples: Optional[int] = None) -> None:
        """
        Reads source and target text files line by line, tokenizes with SentencePiece,
        and converts subwords to vocabulary integer IDs.
        """
        with open(self.src_file_path, "r", encoding="utf-8") as f_src, \
             open(self.tgt_file_path, "r", encoding="utf-8") as f_tgt:

            for line_idx, (line_src, line_tgt) in enumerate(zip(f_src, f_tgt)):
                if max_samples is not None and line_idx >= max_samples:
                    break

                # Encode source line into subword pieces, then map to integer token IDs
                src_pieces = self.sp_src.encode_as_pieces(line_src.strip())
                src_ids = [self.src_vocab[p] for p in src_pieces]

                # Encode target line: add <s> (start) and </s> (end) special tokens
                tgt_pieces = ["<s>"] + self.sp_tgt.encode_as_pieces(line_tgt.strip()) + ["</s>"]
                tgt_ids = [self.tgt_vocab[p] for p in tgt_pieces]

                self.src_data.append(src_ids)
                self.tgt_data.append(tgt_ids)

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample as integer tensors.
        @param idx (int): Sample index.
        @returns (src_ids, tgt_ids): Tuple of 1D LongTensors.
        """
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )


class TranslationCollate:
    """
    Collate callable for batching TranslationDataset samples:
    1. Sorts the batch by source sequence length in descending order (required for pack_padded_sequence).
    2. Measures source lengths.
    3. Pads source and target sequences to batch maximum lengths using pad_sequence.
    4. Returns tensors shaped (src_len, batch_size) and (tgt_len, batch_size).
    """

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        @param batch: List of (src_tensor, tgt_tensor) tuples.
        @returns:
            src_padded (Tensor): Shape (src_len, batch_size), padded with pad_id.
            src_lengths (List[int]): List of unpadded source sequence lengths.
            tgt_padded (Tensor): Shape (tgt_len, batch_size), padded with pad_id.
        """
        # Sort batch samples by source length descending
        batch.sort(key=lambda item: len(item[0]), reverse=True)

        src_tensors = [item[0] for item in batch]
        tgt_tensors = [item[1] for item in batch]

        src_lengths = [len(s) for s in src_tensors]

        # Pad sequences with batch_first=False -> shape (seq_len, batch_size)
        src_padded = pad_sequence(src_tensors, batch_first=False, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_tensors, batch_first=False, padding_value=self.pad_id)

        return src_padded, src_lengths, tgt_padded


def get_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    pad_id: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Helper function to construct a PyTorch DataLoader with TranslationCollate.
    """
    collate_fn = TranslationCollate(pad_id=pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
