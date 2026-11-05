import torch
from torch import nn

import itertools

from dataclasses import dataclass

#############################################################
# Data
#############################################################

# Our raw data, which consists of sentences
corpus = [
          "We always come to Paris",
          "The professor is from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara"
         ]

#############################################################
# Config
#############################################################
@dataclass
class WWConfig:
    window_size: int = 2
    embed_dim: int = 25
    hidden_dim: int = 25
    learning_rate: float = 0.01
    batch_size: int = 2
    num_epochs: int = 1000
    freeze_embeddings: bool = False

#############################################################
# Data management
#############################################################
class WWVocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenized_texts):
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        unique_words = sorted(
            set(itertools.chain.from_iterable(tokenized_texts)) - set(special_tokens)
        )
        self.vocab = special_tokens + unique_words

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.vocab)

    def word_to_index(self, word):
        return self.word2idx.get(word.lower(), self.unk_idx)

    def index_to_word(self, idx):
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode(self, tokens):
        """Converts a list of word tokens -> list of integer indices."""
        return [self.word_to_index(token) for token in tokens]

    def decode(self, indices):
        """Converts a list of integer indices -> list of word tokens."""
        return [self.index_to_word(idx) for idx in indices]

    def encode_batch(self, batch_of_tokens):
        """Converts a list of token lists -> list of index lists."""
        return [self.encode(tokens) for tokens in batch_of_tokens]


class WWDataset(torch.utils.data.Dataset):
    DEFAULT_LOCATIONS = {"australia", "ankara", "paris", "stanford", "taiwan", "turkey"}

    def __init__(self, sentences, labels=None, locations=None):
        """
        sentences: list of raw string sentences, e.g. ["We always come to Paris", ...]
        labels: optional explicit labels. If None, generated from `locations`.
        """
        # Tokenize sentences
        self.data = [s.lower().split() for s in sentences]

        # Assign labels if not provided
        if labels is not None:
            self.labels = labels
        else:
            locs = locations if locations is not None else self.DEFAULT_LOCATIONS
            self.labels = [[1 if w in locs else 0 for w in sent] for sent in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class WWCollator:
    def __init__(self, vocab: WWVocabulary, config: WWConfig):
        self.vocab = vocab
        self.window_size = config.window_size
        self.pad_token = vocab.PAD_TOKEN
        self.pad_idx = vocab.pad_idx

    def _pad_window(self, sentence):
        window = [self.pad_token] * self.window_size
        return window + sentence + window

    def __call__(self, batch):
        # Extract sentences and labels from the batch
        sentences, labels = zip(*batch)

        # 1. Record true sentence lengths BEFORE padding (needed for loss normalization)
        lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)

        # 2. Window-pad sentences with <pad>
        sentences = [self._pad_window(sent) for sent in sentences]

        # 3. Convert tokens to index tensors
        sentences_encoded = self.vocab.encode_batch(sentences)

        # 4. Pad batch sequences to equal lengths
        sentences_tensor = [torch.tensor(sent, dtype=torch.long) for sent in sentences_encoded]
        sentences_padded = torch.nn.utils.rnn.pad_sequence(
            sentences_tensor, 
            batch_first=True, 
            padding_value=self.pad_idx
        )

        labels_tensor = [torch.tensor(lbl, dtype=torch.float) for lbl in labels]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_tensor, 
            batch_first=True, 
            padding_value=0.0
        )

        return sentences_padded, labels_padded, lengths