import torch
from torch import nn

import itertools

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
# Vocabulary and Dataset
#############################################################

# Create a vocabulary from the corpus
class WWVocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, corpus):

        # Preprocess the corpus
        self.train_tokens = self._tokenize_corpus()

        # Build the vocabulary from the tokenized sentences
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self._build_vocab()

    def _tokenize_sentence(self, sentence):
        return sentence.lower().split()

    def _tokenize_corpus(self):
        return [self._tokenize_sentence(sentence) for sentence in corpus]

    def _build_vocab(self):

        # Define special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]

        # Extract unique words from the training tokens
        unique_words = set(itertools.chain.from_iterable(self.train_tokens))
        unique_words = sorted(unique_words)

        # Add special tokens to the vocabulary 
        self.vocab = special_tokens + unique_words

        # Build mapping dictionaries
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

        # Store specific indices for quick access
        self.pad_idx = self.word2idx[self.PAD_TOKEN]  # 0
        self.unk_idx = self.word2idx[self.UNK_TOKEN]  # 1
        

    def __len__(self):
        return len(self.word2idx)

    def word_to_index(self, word):
        return self.word2idx.get(word.lower(), self.word2idx.get(self.UNK_TOKEN))

    def index_to_word(self, index):
        return self.idx2word.get(index, self.UNK_TOKEN)

    def _encode_train_tokens(self):
        return [[self.word_to_index(word) for word in sentence] for sentence in self.train_tokens]


class WWDataset(torch.utils.data.Dataset):
    def __init__(self, vocab):
        super(WWDataset, self).__init__()
        self.vocab = vocab
        
        self.labels = self._assign_labels()
        self.data = [(encoded_sentence, labels) 
                     for encoded_sentence, labels in zip(vocab._encode_train_tokens(), self.labels)]

    def _assign_labels(self, 
                       locations = set(["australia", "ankara", "paris", 
                                        "stanford", "taiwan", "turkey"])):
        train_labels = [[1 if word in locations else 0 for word in sent] 
                        for sent in self.vocab.train_tokens]
        return train_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]