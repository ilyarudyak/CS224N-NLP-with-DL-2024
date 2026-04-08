from nltk.corpus import reuters
from collections import Counter


class ReutersDataset:

    def __init__(self, is_toy=True, limit=1000):
        self.sentences = reuters.sents()

        if is_toy:
            self.sentences = self.sentences[:limit]
            # Lowercase all words in the sentences
            self.sentences = [[word.lower() for word in sent] for sent in self.sentences]

        self.words = [word for sent in self.sentences for word in sent]
        self.vocab = self._build_vocab()
        

    def _build_vocab(self):
        vocab = Counter(self.words)
        vocab.update(["<s>", "</s>", "<UNK>"])
        return vocab