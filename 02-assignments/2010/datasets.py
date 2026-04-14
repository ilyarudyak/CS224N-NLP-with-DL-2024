import os
from typing import List, Set, Collection
from collections import Counter
    

class TextDataset:
    """
    Base class for loading and preprocessing text datasets for language modeling.

    Each dataset is expected to be a text file with one sentence per line.
    Its already lowercased and space-tokenized, so we just need to read and split.

    We create a dataset with the provided vocabulary or create one from the data if not provided.
    In language modeling, the vocabulary must be fixed at training time.
    So we create a vocabulary for training set and then pass it to the test sets to ensure consistent OOV handling.

    @sentences: List of sentences. Each sentence is a list of tokens. List[List[str]]
    @vocab: Set of unique tokens in the dataset. We add special tokens <s>, </s>, and <UNK>.
    """
    def __init__(self, file_path: str, vocab: Set[str] = None, limit: int = None):
        self.file_path = file_path
        self.limit = limit
        self.sentences = [] # List[List[str]]

        # If a vocabulary is provided, we use it; 
        # otherwise, we will build it from the data.
        self.vocab = vocab if vocab is not None else set()

        # List of all words in the dataset, used to build the vocabulary using `nltk.lm.Vocabulary`
        self.all_words = [] # List[str]
        
        if os.path.exists(file_path):
            self._load_data()
        else:
            print(f"Warning: File {file_path} not found.")

    def _load_data(self):
        """
        Reads the file, tokenizes sentences, and updates vocabulary if not provided.
        Each line is a sentence.
        """
        # We only build the vocabulary if it was not provided, 
        # to ensure consistent OOV handling.
        # In language modeling, the vocabulary must be fixed at training time.
        build_vocab = len(self.vocab) == 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    break
                
                # Basic preprocessing: lowercase and split
                # Note: The assignment data seems already lowercased and space-tokenized.
                tokens = line.strip().split() # List[str]
                if not tokens:
                    continue
                
                self.sentences.append(tokens)
                if build_vocab:
                    self.vocab.update(tokens)
                    self.all_words.extend(tokens)
        
        # Ensure special tokens are always in vocabulary
        if build_vocab:
            self.vocab.update(["<s>", "</s>", "<UNK>"])
            self.all_words.extend(["<s>", "</s>", "<s>", "</s>"]) 

    def get_sentences(self) -> List[List[str]]:
        return self.sentences


class EuroparlDataset(TextDataset):
    """
    Dataset for the formal European Parliament corpus.
    Used primarily for training and in-domain testing.
    """
    DATA_DIR = "/Users/iliarudiak/Library/Mobile Documents/com~apple~CloudDocs/_courses/2026/working/01-deep-learning/03-cs224N-NLP-with-DL/02-assignments/2010/data"

    def __init__(self, split: str = "train", vocab: Set[str] = None, limit: int = None):
        filename = f"europarl-{split}.sent.txt"
        # Navigate to the workspace root then to the data dir
        # In a real setup, we'd use a more robust path resolution
        path = os.path.join(self.DATA_DIR, filename)
        super().__init__(path, vocab=vocab, limit=limit)


class EnronDataset(TextDataset):
    """
    Dataset for the informal Enron Email corpus.
    Used primarily for out-of-domain testing.
    """
    DATA_DIR = "/Users/iliarudiak/Library/Mobile Documents/com~apple~CloudDocs/_courses/2026/working/01-deep-learning/03-cs224N-NLP-with-DL/02-assignments/2010/data"

    def __init__(self, vocab: Set[str], limit: int = None):
        # Enron testing MUST use the vocabulary from Europarl training
        path = os.path.join(self.DATA_DIR, "enron-test.sent.txt")
        super().__init__(path, vocab=vocab, limit=limit)
