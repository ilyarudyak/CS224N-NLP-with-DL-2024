import random
from typing import List, Collection, Iterable
from nltk.lm.api import LanguageModel
from nltk.lm import Vocabulary, NgramCounter
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline

class EmpiricalUnigramNltkModel(LanguageModel):
    """
    A professional Unigram model leveraging the official NLTK LanguageModel API.
    Inheriting from NLTK's base class gives us access to .perplexity(), .entropy(),
    and .generate() for free.
    """
    def __init__(self, order: int = 1, vocabulary: Vocabulary = None, counter: NgramCounter = None):
        super().__init__(order, vocabulary, counter)

    def fit(self, sentences: Collection[List[str]], vocabulary_text: Iterable[str] = None):
        """
        Fits the model using NLTK's optimized pipelines.
        """
        # 1. Prepare Vocabulary if not already provided
        if not self.vocab:
            # unk_cutoff=1 ensures any word seen only once (or not at all) can be <UNK>
            self.vocab = Vocabulary(vocabulary_text, unk_cutoff=1)
        
        # 2. Prepare NgramCounter
        if not self.counts:
            self.counts = NgramCounter()
            
        # 3. Use NLTK's padded_everygram_pipeline for maximum performance
        # This handles padding and n-gram generation in one efficient pass
        train_data, _ = padded_everygram_pipeline(self.order, sentences)
        self.counts.update(train_data)

    def score(self, word: str, context: tuple = None) -> float:
        """
        Returns the probability P(word | context).
        For Unigrams, context is ignored.
        """
        # score(word) in NLTK uses counts.unigrams.freq(word) internally
        # but handles the <UNK> mapping automatically via the Vocab object.
        return self.counts.unigrams.freq(word)

    def logscore(self, word: str, context: tuple = None) -> float:
        """
        Returns the log2 probability. NLTK API uses base 2 by default.
        """
        return super().logscore(word, context)

    def generate_sentence(self, num_words: int = 20) -> List[str]:
        """
        Generates a sentence using NLTK's built-in sampler.
        """
        # NLTK generate returns a list of tokens
        tokens = self.generate(num_words, text_seed=["<s>"])
        
        # Clean up output (remove padding symbols for display)
        result = []
        for t in tokens:
            if t == "</s>":
                break
            if t != "<s>":
                result.append(t)
        return result
