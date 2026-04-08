import math
import random
from typing import List, Collection
from nltk.probability import FreqDist
from nltk.lm.preprocessing import pad_sequence

class NgramCounter:
    """
    Decoupled counter for n-grams. Currently supports unigrams ONLY.
    """
    def __init__(self, sentences: Collection[List[str]] = None):
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }
        self.unknown = "<UNK>"
        self.unigrams = FreqDist()
        self.total_unigrams = 0

        # If sentences are provided, train the counter immediately
        # It allows for a one-line instantiation: counter = NgramCounter(data).
        if sentences is not None:
            self.train(sentences)

    def train(self, sentences: Collection[List[str]]) -> None:
        for sentence in sentences:
            # Standard <s> and </s> padding
            padded = list(pad_sequence(sentence, n=1, **self.padding))
            self.unigrams.update(padded)
        self.total_unigrams = self.unigrams.N()

class LanguageModel:
    """
    Base interface for language models.
    """
    def train(self, sentences: Collection[List[str]]) -> None:
        raise NotImplementedError

    def get_word_log_probability(self, sentence: List[str], index: int) -> float:
        raise NotImplementedError

    def get_sentence_log_probability(self, sentence: List[str]) -> float:
        raise NotImplementedError

    def check_model(self) -> float:
        raise NotImplementedError

    def generate_sentence(self) -> List[str]:
        raise NotImplementedError


class EmpiricalUnigramLanguageModel(LanguageModel):
    """
    A professional UNIGRAM language model utilizing NLTK FreqDist,
    log probabilities, and explicit vocabulary handling.
    """
    def __init__(self, sentences: Collection[List[str]] = None):
        self.counter = NgramCounter()
        if sentences is not None:
            self.train(sentences)

    def train(self, sentences: Collection[List[str]]) -> None:
        self.counter.train(sentences)

    def _get_word_log_prob(self, word: str) -> float:
        """
        Returns the base 2 log probability of a word.
        Uses pure NLTK FreqDist.freq() since <UNK> replacement
        should be handled prior to/during training.
        """
        prob = self.counter.unigrams.freq(word)
        if prob == 0:
            # Fallback for unseen words if <UNK> wasn't properly applied
            # In a rigorous setup, we'd raise an error or assign minimum prob
            prob = 1e-10
        return math.log2(prob)

    def get_word_log_probability(self, sentence: List[str], index: int) -> float:
        word = sentence[index]
        return self._get_word_log_prob(word)

    def get_sentence_log_probability(self, sentence: List[str]) -> float:
        """
        Calculates the sum of log probabilities to avoid underflow.
        """
        padded = list(pad_sequence(sentence, n=1, **self.counter.padding))
        log_prob_sum = 0.0
        for word in padded:
            log_prob_sum += self._get_word_log_prob(word)
        return log_prob_sum

    def check_model(self) -> float:
        """
        Checks if the probability distribution properly sums up to approximately 1.
        """
        sum_prob = sum(self.counter.unigrams.freq(word) for word in self.counter.unigrams.keys())
        return sum_prob
        
    def generate_word(self) -> str:
        """
        Returns a random word sampled according to the model's empirical distribution.
        """
        sample = random.random()
        sum_prob = 0.0
        for word in self.counter.unigrams.keys():
            sum_prob += self.counter.unigrams.freq(word)
            if sum_prob > sample:
                return word
        return self.counter.unknown  # Fallback, should not happen if distribution is correct

    def generate_sentence(self) -> List[str]:
        """
        Returns a random sentence sampled according to the model.
        """
        sentence = []
        word = self.generate_word()
        # Avoid starting with the end token or unknown tokens optionally
        while word == "</s>":
            word = self.generate_word()
            if word == self.counter.unknown:
                word = self.generate_word()  # Try again if we get <UNK>
        while word != "</s>" and len(sentence) < 100: # add cutoff to prevent infinity
            if word != "<s>":
                sentence.append(word)
            word = self.generate_word()
            if word == self.counter.unknown:
                word = self.generate_word()  # Try again if we get <UNK>
        return sentence
