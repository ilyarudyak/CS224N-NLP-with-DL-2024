import math
import random
from typing import List, Collection
from nltk.probability import FreqDist
from nltk.lm.preprocessing import pad_sequence
from nltk.util import ngrams

class NgramCounter:
    """
    Decoupled counter for n-grams. Currently supports unigrams ONLY.
    """
    def __init__(self, n=1, vocabulary: Collection[str] = None, sentences: Collection[List[str]] = None):
        self.n = n
        self.vocabulary = vocabulary
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

    @property
    def is_trained(self):
        """
        Returns True if the counter has been trained on at least one word.
        """
        return self.unigrams.N() > 0

    def train(self, sentences: Collection[List[str]]) -> None:
        for sentence in sentences:
            # Check each word against the vocabulary and replace with <UNK> if not found
            checked_sent = [self.check_against_vocab(word) for word in sentence]
            # Generate padded unigrams for the checked sentence and update the FreqDist
            sentence_unigrams = self.to_ngrams(checked_sent)
            # Update the unigram counts with the new sentence's unigrams
            self.unigrams.update(sentence_unigrams)
        # Update the total count of unigrams after processing all sentences
        self.total_unigrams = self.unigrams.N()

    # This method checks if a word is in the vocabulary, and if not, it returns the unknown token <UNK>.
    def check_against_vocab(self, word):
        if self.vocabulary is None or word in self.vocabulary:
            return word
        return self.unknown

    # This method uses the NLTK ngrams function to generate n-grams
    # for a given n, 
    # with custom padding on both sides.
    def to_ngrams(self, sequence):
        """
        Wrapper for NLTK ngrams method
        """
        return ngrams(sequence, self.n, **self.padding)


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
    def __init__(self, vocabulary: Collection[str] = None, sentences: Collection[List[str]] = None):
        self.counter = NgramCounter(vocabulary=vocabulary, sentences=sentences)
        if sentences is not None:
            self.train(sentences)

    def train(self, sentences: Collection[List[str]]) -> None:
        if not self.counter.is_trained:
            self.counter.train(sentences)

    def _get_word_log_prob(self, word: str) -> float:
        """
        Returns the base 2 log probability of a word.
        Internal storage uses tuples for consistency with NLTK ngrams().
        """
        # Convert string word to unigram tuple for FreqDist lookup
        token = (word,)
        prob = self.counter.unigrams.freq(token)
        if prob == 0:
            # Fallback for unseen words (e.g., if <UNK> wasn't in train)
            prob = 1e-10
        return math.log2(prob)

    def get_word_log_probability(self, sentence: List[str], index: int) -> float:
        word = sentence[index]
        return self._get_word_log_prob(word)

    def get_sentence_log_probability(self, sentence: List[str]) -> float:
        """
        Calculates the sum of log probabilities for the whole padded sequence.
        """
        # We must pad here to match the training distribution (<s>/</s>)
        padded = list(pad_sequence(sentence, n=1, **self.counter.padding))
        log_prob_sum = 0.0
        for word in padded:
            log_prob_sum += self._get_word_log_prob(word)
        return log_prob_sum

    def check_model(self) -> float:
        """
        Checks if the probability distribution properly sums up to approximately 1.
        """
        sum_prob = sum(self.counter.unigrams.freq(token) for token in self.counter.unigrams.keys())
        return sum_prob
        
    def generate_word(self) -> str:
        """
        Returns a random word sampled according to the model's empirical distribution.
        """
        sample = random.random()
        sum_prob = 0.0
        for token in self.counter.unigrams.keys():
            sum_prob += self.counter.unigrams.freq(token)
            if sum_prob > sample:
                return token[0] # Return the string from the tuple
        return self.counter.unknown 

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
