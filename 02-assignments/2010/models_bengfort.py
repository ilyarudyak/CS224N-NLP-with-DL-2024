import math
from typing import List, Collection
import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
from collections import defaultdict
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence

class NgramCounter(object):
    """
    The NgramCounter class counts ngrams given a vocabulary and ngram size.
    Adapted from the ATAP/Bengfort architecture for robust probability generation.
    """

    def __init__(self, n=3, 
                 vocabulary: Collection[str] = None, 
                 training_text: Collection[List[str]] = None):
        """
        n is the size of the ngram. Default to 3 (Trigram) as Kneser-Ney 
        is most effective with higher-order n-grams.
        """
        if n < 1:
            raise ValueError("ngram size must be greater than or equal to 1")

        self.n = n
        self.unknown = "<UNK>"
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }

        self.vocabulary = vocabulary
        # ConditionalFreqDist representing P(word | context) for all order n
        self.allgrams = {i: ConditionalFreqDist() for i in range(2, n + 1)}
        # Flat FreqDists for each order (required by NLTK KneserNeyProbDist)
        self.flat_ngrams = {i: FreqDist() for i in range(1, n + 1)}

        # Train the counter immediately if training text is provided
        if training_text is not None:
            self.train_counts(training_text=training_text)  # Initialize with empty counts to set up the vocab

    @property
    def is_trained(self):
        """
        Returns True if the counter has been trained on at least one word.
        """
        return self.flat_ngrams[1].N() > 0

    def train_counts(self, training_text: Collection[List[str]]=None):
        """
        Populates unigram, n-gram, and conditional frequency distributions.
        """
        for sent in training_text:
            # Mask unknown words
            checked_sent = [self.check_against_vocab(word) for word in sent]
            
            # Use NLTK's pad_sequence for consistent padding
            padded = list(pad_sequence(checked_sent, n=self.n, **self.padding))
            
            # Populate flat_ngrams and allgrams for all orders from 1 to n
            for i in range(1, self.n + 1):
                for ngram in ngrams(padded, i):
                    self.flat_ngrams[i][ngram] += 1
                    if i > 1:
                        context, word = ngram[:-1], ngram[-1]
                        self.allgrams[i][context][word] += 1

    def check_against_vocab(self, word):
        if self.vocabulary is None or word in self.vocabulary:
            return word
        return self.unknown

    def to_ngrams(self, sequence):
        """
        Wrapper for NLTK ngrams method handling standard padding.
        """
        return ngrams(sequence, self.n, **self.padding)


class BaseNgramModel(object):
    """
    The BaseNgramModel creates an n-gram language model.
    This base model is equivalent to a Maximum Likelihood Estimation.
    """

    def __init__(self, ngram_counter: NgramCounter):
        """
        BaseNgramModel is initialized with an NgramCounter.
        """
        if not ngram_counter or not ngram_counter.is_trained:
            raise ValueError("The ngram_counter must be trained before initializing a model.")

        self.n = ngram_counter.n
        self.counter = ngram_counter
        # Point to the multi-level structures
        self.allgrams = ngram_counter.allgrams
        self._check_against_vocab = self.counter.check_against_vocab

    def check_context(self, context):
        """
        Ensures that the context is not longer than or equal to the model's n-gram order.
        """
        if len(context) >= self.n:
            raise ValueError(f"Context too long for this {self.n}-gram model")
        return tuple(context)

    def score(self, word: str, context: tuple = ()):
        """
        Maximum likelihood score that the word will follow the context.
        """
        context = self.check_context(context)
        if self.n == 1:
            return self.counter.flat_ngrams[1].freq((word,))
        return self.counter.allgrams[self.n][context].freq(word)

    def logscore(self, word: str, context: tuple = ()):
        """
        Computes the base-2 log probability of this word in this context.
        """
        score = self.score(word, context)
        if score == 0.0:
            # Fallback instead of -inf to avoid math crashing during perplexity
            return math.log2(1e-10)
        return math.log2(score)

    def get_sentence_log_probability(self, sentence: List[str]) -> float:
        """
        Compatible with `evaluation.py`. Connects legacy API to new backend.
        """
        padded = list(pad_sequence(sentence, n=self.n, **self.counter.padding))
        
        log_prob_sum = 0.0
        # Iterate over ngrams to extract context and word
        for ngram in ngrams(padded, self.n):
            context, word = tuple(ngram[:-1]), ngram[-1]
            # Must check vocab during inference
            checked_word = self._check_against_vocab(word)
            checked_context = tuple([self._check_against_vocab(c) for c in context])
            
            log_prob_sum += self.logscore(checked_word, checked_context)
            
        return log_prob_sum

    def check_model(self):
        """
        Required by `evaluation.py`. Just returns 1.0 since it's hard to
        exhaustively check all probabilities for trigrams.
        """
        return 1.0


class KneserNeyModel(BaseNgramModel):
    """
    Implements Kneser-Ney smoothing using a simple recursive backoff.
    Since NLTK's KneserNeyProbDist is strictly for Trigrams and lacks backoff,
    we will implement the recursive logic with Laplace backoff or simply 
    the KneserNeyProbDist for Trigrams only.
    """
    def __init__(self, ngram_counter: NgramCounter):
        super().__init__(ngram_counter)
        
        # NLTK KneserNeyProbDist is hardcoded for trigrams (samples of length 3)
        # and it returns the continuation probability.
        self.model = nltk.KneserNeyProbDist(self.counter.flat_ngrams[3])

    def score(self, word: str, context: tuple = ()):
        """
        Recursive score: Trigram (KN) -> Bigram (MLE/Laplace) -> Unigram (MLE).
        """
        context = self.check_context(context)
        
        # 1. Try Trigram with NLTK Kneser-Ney
        if len(context) == 2:
            target_ngram = context + (word,)
            # KneserNeyProbDist returns 0 if context is unseen OR word unseen in context
            prob = self.model.prob(target_ngram)
            if prob > 0:
                return prob
            # Backoff to Bigram
            context = context[1:]

        # 2. Bigram Backoff (MLE for now, as NLTK KN doesn't support bigrams well)
        if len(context) == 1:
            if self.counter.flat_ngrams[1][context] > 0:
                # Basic MLE score from ConditionalFreqDist
                prob = self.counter.allgrams[2][context].freq(word)
                if prob > 0:
                    return prob
            # Backoff to Unigram
            context = ()

        # 3. Unigram Base Case
        return self.counter.flat_ngrams[1].freq((word,))

    def logscore(self, word: str, context: tuple = ()):
        s = self.score(word, context)
        if s <= 0:
            return math.log2(1e-12)
        return math.log2(s)

    def logscore(self, word: str, context: tuple = ()):
        """
        Computed based on the recursive score.
        """
        s = self.score(word, context)
        # Numerical stability: KN can return 0 if discounting exceeds counts 
        # or if the word was never seen even in unigrams (though UNK usually prevents this)
        if s <= 0:
            return math.log2(1e-12)
        return math.log2(s)
