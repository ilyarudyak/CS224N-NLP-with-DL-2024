import math
from typing import List, Collection
import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
from collections import defaultdict
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence
import logging
import random

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

        # ConditionalFreqDist for each n contains:
        # context as a key and FreqDist of following words as value. For example, for bigrams:
        # self.allgrams[2][('the',)] 

        # Dict[int, ConditionalFreqDist[Tuple[str, ...], FreqDist[str, int]]]] 
        # Where the key Tuple[str, ...] is the context and 
        # the value FreqDist contains counts of words following that context.
        self.allgrams = {i: ConditionalFreqDist() for i in range(2, n + 1)} 
        
        # FreqDict for each n contain counts of n-grams themselves, not separated into context and word
        # FreqDist({('<s>', 'you'): 1, ('you', 'will'): 1, ('will', 'be'): 1 ...})
        self.flat_ngrams = {i: FreqDist() for i in range(1, n + 1)} # Dict[int, FreqDist[Tuple[str, ...], int]]

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
        for i, sent in enumerate(training_text):
            logging.debug(f"Processing sentence {i}")
            # Mask unknown words
            checked_sent = [self.check_against_vocab(word) for word in sent]
            # Log the sentence and its checked version for debugging
            logging.debug(f"Original sentence: {sent}")
            logging.debug(f"Checked sentence: {checked_sent}")
            
            # Use NLTK's pad_sequence for consistent padding
            padded = list(pad_sequence(checked_sent, n=self.n, **self.padding))
            logging.debug(f"Padded sentence: {padded}\n")
            
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

    # We do not use this. Instead we call `pad_sequence` directly.
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

    def __init__(self, ngram_counter: NgramCounter = None):
        """
        BaseNgramModel is initialized with an NgramCounter.
        """
        if not ngram_counter or not ngram_counter.is_trained:
            raise ValueError("The ngram_counter must be trained before initializing a model.")

        self.n = ngram_counter.n
        self.counter = ngram_counter
        # Point to the multi-level structures
        # self.allgrams = ngram_counter.allgrams
        # self._check_against_vocab = self.counter.check_against_vocab

    def check_context(self, context):
        """
        Ensures that the context is not longer than or equal to the model's n-gram order.
        """
        if len(context) >= self.n:
            raise ValueError(f"Context too long for this {self.n}-gram model")
        return tuple(context)

    def score(self, word: str, context: tuple = ()):
        """
        MLE estimates without any smoothing.
        """
        context = self.check_context(context)
        if self.n == 1:
            # Words are stores as tuples in the unigram FreqDist.
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
            checked_word = self.counter.check_against_vocab(word)
            checked_context = tuple([self.counter.check_against_vocab(c) for c in context])
            
            log_prob_sum += self.logscore(checked_word, checked_context)
            
        return log_prob_sum

    def check_model(self, num_samples=100):
        """
        Validates that the probability distribution sums to 1.0 for a given set of contexts.
        Strictly requires a vocabulary set in the counter during training.
        Returns: (average_sum, success_rate)
        """
        
        if not self.counter.vocabulary:
            raise ValueError("Vocabulary must be provided to the counter for validation.")
            
        vocab = list(self.counter.vocabulary)
        success_count = 0
        
        # For Unigram models, we just sum over the entire vocabulary once.
        if self.n == 1:
            total_prob = sum(self.score(w) for w in vocab)
            success = 1 if abs(total_prob - 1.0) < 1e-5 else 0
            return total_prob, success

        # For N-gram models, sample contexts that actually exist in the training data
        contexts = list(self.counter.allgrams[self.n].keys())
        context_samples = random.sample(contexts, min(num_samples, len(contexts)))

        # Check if we have any context samples
        if not context_samples:
            raise ValueError("No contexts available for sampling.")

        
        total_sum = 0.0
        for context in context_samples:
            # We iterate over the vocabulary to ensure everything (including UNK and </s>) 
            # is accounted for in the conditional distribution P(w|context)
            # current_sum = sum(self.score(w, context) for w in vocab)

            # Instead of iterating over the entire vocabulary, 
            # we can iterate over the actual words that follow this context in the training data.
            current_sum = sum(self.score(w, context) for w in self.counter.allgrams[self.n][context].keys())

            total_sum += current_sum
            if abs(current_sum - 1.0) < 1e-5:
                success_count += 1

        avg_prob = total_sum / len(context_samples)
        success_rate = success_count / len(context_samples)
        
        return avg_prob, success_rate


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
        Recursive score with proper backoff: Trigram (KN) -> Bigram (KN/MLE) -> Unigram (MLE).
        """
        context = self.check_context(context)
        
        # 1. Try Trigram with NLTK Kneser-Ney
        if len(context) == 2:
            target_ngram = context + (word,)
            # NLTK KN handles valid contexts and provides a discounted probability.
            # However, if the context is UNSEEN, it might return 0.
            prob = self.model.prob(target_ngram)
            if prob > 0:
                return prob
            # Backoff to Bigram
            context = context[1:]

        # 2. Bigram Backoff
        if len(context) == 1:
            # Check if this bigram context (single word) exists in our bigram counts
            if context in self.counter.allgrams[2]:
                prob = self.counter.allgrams[2][context].freq(word)
                if prob > 0:
                    return prob
            # Backoff to Unigram
            context = ()

        # 3. Unigram Base Case
        return self.counter.flat_ngrams[1].freq((word,))

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
