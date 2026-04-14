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
            # MUST iterate over the entire vocabulary to account for 
            # probability mass shifted to unseen words by lambda
            current_sum = sum(self.score(w, context) for w in vocab)

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

        # 2. Bigram Backoff: return vanilla MLE for bigrams.
        if len(context) == 1:
            # Check if this bigram context (single word) exists in our bigram counts
            if context in self.counter.allgrams[2]:
                prob = self.counter.allgrams[2][context].freq(word)
                if prob > 0:
                    return prob
            # Backoff to Unigram
            context = ()

        # 3. Unigram Base Case: return vanilla MLE unigram probability.
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


class InterpolatedNgramModel(BaseNgramModel):
    """
    Provides interpolated n-gram scores.
    """
    # Note: We now require `ngram_counter` as the first argument for the new architecture
    def __init__(self, ngram_counter, lambdas=(0.1, 0.3, 0.6)):
        super().__init__(ngram_counter)

        if len(lambdas) != self.n:
            raise ValueError(f"Length of lambdas ({len(lambdas)}) must match n-gram order ({self.n})")

        self.lambdas = lambdas

    def score(self, word: str, context: tuple = ()):
        """
        With interpolated n-grams, the score is a weighted sum of the probabilities
        from all n-gram orders up to n.
        """
        context = self.check_context(context)
        score_val = 0.0

        for i in range(self.n):
            n = i + 1
            lambda_weight = self.lambdas[i]
            ngram_prob = self._get_ngram_prob(word, context, n)
            score_val += lambda_weight * ngram_prob

        return score_val
    
    def _get_ngram_prob(self, word, context, n):
        """
        Helper method to get the probability of a word given a context for a specific n-gram order.
        """
        if n == 1:
            # Unigrams are accessed slightly differently in the new NgramCounter
            return self.counter.flat_ngrams[1].freq((word,))
        else:
            # For Bigrams and Trigrams, slice the context properly
            ngram_context = context[-(n-1):]
            freqdict = self.counter.allgrams[n][ngram_context]
            total = freqdict.N()
            if total == 0:
                return 0.0
            return freqdict[word] / total
        

class StupidBackoffNgramModel(BaseNgramModel):
    """
    Provides stupid backoff n-gram scores.
    """
    def __init__(self, ngram_counter, lambda_=0.4):
        super().__init__(ngram_counter)
        self.lambda_ = lambda_

    def score(self, word: str, context: tuple = ()):
        """
        Iterative Stupid Backoff. Backs off to lower order n-grams if count is 0,
        weighed by lambda_.
        """
        context = self.check_context(context)

        # Start with the highest order n-gram
        for n in range(self.n, 0, -1):
            if n == 1:
                # Unigram fallback using flat_ngrams
                unigram_prob = self.counter.flat_ngrams[1].freq((word,))
                if unigram_prob > 0:
                    return unigram_prob
            else:
                # Higher-order n-gram probability
                ngram_context = context[-(n-1):]
                freqdict = self.counter.allgrams[n][ngram_context]
                total = freqdict.N()
                
                if total > 0:
                    ngram_prob = freqdict[word] / total
                    if ngram_prob > 0:
                        return (self.lambda_ ** (self.n - n)) * ngram_prob

        # If all n-grams have zero counts (rare if <UNK> is handled)
        return 0.0


class AbsoluteDiscountingNgramModelRecursive(BaseNgramModel):
    """
    Provides absolute discounting n-gram scores using a recursive mathematical approach.
    """
    def __init__(self, ngram_counter, d=0.75):
        super().__init__(ngram_counter)
        self.d = d

    def score(self, word: str, context: tuple = ()):
        context = self.check_context(context)
        n = len(context) + 1

        # BASE CASE: The lowest order (unigrams)
        if n == 1:
            return self.base_score(word)

        # RECURSIVE STEP
        freqdict = self.counter.allgrams[n][context]
        count = freqdict[word]
        total = freqdict.N()

        if total > 0:
            discounted_mle = max(count - self.d, 0) / total
            lambda_weight = self._compute_lambda(context, n)
            # Recursive call: back off to (n-1) context
            return discounted_mle + lambda_weight * self.score(word, context[1:])
        else:
            # If context is unseen, back off entirely
            return self.score(word, context[1:])

    def base_score(self, word: str):
        """The unigram distribution. Overridden by Kneser-Ney."""
        return self.counter.flat_ngrams[1].freq((word,))

    def _compute_lambda(self, context: tuple, n: int):
        freqdict = self.counter.allgrams[n][context]
        if freqdict.N() == 0:
            return 1.0
        return (self.d * len(freqdict)) / freqdict.N()


class KneserNeyNgramModel(AbsoluteDiscountingNgramModelRecursive):
    """
    Kneser-Ney smoothing using mathematically rigorous recursion.
    Overrides Absolute Discounting by using the Continuation Probability for unigrams.
    """
    def __init__(self, ngram_counter, d=0.75):
        super().__init__(ngram_counter, d=d)
        
        # Precompute denominator for Continuation Probability: total unique bigram types
        # equivalent to |{(u, v): c(u, v) > 0}|
        self._total_bigram_types = sum(len(fd) for fd in self.counter.allgrams[2].values())

        # Precompute numerator: count how many unique contexts precede each word: |{v : C(vw) > 0}|
        from collections import Counter
        self._preceding_context_counts = Counter()
        for fd in self.counter.allgrams[2].values():
            for word in fd:
                self._preceding_context_counts[word] += 1

    def base_score(self, word: str):
        """Overrides MLE unigram with Continuation Probability."""
        if self._total_bigram_types == 0:
            return 0.0
            
        # Count how many unique contexts precede this word: |{v : C(vw) > 0}|
        # meaning, how many distinct bigrams end with this word?
        num_contexts = self._preceding_context_counts.get(word, 0)
        
        return num_contexts / self._total_bigram_types