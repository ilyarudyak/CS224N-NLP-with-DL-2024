import random
from typing import List, Collection, Iterable
from nltk.lm.api import LanguageModel
from nltk.lm import Vocabulary, NgramCounter, KneserNeyInterpolated, StupidBackoff, WittenBellInterpolated
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline

class NltkModelBase(LanguageModel):
    """
    Base class for NLTK-based models to provide common evaluation and generation methods.
    """
    def __init__(self, order: int, vocabulary: Vocabulary = None, counter: NgramCounter = None):
        super().__init__(order, vocabulary, counter)
        self.counter = self.counts
        if self.counter:
            self.counter.n = self.order
            if not hasattr(self.counter, 'padding'):
                self.counter.padding = {
                    "pad_left": True,
                    "pad_right": True,
                    "left_pad_symbol": "<s>",
                    "right_pad_symbol": "</s>"
                }

    def get_sentence_log_probability(self, sentence: List[str]) -> float:
        """Compatibility method for LanguageModelTester."""
        return sum(self.logscore(w, tuple(c)) for w, c in self._split_padded(sentence))

    def _split_padded(self, sentence):
        """Helper to pad and split sentence into word/context pairs."""
        from nltk.util import ngrams
        padded = list(pad_sequence(sentence, n=self.order, pad_left=True, pad_right=True, 
                                 left_pad_symbol="<s>", right_pad_symbol="</s>"))
        for ngram in ngrams(padded, self.order):
            yield ngram[-1], ngram[:-1]

    def check_model(self, num_samples=100):
        """
        Modified check_model for NLTK API.
        Returns: (average_sum, success_rate)
        """
        if not self.vocab:
            raise ValueError("Model must have a vocabulary.")
            
        success_count = 0
        vocab_list = list(self.vocab)
        
        if self.order == 1:
            total_prob = sum(self.score(w) for w in vocab_list)
            success = 1 if abs(total_prob - 1.0) < 1e-5 else 0
            return total_prob, success

        contexts = list(self.counts[self.order].keys())
        if not contexts:
            return 0.0, 0.0
            
        # NLTK models (especially Kneser-Ney) are extremely slow to sum over whole vocab
        local_samples = min(num_samples, 5)
        context_samples = random.sample(contexts, min(local_samples, len(contexts)))
        total_sum = 0.0
        for context in context_samples:
            current_sum = sum(self.score(w, context) for w in vocab_list)
            total_sum += current_sum
            if abs(current_sum - 1.0) < 1e-5:
                success_count += 1

        return total_sum / len(context_samples), success_count / len(context_samples)

    def generate_sentence(self, num_words: int = 20) -> List[str]:
        """
        Generates a sentence using NLTK's built-in sampler.
        """
        tokens = self.generate(num_words, text_seed=["<s>"])
        result = []
        for t in tokens:
            if t == "</s>":
                break
            if t != "<s>":
                result.append(t)
        return result

class EmpiricalUnigramNltkModel(NltkModelBase):
    """
    A professional Unigram model leveraging the official NLTK LanguageModel API.
    """
    def unmasked_score(self, word: str, context: tuple = None) -> float:
        prob = self.counts.unigrams.freq(word)
        if prob == 0:
            return 1e-10
        return prob

class KneserNeyNltkModel(KneserNeyInterpolated, NltkModelBase):
    """
    NLTK's Kneser-Ney Interpolated model.
    """
    def __init__(self, order, vocabulary=None, counter=None):
        KneserNeyInterpolated.__init__(self, order, vocabulary=vocabulary, counter=counter)
        NltkModelBase.__init__(self, order, vocabulary=vocabulary, counter=counter)

class StupidBackoffNltkModel(StupidBackoff, NltkModelBase):
    """
    NLTK's Stupid Backoff model.
    """
    def __init__(self, order, vocabulary=None, counter=None, **kwargs):
        # StupidBackoff(alpha=0.4, order=3, vocabulary=..., counter=...)
        # We manually separate alpha if provided, else use default 0.4
        alpha = kwargs.pop('alpha', 0.4)
        StupidBackoff.__init__(self, alpha, order, vocabulary=vocabulary, counter=counter)
        NltkModelBase.__init__(self, order, vocabulary=vocabulary, counter=counter)

class WittenBellNltkModel(WittenBellInterpolated, NltkModelBase):
    """
    NLTK's Witten-Bell Interpolated model.
    """
    def __init__(self, order, vocabulary=None, counter=None):
        WittenBellInterpolated.__init__(self, order, vocabulary=vocabulary, counter=counter)
        NltkModelBase.__init__(self, order, vocabulary=vocabulary, counter=counter)

