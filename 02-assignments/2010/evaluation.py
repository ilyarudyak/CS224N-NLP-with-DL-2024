import math
from typing import List, Collection, Type, Any
import numpy as np
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline
from nltk.lm import NgramCounter, Vocabulary

class NltkModelFactory:
    """
    Factory for creating and training NLTK-based language models.
    """
    @staticmethod
    def train(
        model_class: Type,
        order: int,
        train_set: Any,
        unk_cutoff: int = 2
    ):
        """
        Automates the NLTK training pipeline using the dataset's precomputed all_words.
        """
        # 1. Vocabulary - Use the pre-flattened list from the dataset
        vocab = Vocabulary(train_set.all_words, unk_cutoff=unk_cutoff)

        # 2. Masking
        masked_text = [list(vocab.lookup(sent)) for sent in train_set.sentences]

        # 3. Pipeline
        train_data, _ = padded_everygram_pipeline(order, masked_text)

        # 4. Counter
        counter = NgramCounter()
        counter.update(train_data)

        # 5. Model
        return model_class(order, vocabulary=vocab, counter=counter)


class NltkLanguageModelTester:
    """
    Evaluator specifically for NLTK models using native library methods.
    """
    def __init__(self, model):
        self.model = model

    def compute_perplexity(self, sentences: Collection[List[str]]) -> float:
        """
        Uses NLTK's native perplexity(text_ngrams) method.
        """
        # Prepare the data as a stream of ngrams, just as NLTK's .perplexity() expects
        # We need to pad the sequences using the model's order
        from nltk.lm.preprocessing import flatten
        
        # NLTK's perplexity expects a flat list of n-grams
        # We use the internal _split_padded or similar logic to get the test n-grams
        test_data = []
        for sent in sentences:
            # We don't mask test data here because NLTK's model.score() 
            # handles unknown words via the internal vocabulary lookup.
            padded = list(pad_sequence(
                sent, 
                n=self.model.order, 
                pad_left=True, pad_right=True,
                left_pad_symbol="<s>", right_pad_symbol="</s>"
            ))
            # Generate ngrams of the specified order
            sent_ngrams = list(ngrams(padded, self.model.order))
            test_data.extend(sent_ngrams)

        return self.model.perplexity(test_data)

    def run_full_evaluation(self, test_sets: dict):
        """
        Runs perplexity tests using native NLTK logic.
        """
        print(f"--- Native NLTK Evaluation for {self.model.__class__.__name__} ---")
        
        # Self-consistency check (using the model's internal method)
        avg_prob, success_rate = self.model.check_model()
        print(f"Model Integrity (average probability): {avg_prob:.4f}")
        print(f"Model Integrity (success rate): {success_rate:.4f}")
        
        for name, data in test_sets.items():
            pp = self.compute_perplexity(data)
            print(f"Native Perplexity on {name}: {pp:.2f}")
        print("-" * 40)


class LanguageModelTester:
    """
    Coordinator class for evaluating manual language models on different datasets.
    """
    def __init__(self, model=None):
        self.model = model

    def compute_perplexity_manual(self, sentences: Collection[List[str]]) -> float:
        """
        Manually computes perplexity using the formula:
        PP(S) = 2^(-1/N * sum(log2(P(w|h))))
        where N is the total number of items the model predicted (including stop token).
        """
        total_log_prob = 0.0
        total_tokens = 0

        # Retrieve metadata from model's counter for universal compatibility
        n = self.model.counter.n
        padding_args = self.model.counter.padding

        for sentence in sentences:
            # log_probability should already account for padding internally
            log_prob = self.model.get_sentence_log_probability(sentence)
            total_log_prob += log_prob
            
            # Count the actual number of predictions made
            # For n=1: we predict every token in padded sequence
            # For n=2: we skip the first <s> because it's just context
            padded = list(pad_sequence(sentence, n=n, **padding_args))
            num_predictions = len(padded) - (n - 1)
            total_tokens += num_predictions

        if total_tokens == 0:
            return float('inf')

        # Average log probability per token
        avg_log_prob = total_log_prob / total_tokens
        return math.pow(2, -avg_log_prob)

    def compute_perplexity_nltk(self, sentences: Collection[List[str]]) -> float:
        """
        Simulates NLTK's entropy-based perplexity calculation.
        """
        total_log_prob = 0.0
        total_tokens = 0
        n = self.model.counter.n
        padding_args = self.model.counter.padding
        
        for sentence in sentences:
            log_prob = self.model.get_sentence_log_probability(sentence)
            total_log_prob += log_prob
            
            padded = list(pad_sequence(sentence, n=n, **padding_args))
            total_tokens += len(padded) - (n - 1)
            
        cross_entropy = -total_log_prob / total_tokens
        return math.pow(2, cross_entropy)

    def run_full_evaluation(self, train_set, test_sets: dict):
        """
        Runs perplexity tests across multiple datasets.
        """
        print(f"--- Evaluation for {self.model.__class__.__name__} ---")
        
        # Self-consistency check
        avg_prob, success_rate = self.model.check_model()
        print(f"Model Integrity (average probability): {avg_prob:.4f}")
        print(f"Model Integrity (success rate): {success_rate:.4f}")
        
        for name, data in test_sets.items():
            pp = self.compute_perplexity_manual(data)
            print(f"Perplexity on {name}: {pp:.2f}")
        print("-" * 40)
