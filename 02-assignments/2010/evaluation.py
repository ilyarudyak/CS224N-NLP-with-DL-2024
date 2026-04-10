import math
from typing import List, Collection
import numpy as np
from nltk.lm.preprocessing import pad_sequence

class LanguageModelTester:
    """
    Coordinator class for evaluating language models on different datasets.
    """
    def __init__(self, model):
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
        _, sum_prob = self.model.check_model()
        print(f"Model Integrity (sum of P(w)): {sum_prob:.4f}")
        
        for name, data in test_sets.items():
            pp = self.compute_perplexity_manual(data)
            print(f"Perplexity on {name}: {pp:.2f}")
        print("-" * 40)
