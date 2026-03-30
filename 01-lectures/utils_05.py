from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random

class TrigramModel:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, sentences):
        """Train the model on a list of sentences (each sentence is a list of words)."""
        # Count frequencies
        for sentence in sentences:
            for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                self.model[(w1, w2)][w3] += 1
        
        # Transform counts to probabilities
        for context in self.model:
            total_count = float(sum(self.model[context].values()))
            if total_count > 0:
                for w3 in self.model[context]:
                    self.model[context][w3] /= total_count

    def generate(self, seed_words=["today", "the"]):
        """Generate a sentence starting with seed_words."""
        text = list(seed_words)
        sentence_finished = False
        
        # Limit generation to avoid infinite loops if no terminator is found
        max_length = 50
        
        while not sentence_finished and len(text) < max_length:
            r = random.random()
            accumulator = 0.0
            
            context = tuple(text[-2:])

            # Get possible next words and their probabilities
            possible_words = self.model[context]
            
            if not possible_words:
                break

            # We iterate through the possible words and their probabilities, 
            # accumulating the probabilities until we exceed r - Roulette Wheel Selection
            for word, prob in possible_words.items():
                accumulator += prob
                if accumulator >= r:
                    text.append(word)
                    break
            
            if text[-2:] == [None, None]:
                sentence_finished = True
        
        return ' '.join([t for t in text if t])
