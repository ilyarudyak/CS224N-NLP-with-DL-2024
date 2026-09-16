import string
from collections import Counter


class BleuScoreV1:
    @staticmethod
    def tokenize(sentence: str) -> list[str]:
        translation_table = str.maketrans("", "", string.punctuation)

        return [
            token.translate(translation_table).lower()
            for token in sentence.split()
        ]

    def modified_unigram_precision(
        self,
        candidate: str,
        references: list[str],
    ) -> tuple[int, int, float]:
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = [
            self.tokenize(reference)
            for reference in references
        ]

        candidate_counts = Counter(candidate_tokens)
        maximum_reference_counts = Counter()

        for tokens in reference_tokens:
            reference_counts = Counter(tokens)

            for word, count in reference_counts.items():
                maximum_reference_counts[word] = max(
                    maximum_reference_counts[word],
                    count,
                )

        clipped_count = sum(
            min(count, maximum_reference_counts[word])
            for word, count in candidate_counts.items()
        )

        candidate_count = len(candidate_tokens)
        modified_precision = clipped_count / candidate_count

        return clipped_count, candidate_count, modified_precision


class BleuScoreV2(BleuScoreV1):
    @staticmethod
    def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        if n < 1:
            raise ValueError("n must be at least 1")

        return [
            tuple(tokens[index:index + n])
            for index in range(len(tokens) - n + 1)
        ]

    def modified_precision(
        self,
        candidate: str,
        references: list[str],
        n: int,
    ) -> tuple[int, int, float]:
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = [
            self.tokenize(reference)
            for reference in references
        ]

        candidate_ngrams = self.ngrams(candidate_tokens, n)
        reference_ngrams = [
            self.ngrams(tokens, n)
            for tokens in reference_tokens
        ]

        candidate_counts = Counter(candidate_ngrams)
        maximum_reference_counts = Counter()

        for ngrams in reference_ngrams:
            reference_counts = Counter(ngrams)

            for ngram, count in reference_counts.items():
                maximum_reference_counts[ngram] = max(
                    maximum_reference_counts[ngram],
                    count,
                )

        clipped_count = sum(
            min(count, maximum_reference_counts[ngram])
            for ngram, count in candidate_counts.items()
        )

        candidate_count = len(candidate_ngrams)

        if candidate_count == 0:
            return 0, 0, 0.0

        modified_precision = clipped_count / candidate_count

        return clipped_count, candidate_count, modified_precision

    def modified_unigram_precision(
        self,
        candidate: str,
        references: list[str],
    ) -> tuple[int, int, float]:
        return self.modified_precision(candidate, references, n=1)

    def modified_bigram_precision(
        self,
        candidate: str,
        references: list[str],
    ) -> tuple[int, int, float]:
        return self.modified_precision(candidate, references, n=2)