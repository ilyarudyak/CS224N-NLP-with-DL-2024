import unittest
import torch
import random
from dataset import CharCorruptionDataset

class TestCharCorruptionDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample mini-corpus with diverse sentence lengths
        cls.raw_data = (
            "Khatchig Mouradian. Khatchig Mouradian is a journalist born in Lebanon .\n"
            "Jacob Henry Studer. Born in Columbus, Ohio .\n"
            "Short doc.\n"
            "John Stephen. Born in Glasgow, Stephen became a welder's apprentice on leaving school .\n"
        )
        cls.block_size = 64
        cls.dataset = CharCorruptionDataset(cls.raw_data, cls.block_size)

    def test_tensor_shapes_and_types(self):
        """x and y must be 1D Long tensors of exact length block_size."""
        for idx in range(len(self.dataset) - 1):
            x, y = self.dataset[idx]
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x.dtype, torch.long)
            self.assertEqual(y.dtype, torch.long)
            self.assertEqual(x.shape, (self.dataset.block_size,))
            self.assertEqual(y.shape, (self.dataset.block_size,))

    def test_autoregressive_shift(self):
        """y must be x shifted by 1 position: y[:-1] == x[1:]."""
        for idx in range(len(self.dataset) - 1):
            x, y = self.dataset[idx]
            self.assertTrue(
                torch.equal(x[1:], y[:-1]),
                f"Shift alignment failed at sample {idx}!"
            )

    def test_token_validity(self):
        """All token IDs must lie within [0, vocab_size - 1]."""
        for idx in range(len(self.dataset) - 1):
            x, y = self.dataset[idx]
            self.assertTrue((x >= 0).all() and (x < self.dataset.vocab_size).all())
            self.assertTrue((y >= 0).all() and (y < self.dataset.vocab_size).all())

    def test_delimiter_and_pad_structure(self):
        """
        Decoded string must have:
        1. Exactly two MASK_CHAR delimiters.
        2. PAD_CHAR only at the end.
        """
        mask_char = self.dataset.MASK_CHAR
        pad_char = self.dataset.PAD_CHAR
        
        for idx in range(len(self.dataset) - 1):
            x, y = self.dataset[idx]
            # reconstruct full masked_string: x + [last character of y]
            full_str = "".join([self.dataset.itos[i.item()] for i in x]) + self.dataset.itos[y[-1].item()]
            self.assertEqual(len(full_str), self.dataset.block_size + 1)
            
            # Check mask characters
            mask_count = full_str.count(mask_char)
            self.assertEqual(mask_count, 2, f"Expected 2 MASK_CHARs, got {mask_count} in '{full_str}'")

            # Check padding is strictly at the tail
            first_pad = full_str.find(pad_char)
            if first_pad != -1:
                tail = full_str[first_pad:]
                self.assertTrue(
                    all(c == pad_char for c in tail),
                    f"Non-pad character found after pad start: '{tail}'"
                )

    def test_content_preservation(self):
        """
        prefix + masked_content + suffix must equal the original truncated document.
        """
        mask_char = self.dataset.MASK_CHAR
        pad_char = self.dataset.PAD_CHAR
        
        for idx in range(len(self.dataset) - 1):
            x, y = self.dataset[idx]
            full_str = "".join([self.dataset.itos[i.item()] for i in x]) + self.dataset.itos[y[-1].item()]
            
            # strip trailing pads
            unpadded = full_str.rstrip(pad_char)
            # format is: prefix + MASK_CHAR + suffix + MASK_CHAR + masked_content
            parts = unpadded.split(mask_char)
            self.assertEqual(len(parts), 3, f"Split by MASK_CHAR failed on '{unpadded}'")
            prefix, suffix, masked_content = parts[0], parts[1], parts[2]
            
            reconstructed = prefix + masked_content + suffix
            orig_doc = self.dataset.data[idx]
            self.assertTrue(
                orig_doc.startswith(reconstructed),
                f"Reconstructed '{reconstructed}' is not a prefix of original '{orig_doc}'"
            )

    def test_truncation_randomness(self):
        """Multiple calls for the same index should yield different lengths/spans."""
        lengths = set()
        for _ in range(30):
            x, y = self.dataset[0]
            full_str = "".join([self.dataset.itos[i.item()] for i in x]) + self.dataset.itos[y[-1].item()]
            unpadded = full_str.rstrip(self.dataset.PAD_CHAR)
            lengths.add(len(unpadded))
        
        # Verify that truncation produces varying lengths
        self.assertGreater(len(lengths), 1, "Length of truncated doc is not random across calls!")

if __name__ == '__main__':
    unittest.main()