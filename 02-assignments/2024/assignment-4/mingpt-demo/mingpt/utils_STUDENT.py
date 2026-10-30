import torch
from torch.utils.data import Dataset

class CharTokenizer:

    def __init__(self, text: str):

        # Build the vocabulary from the input text
        self._build_vocab(text)

    def _build_vocab(self, text):
        chars = sorted(list(set(text)))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = len(chars)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, indices: list):
        return ''.join([self.itos[int(i)] for i in indices])


class CharDataset(Dataset):

    def __init__(self, 
                 data: str, 
                 tokenizer: CharTokenizer, 
                 block_size: int = 128):

        # Set the data and the tokenizer for the dataset
        self.data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
        self.tokenizer = tokenizer

        # Set the maximum sequence length for the dataset
        # We can not process sequences longer than block_size: 
        #    1) due to positional encoding limitations in the model, 
        #    2) due to future masking of fixed size.
        self.block_size = block_size

        # Report the number of characters in the dataset
        print(f'Read {len(self.data):,} characters')
        # Report the block size for the dataset
        print(f'Set block size to {self.block_size}')
        # Report the vocabulary size for the dataset
        print(f'Vocabulary size: {self.tokenizer.vocab_size}')
        
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        # (1) Grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]

        # (2) Create input and target sequences
        # Remove the LAST character for the input sequence 
        x = chunk[:-1]
        # Remove the FIRST character for the target sequence
        y = chunk[1:]

        return x, y
