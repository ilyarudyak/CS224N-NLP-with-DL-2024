import argparse
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

# [part e]

# CharCorruptionDataset is a class that yields examples of a simplified
# span corruption objective.
# Implement the __getitem__ function based on the following instructions.
# Do not change the signature of the __init__ or __getitem__ functions.

# Make sure to implement the full spec for full credit -- we list below the
# criteria that must be satisfied for a full implementation.

# --------------
# Masking Specification

# The __getitem__ function takes an index and returns a data point (x, y) where
# x and y are Long tensors of length self.block_size. x encodes the input
# sequence, and y encodes the output sequence.

# 0. Use the idx argument of __getitem__ to retrieve the element of self.data
#    at the given index. We'll call the resulting data entry a document.

# 1. Randomly truncate the document to a length no less than 4 characters,
#    and no more than int(self.block_size*7/8) characters.

# - IMPORTANT: You are free to decide how to perform this random truncation, but
# make sure that the length is picked _randomly_ (every possible length from 4
# to int(self.block_size*7/8) has a chance of being picked) for full credit.

# 2. Now, break the (truncated) document into three substrings:

#     [prefix] [masked_content] [suffix]

#   In other words, choose three strings prefix, masked_content and suffix
#     such that prefix + masked_content + suffix = [the truncated document].
#   The length of [masked_content] should be random, and 1/4 the length of the
#     truncated document on average.

# - IMPORTANT: You are free to decide how to perform this operation, but
# make sure that the length is picked _randomly_ (has a chance of being more or
# less than 1/4 the length of the truncated document) for full credit.

# 3. Rearrange these substrings into the following form:

#     [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]

#   This resulting string, denoted masked_string, serves as the output example.
#   Here MASK_CHAR is the masking character and [pads] is a string of repeated
#     PAD_CHAR characters chosen so that the entire string is of length
#     self.block_size + 1.
#   Intuitively, the string [masked_content] is removed from the document and
#     replaced with MASK_CHAR. After the suffix of the string, MASK_CHAR is seen
#     again, followed by the removed content, and the padding characters.

# 4. We now use masked_string to construct the input and output example pair. To
#    do so, simply take the input string to be masked_string[:-1], and the output
#    string to be masked_string[1:]. In other words, for each character, the goal
#    is to predict the next character in the masked string.

# 5. Making use of the vocabulary that you defined, encode the resulting input
#    and output strings as Long tensors and return the resulting data point.

# ----------------
# Here are some examples of input-output pairs (x, y):

#   x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

class CharCorruptionDataset(Dataset):
    def __init__(self, data: str, block_size: int = 128):
        self.MASK_CHAR = "\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        print(f'Unique characters in data: {len(chars)}')
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size:,} characters, {vocab_size} unique.')

        self.block_size = block_size
        self.vocab_size = vocab_size

        # Split the data into list of sentences by newline character
        # ['Khatchig Mouradian. Khatchig Mouradian is a journalist, writer and translator born in Lebanon .', 
        # 'Jacob Henry Studer. Jacob Henry Studer (26 February 1840 Columbus, Ohio - 2 August 1904 ...,
        self.data = data.split('\n')
        print(f'Number of lines in data: {len(self.data):,}')
        # Print the first few lines of the data for inspection
        print(f'First few lines of data: {self.data[:5]}')
        # Check if the last line is empty 
        print(f'Last line: {self.data[-1]}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        ### YOUR CODE HERE ###

        # Step 0. Retrieve the document at the given index from self.data
        document = self.data[idx]
        # print(f'Document at index {idx}: {document}')

        # Step 1. Truncate the document randomly
        document = self._truncate_document(document)
        # print(f'Truncated document: {document}')

        # Step 2. Break the truncated document into prefix, masked_content, and suffix
        prefix, masked_content, suffix = self._break_document(document)
        # print(f'[{prefix}] [{masked_content}] [{suffix}]')

        # Step 3. Rearrange the document into the masked string format
        masked_string = self._rearrange_document(prefix, masked_content, suffix)
        # print(f'Rearranged document: {masked_string}')

        # Step 4. Convert the masked string into input and target tensors
        x, y = self._prepare_pair(masked_string)
        ### END YOUR CODE ###
        return x, y

    def _truncate_document(self, document: str) -> str:
        """
        Randomly truncate the document to a length no less than 4 characters,
        and no more than `int(self.block_size*7/8)` characters.

        - IMPORTANT: You are free to decide how to perform this random truncation, but
        make sure that the length is picked _randomly_ (every possible length from 4
        to `int(self.block_size*7/8)` has a chance of being picked) for full credit.
        """
        max_length = min(int(self.block_size * 7 / 8), len(document))
        if len(document) < 4:
            # Handle rare cases where document is shorter than 4 chars
            trunc_length = len(document)
        else:
            trunc_length = random.randint(4, max_length)
        return document[:trunc_length]

    def _break_document(self, document: str) -> list[str]:
        """
        Now, break the (truncated) document into three substrings:

            [prefix] [masked_content] [suffix]

        In other words, choose three strings prefix, masked_content and suffix
            such that prefix + masked_content + suffix = [the truncated document].
        The length of [masked_content] should be random, and 1/4 the length of the
            truncated document on average.

        - IMPORTANT: You are free to decide how to perform this operation, but
        make sure that the length is picked _randomly_ (has a chance of being more or
        less than 1/4 the length of the truncated document) for full credit.
        """

        # masked_length = max(1, int(len(document) / 4))
        # average length is L / 4, with random variation
        avg_len = len(document) / 4.0
        # e.g., uniform between max(1, int(avg_len * 0.5)) and int(avg_len * 1.5) or similar:
        masked_length = random.randint(max(1, int(avg_len * 0.5)), max(1, int(avg_len * 1.5)))


        start = random.randint(0, len(document) - masked_length)
        prefix = document[:start]
        masked_content = document[start:start + masked_length]
        suffix = document[start + masked_length:]
        return prefix, masked_content, suffix

    def _rearrange_document(self, prefix: str, masked_content: str, suffix: str) -> str:
        """
        Rearrange these substrings into the following form:

            [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]

        This resulting string, denoted masked_string, serves as the output example.
        Here MASK_CHAR is the masking character and [pads] is a string of repeated
            PAD_CHAR characters chosen so that the entire string is of length
            self.block_size + 1.
        Intuitively, the string [masked_content] is removed from the document and
            replaced with MASK_CHAR. After the suffix of the string, MASK_CHAR is seen
            again, followed by the removed content, and the padding characters.
        """
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        pads = self.PAD_CHAR * (self.block_size + 1 - len(masked_string))
        return masked_string + pads

    def _prepare_pair(self, masked_string: str) -> Tuple[str, str]:

        """
        We now use masked_string to construct the input and output example pair. To
        do so, simply take the input string to be masked_string[:-1], and the output
        string to be masked_string[1:]. In other words, for each character, the goal
        is to predict the next character in the masked string.

        Making use of the vocabulary that you defined, encode the resulting input
        and output strings as Long tensors and return the resulting data point.
        """
        x = torch.tensor([self.stoi[c] for c in masked_string[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in masked_string[1:]], dtype=torch.long)
        return x, y


# The input-output pairs (x, y) of the NameDataset are of the following form:

# x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

# Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
# optimizing the model to predict the question, "Where was...".

# Note that the NameDataset should take the pretraining_dataset defined in run.py
# as an input. This is to allow the vocab specification of the NameDataset to be
# the same as that of the pretraining dataset.

# You don't need to implement anything in NameDataset.

class NameDataset(Dataset):
    def __init__(self, 
                 pretraining_dataset: CharCorruptionDataset, 
                 data: str):
        
        self.MASK_CHAR = "\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1" # the empty square character, for pad


        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi

        self.block_size = pretraining_dataset.block_size
        # print(f'block_size: {self.block_size}')

        # Convert the input data to a list of strings, ignoring non-ASCII characters.
        # ['Where was Khatchig Mouradian born?\tLebanon', 'Where was Jacob Henry Studer born?\tColumbus', ...]
        self.data: list[str] = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))
        # print(f'self.data {self.data[:5]}')

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):

        # Fetch the sentence at the given index and split it into input and output parts.
        # Sentense example: 'Where was Khatchig Mouradian born?\tLebanon'
        # inp = 'Where was Khatchig Mouradian born?' 
        # oup = 'Lebanon'
        inp, oup = self.data[idx].split('\t')

        # Construct the input sequence x and the target sequence y.

        # x = 'Where was Khatchig Mouradian born?⁇Lebanon⁇'
        # Here we have different question marks:
        #   - '?' is the original question mark in the input;
        #   - '⁇' is the MASK_CHAR used to separate input and output.
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        # print(f'x before padding:\n{x}')
        # print(f'length of x before padding: {len(x)}')

        # Pad the input sequence x to the block size with the PAD_CHAR.
        # This ensures that the input sequence has a fixed length equal to the block size.
        # x = 'Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□'
        # The length of x after padding should be equal to the block size (128).
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        # print(f'x after padding:\n{x}')
        # print(f'length of x after padding: {len(x)}')   

        # Construct the target sequence y.
        # The target sequence y is the same as the input sequence x, but with the input part replaced by PAD_CHARs.
        # y = '□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇'□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
        # The length of y after padding should also be equal to the block size - 1 (127).
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        # print(f'y after padding:\n{y}')
        # print(f'length of y after padding: {len(y)}')

        # Remove the last character from x to match the target sequence length.
        x = x[:-1]
        # print(f'x after removing last char:\n{x}')
        # print(f'length of x after removing last char: {len(x)}')

        # Convert the input and target sequences to tensors of token indices.
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        
        return x, y


# Code below is strictly for your debugging purposes; feel free to modify
# as desired.

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(f"Unknown dataset type in command line args: {args.dataset_type}")
