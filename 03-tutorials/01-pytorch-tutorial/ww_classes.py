import torch
from torch import nn

import itertools

from dataclasses import dataclass

#############################################################
# Data
#############################################################

# Our raw data, which consists of sentences
corpus = [
          "We always come to Paris",
          "The professor is from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara"
         ]

#############################################################
# Config
#############################################################
@dataclass
class WWConfig:
    window_size: int = 2
    embed_dim: int = 25
    hidden_dim: int = 25
    learning_rate: float = 0.01
    batch_size: int = 2
    num_epochs: int = 1000
    freeze_embeddings: bool = False

#############################################################
# Data management
#############################################################
class WWVocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenized_texts):
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        unique_words = sorted(
            set(itertools.chain.from_iterable(tokenized_texts)) - set(special_tokens)
        )
        self.vocab = special_tokens + unique_words

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.token_pad_idx = self.word2idx[self.PAD_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.vocab)

    def word_to_index(self, word):
        return self.word2idx.get(word.lower(), self.unk_idx)

    def index_to_word(self, idx):
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode(self, tokens):
        """Converts a list of word tokens -> list of integer indices."""
        return [self.word_to_index(token) for token in tokens]

    def decode(self, indices):
        """Converts a list of integer indices -> list of word tokens."""
        return [self.index_to_word(idx) for idx in indices]

    def encode_batch(self, batch_of_tokens):
        """Converts a list of token lists -> list of index lists."""
        return [self.encode(tokens) for tokens in batch_of_tokens]


class WWDataset(torch.utils.data.Dataset):
    DEFAULT_LOCATIONS = {"australia", "ankara", "paris", "stanford", "taiwan", "turkey"}

    def __init__(self, sentences, labels=None, locations=None):
        """
        sentences: list of raw string sentences, e.g. ["We always come to Paris", ...]
        labels: optional explicit labels. If None, generated from `locations`.
        """
        # Tokenize sentences
        self.data = [s.lower().split() for s in sentences]

        # Assign labels if not provided
        if labels is not None:
            self.labels = labels
        else:
            locs = locations if locations is not None else self.DEFAULT_LOCATIONS
            self.labels = [[1 if w in locs else 0 for w in sent] for sent in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class WWCollatorBin:
    def __init__(self, vocab: WWVocabulary, config: WWConfig):
        self.vocab = vocab
        self.window_size = config.window_size
        self.pad_token = vocab.PAD_TOKEN
        self.token_pad_idx = vocab.token_pad_idx
        # self.label_ignore_index = -100

    def _pad_window(self, sentence):
        window = [self.pad_token] * self.window_size
        return window + sentence + window

    def __call__(self, batch):
        # Extract sentences and labels from the batch
        sentences, labels = zip(*batch)

        # 1. Record true sentence lengths BEFORE padding (needed for loss normalization)
        lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
        # print(f"Lengths tensor:\n{lengths}")

        # 2. Window-pad sentences with <pad>
        sentences = [self._pad_window(sent) for sent in sentences]

        # 3. Convert tokens to index tensors
        sentences_encoded = self.vocab.encode_batch(sentences)

        # 4. Pad batch sequences to equal lengths
        sentences_tensor = [torch.tensor(sent, dtype=torch.long) for sent in sentences_encoded]
        # print(f"Sentences tensor before padding:\n{sentences_tensor}")

        sentences_padded = torch.nn.utils.rnn.pad_sequence(
            sentences_tensor, 
            batch_first=True, 
            # Pad sentences with the *token pad index*
            padding_value=self.token_pad_idx
        )
        # print(f"Sentences tensor after padding:\n{sentences_padded}")

        labels_tensor = [torch.tensor(lbl, dtype=torch.float) for lbl in labels]
        # print(f"\nLabels tensor before padding:\n{labels_tensor}")
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_tensor, 
            batch_first=True, 
            # Pad labels with 0
            padding_value=self.token_pad_idx
        )
        # print(f"Labels tensor after padding:\n{labels_padded}")

        return sentences_padded, labels_padded, lengths

class WWCollatorMulti:
    def __init__(self, vocab: WWVocabulary, config: WWConfig):
        self.vocab = vocab
        self.window_size = config.window_size
        self.pad_token = vocab.PAD_TOKEN
        self.token_pad_idx = vocab.token_pad_idx
        self.label_ignore_index = -100

    def _pad_window(self, sentence):
        window = [self.pad_token] * self.window_size
        return window + sentence + window

    def __call__(self, batch):
        # Extract sentences and labels from the batch
        sentences, labels = zip(*batch)

        # 1. Record true sentence lengths BEFORE padding (needed for loss normalization)
        lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)

        # 2. Window-pad sentences with <pad>
        sentences = [self._pad_window(sent) for sent in sentences]

        # 3. Convert tokens to index tensors
        sentences_encoded = self.vocab.encode_batch(sentences)

        # 4. Pad batch sequences to equal lengths
        sentences_tensor = [torch.tensor(sent, dtype=torch.long) for sent in sentences_encoded]
        sentences_padded = torch.nn.utils.rnn.pad_sequence(
            sentences_tensor, 
            batch_first=True, 
            # Pad sentences with the *token pad index*
            padding_value=self.token_pad_idx
        )
        # Changed labels tensor dtype from float to long for multi-class classification
        labels_tensor = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_tensor, 
            batch_first=True, 
            # Pad labels with the *label ignore index*
            padding_value=self.label_ignore_index
        )

        return sentences_padded, labels_padded, lengths


def get_data_loader(sentences,
                     dataset: WWDataset,
                     config: WWConfig,
                     vocab: WWVocabulary,
                     shuffle: bool = True,
                     batch_size: int = None,
                     model_type='bin'):


    if model_type == 'bin':
        collator = WWCollatorBin(vocab, config)
    elif model_type == 'multi':
        collator = WWCollatorMulti(vocab, config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    bs = batch_size if batch_size is not None else config.batch_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=shuffle, collate_fn=collator
    )

    return loader

#############################################################
# Model
#############################################################

class WWClassifierBin(nn.Module):

  def __init__(self, config, vocab, num_classes=1, loss_type='bce_with_logits'):
    super().__init__()

    self.window_size = config.window_size
    self.embed_dim = config.embed_dim
    self.hidden_dim = config.hidden_dim
    self.freeze_embeddings = config.freeze_embeddings

    self.vocab = vocab
    self.num_classes = num_classes
    self.loss_type = loss_type

    self.embeds = nn.Embedding(len(vocab), self.embed_dim, padding_idx=self.vocab.token_pad_idx)
    if self.freeze_embeddings:
        self.embeds.weight.requires_grad = False

    # Window size is the number of tokens to the left AND to the right of the target token,
    # so the full window size is 2 * window size + 1 (for the target token itself)
    self.full_window_size = 2 * self.window_size + 1

    self.hidden_layer = nn.Sequential(
        nn.Linear(self.full_window_size * self.embed_dim, self.hidden_dim),
        nn.Tanh()
    )

    self.output_layer = nn.Linear(self.hidden_dim, self.num_classes)

    self.probabilities = nn.Sigmoid()
    

  def forward(self, inputs):

    """
    `unfold()` is PyTorch's built-in tool for creating a 
    **sliding window view** over a tensor without copying memory.

    Its signature:
    ```python
    tensor.unfold(dimension, size, step)
    ```
    - **`dimension`**: which axis to slide along.
    - **`size`**: how wide each window is.
    - **`step`**: how many positions to advance each time (stride).

    Example usage:
    ```python
    x = torch.tensor([10, 20, 30, 40, 50, 60, 70])
    # We want windows of size 3, sliding by 1 step:
    windows = x.unfold(dimension=0, size=3, step=1)
    ```
    We may create 5 sliding windows from this sequence of length 7 with 
    a window size of 1 and the full window size of 3. The general formulas are:
    - Full window size = 2 * window size + 1
    - Adjusted length = sequence length - 2 * window size

    Output:
    ```text
    tensor([[10, 20, 30],    # Window 1 (starts at idx 0)
            [20, 30, 40],    # Window 2 (starts at idx 1)
            [30, 40, 50],    # Window 3 (starts at idx 2)
            [40, 50, 60],    # Window 4 (starts at idx 3)
            [50, 60, 70]])   # Window 5 (starts at idx 4)
    ```
    """

    # Batch contains batch_inputs, batch_labels, batch_lengths.
    # Here we use input = batch_inputs [batch_size, seq_len]
    batch_size, seq_len = inputs.size()

    ##################################################################
    # Step 1. Create sliding windows for each token in the input sequence.
    ##################################################################

    # The adjusted length is the number of valid windows we can create from the sequence.
    # Depends on the sequence length of the current batch.
    adjusted_length = seq_len - 2 * self.window_size
    token_windows = inputs.unfold(1, self.full_window_size, 1) #  [batch_size, adjusted_length, full_window_size]
    # _, adjusted_length, _ = token_windows.size()

    # Good idea to do internal tensor-size sanity checks, at the least in comments!
    assert token_windows.size() == (batch_size, adjusted_length, self.full_window_size)

    ##################################################################
    # Step 2. Embed the sliding windows.
    ##################################################################

    # Embed each token window using the embedding layer.
    embedded_windows = self.embeds(token_windows) # [batch_size, adjusted_length, full_window_size, embed_dim]

    # Flatten the last two dimensions to prepare for the hidden linear layer.
    embedded_windows = embedded_windows.view(
        batch_size, adjusted_length, -1) # [batch_size, adjusted_length, full_window_size * embed_dim]

    ##################################################################
    # Step 3. Apply the hidden linear layer and the output layer.
    ##################################################################

    # Apply the hidden linear layer and activation to the embedded windows.
    out_hidden = self.hidden_layer(embedded_windows) # [batch_size, adjusted_length, hidden_dim]

    # Apply an output layer to the hidden layer's output.
    logits = self.output_layer(out_hidden) # [batch_size, adjusted_length, 1]

    # print(f"Logits:\n{logits.squeeze(-1)}")

    ##################################################################
    # Step 4. Apply the probability function to the logits.
    ##################################################################

    if self.loss_type == 'bce_with_logits':
        # For BCE with logits, we can directly use the logits as output.
        output = logits # [batch_size, adjusted_length, 1]
    elif self.loss_type == 'bce':
        output = self.probabilities(logits) # [batch_size, adjusted_length, 1]
    else:
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    # Reshape the output to remove the last dimension for consistency.
    output = output.view(batch_size, -1) # [batch_size, adjusted_length]

    # print(f"\nOutput:\n{output}")

    return output


class WWClassifierMulti(nn.Module):

  def __init__(self, 
               config, 
               vocab, 
               num_classes=2,
               loss_type='cross_entropy'):
    super().__init__()

    self.window_size = config.window_size
    self.embed_dim = config.embed_dim
    self.hidden_dim = config.hidden_dim
    self.freeze_embeddings = config.freeze_embeddings

    self.vocab = vocab
    self.num_classes = num_classes
    self.loss_type = loss_type

    self.embeds = nn.Embedding(len(vocab), self.embed_dim, padding_idx=self.vocab.pad_idx)
    if self.freeze_embeddings:
        self.embeds.weight.requires_grad = False

    # Window size is the number of tokens to the left AND to the right of the target token,
    # so the full window size is 2 * window size + 1 (for the target token itself)
    self.full_window_size = 2 * self.window_size + 1

    self.hidden_layer = nn.Sequential(
        nn.Linear(self.full_window_size * self.embed_dim, self.hidden_dim),
        nn.Tanh()
    )

    self.output_layer = nn.Linear(self.hidden_dim, self.num_classes)

    self.probabilities = nn.LogSoftmax(dim=-1)
    

  def forward(self, inputs):

    # Batch contains batch_inputs, batch_labels, batch_lengths.
    # Here we use input = batch_inputs [batch_size, seq_len]
    batch_size, seq_len = inputs.size()

    ##################################################################
    # Step 1. Create sliding windows for each token in the input sequence.
    ##################################################################

    # The adjusted length is the number of valid windows we can create from the sequence.
    # Depends on the sequence length of the current batch.
    adjusted_length = seq_len - 2 * self.window_size
    token_windows = inputs.unfold(1, self.full_window_size, 1) #  [batch_size, adjusted_length, full_window_size]
    # _, adjusted_length, _ = token_windows.size()

    # Good idea to do internal tensor-size sanity checks, at the least in comments!
    assert token_windows.size() == (batch_size, adjusted_length, self.full_window_size)

    ##################################################################
    # Step 2. Embed the sliding windows.
    ##################################################################

    # Embed each token window using the embedding layer.
    embedded_windows = self.embeds(token_windows) # [batch_size, adjusted_length, full_window_size, embed_dim]

    # Flatten the last two dimensions to prepare for the hidden linear layer.
    embedded_windows = embedded_windows.view(
        batch_size, adjusted_length, -1) # [batch_size, adjusted_length, full_window_size * embed_dim]

    ##################################################################
    # Step 3. Apply the hidden linear layer and the output layer.
    ##################################################################

    # Apply the hidden linear layer and activation to the embedded windows.
    out_hidden = self.hidden_layer(embedded_windows) # [batch_size, adjusted_length, hidden_dim]

    # Apply an output layer to the hidden layer's output.
    logits = self.output_layer(out_hidden) # [batch_size, adjusted_length, num_classes]

    # print(f"Logits:\n{logits.squeeze(-1)}")

    ##################################################################
    # Step 4. Apply the probability function to the logits.
    ##################################################################

    if self.loss_type == 'cross_entropy':
        output = logits # [batch_size, adjusted_length, num_classes]
    elif self.loss_type == 'nll_loss':
        output = self.probabilities(logits) # [batch_size, adjusted_length, num_classes]
    else:
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    # Permute the output, which is rerquired by CrossEntropyLoss and NLLLossin PyTorch.
    output = output.permute(0, 2, 1) # [batch_size, num_classes, adjusted_length]

    # print(f"\nOutput:\n{output}")

    return output

#############################################################
# Trainer
#############################################################
  
class TrainerBin:

    def __init__(self, config, corpus, dataset, vocab, loss_type='bce_with_logits'):

        # Store the configuration and data for the trainer
        self.config = config
        self.corpus = corpus
        self.dataset = dataset
        self.vocab = vocab

        # Initialize the model and optimizer
        self.model = WWClassifierBin(config, vocab=vocab, loss_type=loss_type)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)

        # Initialize the loss function based on the specified loss type
        if loss_type == 'bce_with_logits':
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif loss_type == 'bce':
            self.loss_fn = torch.nn.BCELoss(reduction='none')
        else:   
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Initialize the data loader for the training dataset
        self.loader = get_data_loader(corpus, dataset, config, vocab)

    # def _loss_function(self, batch_outputs, batch_labels, batch_lengths):
    #     # Calculate the loss for the whole batch
    #     loss = self.loss_fn(batch_outputs, batch_labels.float())

    #     # Rescale the loss. Remember that we have used lengths to store the
    #     # number of words in each training example
    #     loss = loss / batch_lengths.sum().float()

    #     return loss

    def _loss_function(self, batch_outputs, batch_labels, batch_lengths):
        # One loss value per token position.
        per_token_loss = self.loss_fn(batch_outputs, batch_labels.float())

        positions = torch.arange(batch_outputs.size(1), device=batch_outputs.device)
        lengths = batch_lengths.to(batch_outputs.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        if not mask.any():
            raise ValueError("Cannot compute loss for a batch with no valid tokens")

        return (per_token_loss * mask).sum() / mask.sum()

    def _train_epoch(self):

        # Keep track of the total loss for the batch
        total_loss = 0
        for batch_inputs, batch_labels, batch_lengths in self.loader:
            # Clear the gradients
            self.optimizer.zero_grad()
            # Run a forward pass
            outputs = self.model.forward(batch_inputs)
            # Compute the batch loss
            loss = self._loss_function(outputs, batch_labels, batch_lengths)
            # Calculate the gradients
            loss.backward()
            # Update the parameteres
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss

    # Function containing our main training loop
    def train(self):

        num_epochs = self.config.num_epochs
        # Iterate through each epoch and call our train_epoch function
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch()
            if epoch % 100 == 0: print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")