"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from mingpt.dataset_v2 import CharDataset, CharTokenizer, make_datasets
from mingpt.model_v2 import GPT, GPTConfig
from mingpt.utils import sample

logger = logging.getLogger("mingpt.trainer_v2")

class TrainerConfig:

    # (1) Main training parameters
    max_epochs = 2
    batch_size = 64
    learning_rate = 3e-4

    # (2) Optimization parameters
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights

    # (3) Learning rate decay parameters
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)

    # (4) Checkpoint settings
    ckpt_path = "checkpoints/ckpt.pt"

    # (5) GPU Settings
    pin_memory = True
    num_workers = 0 
    use_bf16 = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class TrainerConfigA100(TrainerConfig):

    # (1) Main training parameters (used in play_char.ipynb)
    max_epochs = 2
    # The run from play_char uses 512; fits easily on a 40/80 GB A100
    batch_size = 512
    # The run from play_char uses 6e-4; we need to increase it for larger batch sizes
    learning_rate = 6e-4

    # (3) Learning rate decay parameters (used in play_char.ipynb)
    lr_decay = True
    warmup_tokens = 512 * 20 * 128
    # We set final_tokens later on since it is not a fixed value
    # final_tokens=2*len(train_dataset)*block_size

    # (5) GPU Settings
    num_workers = 4

def make_loaders(train_dataset, val_dataset, config):
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,          # every training batch has the same size
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,   # keep workers alive between epochs
    )
    logger.info(f"Created train loader with {len(train_loader)} batches")
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,       # order doesn't matter for evaluation
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0,
        )
        logger.info(f"Created val loader with {len(val_loader)} batches")
    return train_loader, val_loader

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        train_loader, val_loader = make_loaders(train_dataset, val_dataset, config)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # take over whatever gpus are on the system
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def save_checkpoint(self):
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(self.model.state_dict(), self.config.ckpt_path)

    def _lr_at(self, tokens):
        """Learning rate after `tokens` training tokens (warmup, then cosine decay)."""
        c = self.config
        if not c.lr_decay:
            return c.learning_rate
        if tokens < c.warmup_tokens:
            mult = tokens / max(1, c.warmup_tokens)
        else:
            progress = (tokens - c.warmup_tokens) / max(1, c.final_tokens - c.warmup_tokens)
            mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return c.learning_rate * mult

    def _train_epoch(self, optimizer, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader)
        for it, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            _, loss = self.model(x, y)
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            optimizer.step()

            self.tokens += y.numel()
            lr = self._lr_at(self.tokens)
            for group in optimizer.param_groups:
                group['lr'] = lr

            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        losses = []
        for x, y in self.val_loader:
            _, loss = self.model(x.to(self.device), y.to(self.device))
            losses.append(loss.item())
        return float(np.mean(losses))

    def train(self):

        # Configure the optimizer
        optimizer = self.model.configure_optimizers(self.config)

        # Initialize token counter and best validation loss
        self.tokens = 0
        best_loss = float('inf')

        for epoch in range(self.config.max_epochs):

            # Train for one epoch
            self._train_epoch(optimizer, epoch)

            # Check if validation loss improved for saving checkpoint
            improved = True   
            if self.val_loader is not None:
                val_loss = self._evaluate()
                logger.info(f"val loss: {val_loss:.4f}")
                improved = val_loss < best_loss
                best_loss = min(best_loss, val_loss)

            if self.config.ckpt_path is not None and improved:
                self.save_checkpoint()

def sample_shakespeare(tokenizer, model, trainer, context="O God, O God!"):

    x = torch.tensor([tokenizer.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([tokenizer.itos[int(i)] for i in y])
    print(completion)

def train_on_toy_dataset(context="O God, O God!"):

    # Load text 
    text = open('input.txt', 'r').read()

    # Create tokenizer
    tokenizer = CharTokenizer(text)

    # Split into train and validation sets
    train_ds_toy, val_ds_toy = make_datasets(
        text=text[:128*10], 
        tokenizer=tokenizer,
        val_fraction=0.2
        )

    # Create configuration for the model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=train_ds_toy.block_size,
    )

    # Create the model
    model = GPT(model_config)

    # Create trainer config
    trainer_config = TrainerConfig(
        lr_decay=False
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds_toy,
        val_dataset=val_ds_toy,
        config=trainer_config
        )

    # Start training
    trainer.train()

    # Sample from the trained model
    print("\n\n===Sampling from the trained model:===")
    sample_shakespeare(tokenizer, model, trainer, context)

def train_on_full_dataset_a100(context="O God, O God!"):

    # Load text 
    text = open('input.txt', 'r').read()

    # Create tokenizer
    tokenizer = CharTokenizer(text)

    # Split into train and validation sets
    train_ds_toy, val_ds_toy = make_datasets(
        text=text, # Use the FULL dataset for training
        tokenizer=tokenizer,
        )

    # Create configuration for the model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=train_ds_toy.block_size,
    )

    # Create the model
    model = GPT(model_config)

    # Create trainer config
    trainer_config = TrainerConfig(
        lr_decay=False
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds_toy,
        val_dataset=val_ds_toy,
        config=trainer_config
        )

    # Start training
    trainer.train()

    # Sample from the trained model
    print("\n\n===Sampling from the trained model:===")
    sample_shakespeare(tokenizer, model, trainer, context)
