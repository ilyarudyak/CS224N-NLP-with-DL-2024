import torch
import torch.nn as nn
from torch.nn import functional as F

import time
start = time.time()

# hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

# Fast Mac test parameters:
batch_size = 32        # was 64
block_size = 64       # was 256
max_iters = 1000       # was 5000
eval_interval = 250    # was 500
learning_rate = 3e-4
eval_iters = 50        # was 200
n_embd = 128           # was 384
n_head = 4             # was 6
n_layer = 4            # was 6
dropout = 0.2

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # Step 1: compute keys, queries, and values for self-attention
        B,T,C = x.shape
        # print(f"Input shape: {x.shape}")
        k = self.key(x)   # (B,T,head_size)
        # print(f"Key shape: {k.shape}")
        q = self.query(x) # (B,T,head_size)
        # print(f"Query shape: {q.shape}")

        # compute attention scores ("affinities")
        # print(torch.allclose(k.transpose(-2,-1), k.transpose(1, 2)))

        # Step 2: compute scaled dot-product attention (divide by the square root of the head_size)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # print(f"Attention weights shape: {wei.shape}")

        # Step 3: apply the causal mask to prevent attending to future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        # print(f"Value shape: {v.shape}")

        # Step 4: compute the weighted sum of the values using the attention weights§
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        # print(f"Output shape: {out.shape}")

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        print(f"Initializing MultiHeadAttention with num_heads: {num_heads}, head_size: {head_size}")
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        print(f"self.head length: {len(self.heads)}")
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # out = torch.cat([h(x) for h in self.heads], dim=-1)
        outs = []
        for h in self.heads:
            out = h(x)
            print(f"Head output shape: {out.shape}")
            outs.append(out)

        out = torch.cat(outs, dim=-1)
        print(f"Concatenated output shape: {out.shape}")
        out = self.dropout(self.proj(out))
        print(f"Projected output shape: {out.shape}")
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # (1) Token and position embeddings for the input sequence
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # (2) Transformer blocks for processing the embeddings: number of blocks == n_layer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # (3) Final layer norm and language modeling head (after ALL transformer blocks)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def train_model(is_generate=False):

    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # Introduce a step variable for logging purposes
        step = iter + 1 # CHANGED: introduce a step variable for logging purposes

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets (covers 250, 500, 750, 1000)
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Print final validation perplexity
    losses = estimate_loss() # CHANGED: get the final losses before printing final validation perplexity
    perplexity = torch.exp(losses['val'].detach().clone())
    print(f"Final validation perplexity: {perplexity:.4f}")

    # Report training time
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
    
    if is_generate:
        # generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
        #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

def load_csa_weights_into_mha(mha, weights_path='csa_weights.pt'):
    """Maps the batched weights from CausalSelfAttention into individual Heads of MultiHeadAttention."""
    csa_state = torch.load(weights_path, map_location='cpu')
    hs = n_embd // n_head

    with torch.no_grad():
        for i, h in enumerate(mha.heads):
            # Slice the weight and bias corresponding to head i
            h.key.weight.copy_(csa_state['key.weight'][i*hs : (i+1)*hs, :])
            h.key.bias.copy_(csa_state['key.bias'][i*hs : (i+1)*hs])

            h.query.weight.copy_(csa_state['query.weight'][i*hs : (i+1)*hs, :])
            h.query.bias.copy_(csa_state['query.bias'][i*hs : (i+1)*hs])

            h.value.weight.copy_(csa_state['value.weight'][i*hs : (i+1)*hs, :])
            h.value.bias.copy_(csa_state['value.bias'][i*hs : (i+1)*hs])

        # Copy projection weights and bias
        mha.proj.weight.copy_(csa_state['proj.weight'])
        mha.proj.bias.copy_(csa_state['proj.bias'])
    
    print("Successfully copied CausalSelfAttention weights into MultiHeadAttention!")

if __name__ == "__main__":

    # --- Quick test block inside gpt.py ---
    torch.manual_seed(0)
    print("Running quick test block...")
    print(f"\nblock size: {block_size}, n_embd: {n_embd}, n_head: {n_head}")
    dummy_x = torch.randn(2, block_size, n_embd, device='cpu')
    print(f"dummy_x shape: {dummy_x.shape}")  # (2, block_size, n_embd)
    print(f"dummy_x first 5 elements: {dummy_x[0, 0, :5]}")

    # 1. Test single Head
    # print(f"\n===Testing single ===")
    # head_size = n_embd // n_head
    # print(f"Head size: {head_size}")
    # head = Head(head_size=head_size).to(device)
    # out_head = head(dummy_x)
    # print(f"Head output shape: {out_head.shape}")  # (2, block_size, head_size)

    # 2. Test MultiHeadAttention
    # print(f"\n===Testing MultiHeadAttention ===")
    # torch.manual_seed(0)
    # mha = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head).to('cpu')
    # out_mha = mha(dummy_x)
    # print(f"MHA output shape:  {out_mha.shape}")   # (2, block_size, n_embd)
    # print(f"MHA output first 5 elements: {out_mha[0, 0, :5]}")  

    # 3. Test MultiHeadAttention with loaded CSA weights
    print(f"===Testing MultiHeadAttention with loaded CSA weights...===")
    torch.manual_seed(0)
    dummy_x = torch.randn(2, block_size, n_embd, device='cpu')
    print(f"dummy_x shape before MHA: {dummy_x.shape}")  # (2, block_size, n_embd)
    print(f"dummy_x first 5 elements before MHA: {dummy_x[0, 0, :5]}")
    

    mha = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head).to('cpu')
    
    # Load weights from the saved file
    load_csa_weights_into_mha(mha, '../csa_weights.pt')
    
    # Put in eval mode to disable dropout
    mha.eval()

    out_mha = mha(dummy_x)
    print(f"MHA output first 5 elements: {out_mha[0, 0, :5]}")




