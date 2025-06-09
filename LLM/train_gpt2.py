"""
Personal Notes:

1. There's DDP overhead with a single GPU:

    (vp1) omer@verapulse build-nanogpt % python train_gpt2.py
    DDP is not enabled using device cuda
    -> total desired batch size:  524288
    -> calculated gradient accumulation steps: 32
    loaded 338025 tokens
    1 epoch = 20
    num decayed parameter tensors: 50, with 124,354,560 parameters
    num non-decayed parameter tensors: 98, with 121,344, parameters
    using fused AdamW: True
    step    0| loss: 10.938572 | lr: 6.0000e-05 | norm: 27.0145  | dt: 3663.01ms | tok/sec: 143130.26
    step    1| loss: 9.649529 | lr: 1.2000e-04 | norm: 9.5134  | dt: 2713.86ms | tok/sec: 193189.05
    step    2| loss: 9.224648 | lr: 1.8000e-04 | norm: 5.6872  | dt: 2713.59ms | tok/sec: 193208.23
    step    3| loss: 9.812168 | lr: 2.4000e-04 | norm: 8.2264  | dt: 2713.56ms | tok/sec: 193210.12
    step    4| loss: 9.190122 | lr: 3.0000e-04 | norm: 4.3002  | dt: 2712.87ms | tok/sec: 193259.68
    step    5| loss: 8.676892 | lr: 3.6000e-04 | norm: 3.6255  | dt: 2713.46ms | tok/sec: 193217.59
    step    6| loss: 8.295106 | lr: 4.2000e-04 | norm: 1.9540  | dt: 2714.61ms | tok/sec: 193135.76
    step    7| loss: 8.066936 | lr: 4.8000e-04 | norm: 2.8287  | dt: 2714.53ms | tok/sec: 193141.27
    step    8| loss: 7.713205 | lr: 5.4000e-04 | norm: 1.9201  | dt: 2714.45ms | tok/sec: 193147.21
    step    9| loss: 7.346015 | lr: 6.0000e-04 | norm: 1.8049  | dt: 2714.71ms | tok/sec: 193128.73
    step   10| loss: 7.028761 | lr: 6.0000e-04 | norm: 1.8351  | dt: 2715.57ms | tok/sec: 193067.10

    (vp1) omer@verapulse build-nanogpt % torchrun --standalone --nproc_per_node=1 train_gpt2.py
    -> total desired batch size:  524288
    -> calculated gradient accumulation steps: 32
    loaded 338025 tokens
    1 epoch = 20
    num decayed parameter tensors: 50, with 124,354,560 parameters
    num non-decayed parameter tensors: 98, with 121,344, parameters
    using fused AdamW: True
    step    0| loss: 10.938571 | lr: 6.0000e-05 | norm: 27.0146  | dt: 6437.43ms | tok/sec: 81443.69
    step    1| loss: 9.649517 | lr: 1.2000e-04 | norm: 9.5133  | dt: 2815.50ms | tok/sec: 186214.98
    step    2| loss: 9.224654 | lr: 1.8000e-04 | norm: 5.6877  | dt: 2815.18ms | tok/sec: 186236.17
    step    3| loss: 9.812183 | lr: 2.4000e-04 | norm: 8.2261  | dt: 2814.84ms | tok/sec: 186258.35
    step    4| loss: 9.190122 | lr: 3.0000e-04 | norm: 4.3001  | dt: 2814.42ms | tok/sec: 186286.45
    step    5| loss: 8.676889 | lr: 3.6000e-04 | norm: 3.6255  | dt: 2814.15ms | tok/sec: 186304.14
    step    6| loss: 8.295108 | lr: 4.2000e-04 | norm: 1.9539  | dt: 2814.67ms | tok/sec: 186269.52
    step    7| loss: 8.066954 | lr: 4.8000e-04 | norm: 2.8294  | dt: 2814.24ms | tok/sec: 186298.46
    step    8| loss: 7.713212 | lr: 5.4000e-04 | norm: 1.9197  | dt: 2810.93ms | tok/sec: 186517.52
    step    9| loss: 7.346024 | lr: 6.0000e-04 | norm: 1.8048  | dt: 2811.33ms | tok/sec: 186491.01
    step   10| loss: 7.028761 | lr: 6.0000e-04 | norm: 1.8352  | dt: 2813.80ms | tok/sec: 186327.22

2. Maximum microbatch for 5090 is 40 * 1024 -> which is around ~50K tokens per second

"""


from dataclasses import dataclass
import math
import time
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim.optimizer
import inspect
import numpy as np

# config parameters
USE_COMPILE = True


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of layers
    n_embd: int = 768 # embedding dimension
    

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/ HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_eembd)
        # calculate query, key, values for all heads in batch and move head forward
        # to be the batch dim
        # nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=X=768 channels in Transformer

        # This contains all the heads for all the query-key-value multiplied together
        qkv = self.c_attn(x)
        # Split them between tensors
        q, k, v = qkv.split(self.n_embd, dim=2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # flash_attention_alternative start: this is the place where flash attention is implemented
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # flash_attention_alternative end

        # flash_attention start
        y = F.scaled_dot_product_attention(q, k, v, is_causal= True)
        # flash_attention end
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y
    
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # TODO: there's a paper on it
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weights from pretrained gpt: {model_type}')
        
        # n_layer, n_head, and n-embd are determineed from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600)  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch init GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        
        # init a huggingface 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use a 'Conv1D' module, but we only want to use a vanilla linear 
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose copy for transposed params
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters requiring grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is at least 2D will be weight decayed, otherwise no.
        # i.e. all weight tnesors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [ p for n, p in param_dict.items() if p.dim() >= 2 ]
        nodecay_params = [ p for n, p in param_dict.items() if p.dim() < 2 ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params':nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,}, parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ----------------------------------------------------------------------------
import tiktoken


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


# TODO: Karpathy's video 3.18:26 to start modification


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # at init load tokens from disk and store them in memory
        SH_DIR = '/home/omer/data/tiny_shakespeare/input.txt'
        with open(SH_DIR, 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)}")
        
        # state 
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T) 
        # advance the position in hte tensor
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y
# ----------------------------------------------------
# simple launch: 
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel)
# torchrun command sets up special environment variables

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run? 
if ddp:
    assert torch.cuda.is_available(), "you must have cuda to run DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # each gpu will have their own rank
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # multi-node setting
    ddp_world_size = int(os.environ['WORLD_SIZE']) 
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device=device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f'DDP is not enabled using device {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2**19, ~0.5M in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "tot batch size is divisible by B*T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'-> total desired batch size:  {total_batch_size}')
    print(f'-> calculated gradient accumulation steps: {grad_accum_steps}')

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high')

# create model
#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)) # this is for random initialization
#model.eval()
model.to(device)

if USE_COMPILE: # Karpathy says use_compile may fail with HellaSwag
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
raw_model = model.module if ddp else model  #  to use with DDP

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

    
# optimize !
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward()
        # instead of a SUM we want MEAN. Scale the loss here for that
        loss = loss / grad_accum_steps    
        loss_accum += loss.detach()
        if (micro_step != grad_accum_steps - 1) and ddp: # Karpathy has a naughty way, put I'd prefer safety over it:)
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for gpu to finish the work
    t1 = time.time()
    dt = (t1 - t0)# time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec =  tokens_processed / dt
    if master_process:
        print(f"step {step:4d}| loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f}  | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
if ddp:
    destroy_process_group()

exit()

# prefix tokens
num_return_sequences = 5
max_length = 30


tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # TODO: deep dive to this
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        # select a token from the top=k probabilities 
        # note: multinomal does not demands the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
