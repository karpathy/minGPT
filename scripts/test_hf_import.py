import torch
import torch.nn as nn
from mingpt.model import GPT
from mingpt.utils import sample
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import numpy as np
from torch.nn import functional as F

model_type = 'gpt2'

# Initialise HF GPT-2 with pretrained weights
gpt2_hf = GPT2LMHeadModel.from_pretrained(model_type)
gpt2_hf_sd = gpt2_hf.state_dict()

# Initialise minGPT model
conf = GPT.get_default_config()
conf.name = model_type
conf.vocab_size = 50257
conf.block_size = 1024
gpt2 = GPT(conf)
gpt2_sd = gpt2.state_dict()

keys = [k for k in gpt2_hf_sd if not k.endswith('attn.masked_bias')]

# all the keys equal
assert len(keys) == len(gpt2_sd)
for k in keys:
    v1 = gpt2_hf_sd[k]
    v2 = gpt2_sd[k]
    assert v1.shape == v2.shape

# now copy over the checkpoint parameters
for k in keys:
    gpt2_sd[k].copy_(gpt2_hf_sd[k])

# test the forward pass
gpt2.to('cuda')
gpt2_hf.to('cuda')
gpt2.eval()
gpt2_hf.eval()

tokenizer = GPT2Tokenizer.from_pretrained(model_type)
text = "I am a good sampler because"
encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
x = encoded_input['input_ids']

logits1, loss = gpt2(x)
out = gpt2_hf(x)
logits2 = out.logits
assert torch.allclose(logits1, logits2)

# sample from migpt
y = sample(model=gpt2, x=encoded_input['input_ids'], steps=50, temperature=0.9, sample=True, top_k=40)[0]
out = tokenizer.decode(y.cpu().squeeze())
print(out)

import code; code.interact(local=locals())
