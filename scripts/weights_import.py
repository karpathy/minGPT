"""
This script will import the actual weights released by OpenAI,
with the help of code from huggingface/transformers. It also
that the forward pass matches exactly.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import fire

from mingpt.model import GPT
from mingpt.utils import sample

# -----------------------------------------------------------------------------

def get_pretrained(model_type='gpt2'):
    """
    model_type is one of gpt2|gpt2-medium|gpt2-large|gpt2-xl
    returns an initialized GPT class
    """

    # init a mingpt model with the right hyperparams
    conf = GPT.get_default_config()
    conf.name = model_type
    conf.vocab_size = 50257 # openai's model vocabulary
    conf.block_size = 1024  # openai's model block_size
    model = GPT(conf)
    sd = model.state_dict()

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
    # this means that we have to transpose these weights when we import them
    assert len(keys) == len(sd)
    for k in keys:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model

# -----------------------------------------------------------------------------

def run(
    model_type = 'gpt2',
    prompt = "Hello, my dog is a little",
    num_samples = 5,
    do_sample = True,
    device = 'cuda',
):

    # create both a minGPT model and a huggingface model
    model = get_pretrained(model_type) # init a minGPT model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type) # init a HF model too

    # ship both to gpu
    model.to(device)
    model_hf.to(device)
    # set both to eval mode
    model.eval()
    model_hf.eval()

    # tokenize an input prompt
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model_hf.config.pad_token_id = model_hf.config.eos_token_id # suppress a warning
    if prompt == '': # to create unconditional samples we feed in the special start token
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']

    # ensure the logits match exactly
    logits1, loss = model(x)
    logits2 = model_hf(x).logits
    assert torch.allclose(logits1, logits2)

    # draw some samples from the HuggingFace model
    print('-'*80)
    print('huggingface samples')
    print('-'*80)
    for _ in range(num_samples):
        y = model_hf.generate(x, max_new_tokens=20, do_sample=do_sample, top_k=40)
        out = tokenizer.decode(y.cpu().squeeze())
        print('-'*80)
        print(out)

    # draw some samples from mingpt model. TODO I don't think it's correct yet
    print('-'*80)
    print('mingpt samples')
    print('-'*80)
    for _ in range(num_samples):
        y = sample(model=model, x=x, steps=20, sample=do_sample, top_k=40)[0]
        out = tokenizer.decode(y.cpu().squeeze())
        print('-'*80)
        print(out)

if __name__ == '__main__':
    fire.Fire(run)
