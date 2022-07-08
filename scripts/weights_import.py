"""
This script runs inference of GPT-2, both minGPT and huggingface/transformers
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import fire

from mingpt.model import GPT
from mingpt.utils import sample

# -----------------------------------------------------------------------------

def run(
    model_type = 'gpt2',
    prompt = "Hello, my dog is a little",
    num_samples = 5,
    steps = 20,
    do_sample = True,
    device = 'cuda',
    use_mingpt = True, # use mingpt or huggingface/transformers model
):

    # instantiate the model
    if use_mingpt:
        model = GPT.from_pretrained(model_type)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_type)
        model.config.pad_token_id = model.config.eos_token_id # suppress a warning

    # ship model to device and set to eval mode
    model.to(device)
    model.eval()

    # tokenize the input prompt into integer input sequence
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '': # to create unconditional samples we feed in the special start token
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']

    # forward the model
    logits = model(x)

    # draw some samples
    for _ in range(num_samples):
        if use_mingpt:
            y = sample(model=model, x=x, steps=steps, sample=do_sample, top_k=40)[0]
        else:
            y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
        out = tokenizer.decode(y.cpu().squeeze())
        print('-'*80)
        print(out)

if __name__ == '__main__':
    fire.Fire(run)
