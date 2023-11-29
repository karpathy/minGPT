import os
import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import GPT2Tokenizer
import json

import torch
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.ul_model import UL_GPT
from mingpt.trainer import Trainer
from mingpt.ul_trainer import UL_Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = UL_Dataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 
    C.trainer.max_iters = 1001
    C.trainer.batch_size = 4

    return C


class UL_Dataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 64
        return C

    def __init__(self, file_path, block_size):
        super().__init__()
        new_tokens = [ "<|R|>", "<|S|>", "<|X|>", "<|SEN|>", "<|BEGIN|>"]
        self.token_mapping = {}
        self.tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for token in new_tokens:
            self.tokenizer.add_tokens(token)
            self.token_mapping[token] = self.tokenizer(token)['input_ids'][0]
        self.token_mapping[" "] = self.tokenizer(" ")['input_ids'][0]
        self.token_mapping[self.tokenizer.eos_token] = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        self.file_path = os.path.normpath(file_path)
        self.block_size = block_size
        self.jsonlines = []
        with open(self.file_path) as reader:
            for json_line in reader:
                text = json.loads(json_line)["text"]
                if text:
                    self.jsonlines.append(self.tokenizer(text, padding='max_length', truncation=True, max_length=self.block_size - 1)['input_ids'])
        

    def __len__(self):
        return len(self.jsonlines)

    def __getitem__(self, idx):
        rand = np.random.randint(0, 2)
        
        if rand == 0:
            x, y = self.get_r_noise(idx)
        elif rand == 1:
            x, y = self.get_x_noise(idx)
        elif rand == 2:
            x, y = self.get_s_noise(idx)

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
    
    def get_r_noise(self, idx):
        x, y = self.non_con_corruption(idx, 0.15, 3)
        x = [self.token_mapping["<|R|>"]] + x

        return x,y

    def get_x_noise(self, idx):
        x, y = self.non_con_corruption(idx, 0.5, 3)
        x = [self.token_mapping["<|X|>"]] + x
        
        return x,y

    def get_s_noise(self, idx):
        x, y = self.con_corruption(idx, 0.5)
        x = [self.token_mapping["<|S|>"]] + x

        return x,y

    def non_con_corruption(self, idx, corruption_percent, mean_corruption):
        token = self.jsonlines[idx].copy()
        target = [self.token_mapping["<|BEGIN|>"]]
        true_length = len(token) - token.count(50256)
        num_spans = math.floor(math.ceil(true_length * corruption_percent) / mean_corruption)
        spans = np.random.normal(3, 1, num_spans)
        spans = [round(i) for i in spans]
        corruption_area_length = math.floor(true_length/num_spans)
        for i, span in enumerate(spans):
            if span >= corruption_area_length: 
                span = corruption_area_length - 1
            r_idx = (i * corruption_area_length) +  np.random.randint(0, corruption_area_length - span)
            for x in range(r_idx, r_idx + span):
                target.append(token[x])
                token[x] = self.token_mapping["<|SEN|>"]
            if i < len(spans) - 1:
                target.append(self.token_mapping[" "])
            else:
                target.append(self.token_mapping[self.tokenizer.eos_token])

        return token, target
    
    def con_corruption(self, idx, corruption_percent):
        assert corruption_percent > 0.0
        assert corruption_percent < 1.0
        token = self.jsonlines[idx].copy()
        target = [self.token_mapping["<|BEGIN|>"]]
        true_length = len(token) - token.count(50256)
        corruption_span_length = math.floor(true_length * corruption_percent)
        r_idx = np.random.randint(0, true_length - corruption_span_length)
        for x in range(r_idx, r_idx + corruption_span_length):
            target.append(token[x])
            token[x] = self.token_mapping["<|SEN|>"]
        target.append(self.token_mapping[self.tokenizer.eos_token])

        return token, target
    
    def get_vocab_size(self):
        return len(self.tokenizer)
    
    def get_block_size(self):
        return self.block_size
    
    def text_to_token(self, str):
        return [self.tokenizer(str, padding='max_length', truncation=True, max_length=self.block_size)['input_ids']]
    
    def token_to_text(self, tokens):
        return self.tokenizer.decode(tokens)
            

if __name__ == "__main__":
    file_path = os.path.join(Path().cwd(), "test_pile_file.jsonl")
    

     # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    train_dataset = UL_Dataset(file_path, 100)

    print(train_dataset.__getitem__(0))

    
    
    iter = []
    loss = []

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = UL_GPT(config.model)

    # construct the trainer object
    trainer = UL_Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
            loss.append(trainer.loss.item())
            iter.append(trainer.iter_num)

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            print("saving model")
            #ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            #torch.save(model.state_dict(), ckpt_path)
            model.save('my_model.pth')
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()


    plt.plot(iter, loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.savefig("Loss.png")
    