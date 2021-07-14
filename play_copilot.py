# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed
set_seed(42)

from pathlib import Path

from tqdm import tqdm
import numpy as np
import random
import json
from pprint import pprint
import torch
import torch.nn as nn
from torch.nn import functional as F

python_files_train = list(Path('play_copilot_data/python/final/jsonl/train/').glob('*.jsonl')) \
                     + list(Path('play_copilot_data/python/final/jsonl/valid/').glob('*.jsonl'))
python_files_test = list(Path('play_copilot_data/python/final/jsonl/test/').glob('*.jsonl'))

def load_files(json_files):
    """Load raw data into list for training."""
    data = []
    for f in tqdm(json_files, desc='Loading code into memory'):
        with open(f, 'r') as fp:
            file_data = fp.readlines()
        data += [
            str(json.loads(line)['code'].encode('ascii', 'ignore'))  # Drop non-ascii characters
            for line in file_data if len(line) > 100
        ]
    return data

train_data = load_files(python_files_train)
test_data = load_files(python_files_test)

print(f"\n{len(train_data):,} training and {len(test_data):,} testing functions found.")

from torch.utils.data import Dataset


class CopilotDataset(Dataset):
    def __init__(self, data, block_size):
        self.vocab_size = 128  # Use all ascii characters
        self.block_size = block_size
        self.data = data
        random.shuffle(self.data)

        self.null = '\x00'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        chunklen = len(chunk)

        final_idx = np.random.randint(
            self.block_size // 16,
            chunklen + self.block_size // 16,
        )

        if final_idx > chunklen:
            # Pad with null if selection is overrun
            dix = chunk + self.null * (final_idx - chunklen)
            first_idx = final_idx - self.block_size - 1
            dix = dix[first_idx:final_idx]

            # If chunk is still too short, add leading spaces
            if len(dix) < self.block_size:
                dix = self.null * (self.block_size - len(dix) + 1) + dix

        elif final_idx <= self.block_size:
            # Pad with leading spaces if selection is too short
            dix = chunk[:final_idx + 1]
            dix = self.null * (self.block_size - len(dix) + 1) + dix

        elif final_idx > self.block_size:
            first_idx = final_idx - self.block_size - 1
            dix = chunk[first_idx:final_idx]

        dix = [ord(s) for s in dix]

        if len(dix) != self.block_size + 1:
            print(f"Chunklen: {chunklen:,}")
            print(f"Final idx = {final_idx:,}")
            print(f"Dix len: {len(dix)}")
            raise Exception

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y

block_size = 512
train_dataset = CopilotDataset(train_data, block_size)
test_dataset = CopilotDataset(test_data, block_size)

from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

from mingpt.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=20,
    batch_size=8,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512*20,
    final_tokens=2*len(train_dataset)*block_size,
    num_workers=4,
    ckpt_path='play_copilot_checkpoint.pt',
)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()

from mingpt.utils import sample

def sample_model(context, n=500, temperature=1):
    x = torch.tensor([ord(s) for s in context], dtype=torch.long)[None, ...].to(trainer.device)
    y = sample(model, x, 500, temperature=1, sample=True, top_k=10)[0]
    completion = ''.join([chr(i) for i in y])
    completion = completion.replace('\\n', '\n')

    return completion

print(sample_model('def multiply(x, y):\n    """Multiply two numbers together."""\n'))
print(sample_model('def add(a, b):'))
