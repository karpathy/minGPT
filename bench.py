"""
Temporary benchmarking script while integrating Lightning, will remove before merge to master
"""

import os
import time
import math
import logging
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn

from mingpt.model import GPT
from mingpt.lr_decay import WarmupCosineLearningRateDecay
from mingpt.utils import sample

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# -----------------------------------------------------------------------------
import os
if int(os.environ.get('USE_LIGHTNING', 0)):
    logging.info("USING LIGHTNING!!")
    import pytorch_lightning as pl
else:
    import mingpt.fake_lightning as pl
    logging.info("using our humble trainer")
# -----------------------------------------------------------------------------

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        logging.info('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / self.block_size)

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)
        return dix[:-1], dix[1:]

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_workers', type=int, default=0, help="number of workers for dataloading")
args = parser.parse_args()
print(args)

cudnn.benchmark = True

logging.info("loading the dataset")
text = open('input.txt', 'r').read()
block_size = 128
train_dataset = CharDataset(text, block_size)

logging.info("creating the model")
model = GPT(train_dataset.vocab_size, train_dataset.block_size,
            n_layer=4, n_head=4, n_embd=128)

logging.info("creating the dataloaders")
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=0)

t0 = time.time()
logging.info("training...")
nepochs = 5
iter_tokens = batch_size * block_size # number of tokens backpropped in one iteration
epoch_tokens = math.ceil(len(train_dataset) / batch_size) * iter_tokens
lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4, warmup_tokens=epoch_tokens//4, final_tokens=nepochs*epoch_tokens)

trainer = pl.Trainer(gpus=1, max_epochs=nepochs, gradient_clip_val=1.0, callbacks=[lr_decay])
trainer.fit(model, train_loader)
t1 = time.time()
logging.info("%d epochs took %fs, or %fs/epoch", nepochs, t1 - t0, (t1-t0)/nepochs)

context = "O God, O God!"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
y = sample(model, x, 100, temperature=1.0, sample=True, top_k=None)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
