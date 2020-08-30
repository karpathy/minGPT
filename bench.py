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

torch.backends.cudnn.benchmark = True # autotune kernels

# -----------------------------------------------------------------------------
import os
if int(os.environ.get('USE_LIGHTNING', 0)):
    logging.info("USING LIGHTNING!!")
    import pytorch_lightning as pl
else:
    import mingpt.fake_lightning as pl
    logging.info("using our humble trainer")
# -----------------------------------------------------------------------------

class Text8Dataset(Dataset):
    """
    e.g. Text8 dataset is often used: http://mattmahoney.net/dc/textdata.html
    Vocabulary is lowercase English characters and space for total of 27.
    Training data: First 90M characters.
    Validation data: First 5M characters out of the last 10M characters.
    Testing data: Last 5M characters.
    """

    def __init__(self, data_path, block_size, crop=None, override_vocab=None):

        # load the data and crop it appropriately
        with open(data_path, 'r') as f:
            if crop is None:
                data = f.read()
            else:
                f.seek(crop[0])
                data = f.read(crop[1])

        # build a vocabulary from data or inherit it
        vocab = sorted(list(set(data))) if override_vocab is None else override_vocab
        data_size, vocab_size = len(data), len(vocab)
        logging.info('data of crop %s has %d characters, vocab of size %d.' % (str(crop), data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # attempt to fetch a chunk of (block_size + 1) items, but (block_size) will work too
        chunk = self.data[idx*self.block_size : min(len(self.data), (idx+1)*self.block_size + 1)]
        # map the string into a sequence of integers
        ixes = [self.stoi[s] for s in chunk]
        # if stars align (last idx and len(self.data) % self.block_size == 0), pad with -100, to skip training at the last position
        if len(ixes) < self.block_size + 1:
            assert len(ixes) == self.block_size # i believe this is the only way this could happen, make sure
            ixes.append(-100)
        dix = torch.tensor(ixes, dtype=torch.long)
        return dix[:-1], dix[1:]

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--num-epochs', type=int, default=5, help="number of epochs to train for")
parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size to train with")
parser.add_argument('-l', '--block-size', type=int, default=128, help="block size for the model (length of window of context)")
parser.add_argument('-n', '--num-workers', type=int, default=0, help="number of workers for dataloading")
parser.add_argument('-g', '--num-gpus', type=int, default=1, help="number of gpus to train on")
parser.add_argument('-p', '--pin-memory', type=int, default=1, help="pin memory on dataloaders?")
parser.add_argument('-r', '--precision', type=int, default=32, help="fp precision to use, e.g. 32/16")
parser.add_argument('-o', '--default_root_dir', type=str, default='.', help="best model checkpoint will be written at this location")
args = parser.parse_args()
print(vars(args))

logging.info("preparing the data loaders")
# NOTE: REDUCED DATA SIZE FOR DEBUGGING, TODO CLEAN BEFORE MERGE IF EVER
train_dataset = Text8Dataset('text8', args.block_size, crop=(0,         int(1e6)))
val_dataset   = Text8Dataset('text8', args.block_size, crop=(int(90e6), int(1e5)), override_vocab=train_dataset.vocab)
test_dataset  = Text8Dataset('text8', args.block_size, crop=(int(95e6), int(1e5)), override_vocab=train_dataset.vocab)
common = {'batch_size': args.batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
train_dataloader  = DataLoader(train_dataset, shuffle=True, **common)
val_dataloader  = DataLoader(val_dataset, shuffle=False, **common)
test_dataloader  = DataLoader(test_dataset, shuffle=False, **common)

logging.info("creating the model")
model = GPT(train_dataset.vocab_size, args.block_size, n_layer=6, n_head=8, n_embd=256)

logging.info("preparing the learning rate schedule")
iter_tokens = args.batch_size * args.block_size # number of tokens backpropped in one iteration
epoch_tokens = math.ceil(len(train_dataset) / args.batch_size) * iter_tokens
lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4, warmup_tokens=epoch_tokens//2,
                                         final_tokens=args.num_epochs*epoch_tokens)

t0 = time.time()
logging.info("training...")
trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0, callbacks=[lr_decay],
                     precision=args.precision, default_root_dir=args.default_root_dir)
trainer.fit(model, train_dataloader, val_dataloader)
t1 = time.time()
logging.info("%d epochs took %fs, or %fs/epoch", args.num_epochs, t1 - t0, (t1-t0)/args.num_epochs)

# todo below: I don't yet understand the Lightning checkpoint schema
# logging.info("testing...")
# ckpt_path = os.path.join(args.default_root_dir, 'model.pt')
# model.load_from_checkpoint(ckpt_path) # load the best checkpoint we found
# trainer.test(test_dataloader=test_dataloader)

logging.info("sampling:")
context = "anarchism originated as a term of"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...]
if next(model.parameters()).is_cuda:
    x = x.cuda()
y = sample(model, x, 200, temperature=1.0, sample=True, top_k=None)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
