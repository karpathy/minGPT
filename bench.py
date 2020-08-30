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
    """
    e.g. Text8 dataset is often used: http://mattmahoney.net/dc/textdata.html
    Vocabulary is lowercase English characters and space for total of 27.
    Training data: First 90M characters.
    Validation data: First 5M characters out of the last 10M characters.
    Testing data: Last 5M characters.
    """

    def __init__(self, data_path, block_size, split, override_vocab=None):

        # load the data and crop it appropriately
        with open(data_path, 'r') as f:
            data = f.read()

        crop = {
            'train': (0, 90000000),
            'val': (90000000, 95000000),
            'test': (95000000, 100000000),
        }[split]
        data = data[crop[0]:crop[1]]

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
        return math.ceil(len(self.data) / self.block_size)

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)
        return dix[:-1], dix[1:]

class CharDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=64, block_size=128, pin_memory=0, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = CharDataset('text8', self.block_size, 'train')

    def prepare_data(self): # called only on 1 GPU/machine
        pass # could technically download text8 here...

    def setup(self, stage): # called for every GPU/machine
        if stage == 'train' or stage == 'fit':
            pass # nothing to do, the train_dataset is initialized in the constructor
        elif stage == 'val':
            self.val_dataset = CharDataset('text8', self.block_size, 'val', override_vocab=self.train_dataset.vocab)
        elif stage == 'test':
            self.test_dataset = CharDataset('text8', self.block_size, 'test', override_vocab=self.train_dataset.vocab)
        else:
            raise ValueError(f"stage {stage} is not recognized")

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, pin_memory=bool(self.pin_memory),
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=bool(self.pin_memory),
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=bool(self.pin_memory),
                            num_workers=self.num_workers)
        return loader


# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--num-epochs', type=int, default=5, help="number of epochs to train for")
parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size to train with")
parser.add_argument('-l', '--block-size', type=int, default=128, help="block size for the model (length of window of context)")
parser.add_argument('-n', '--num-workers', type=int, default=0, help="number of workers for dataloading")
parser.add_argument('-g', '--num-gpus', type=int, default=1, help="number of gpus to train on")
parser.add_argument('-p', '--pin-memory', type=int, default=1, help="pin memory on dataloaders?")
parser.add_argument('-r', '--precision', type=int, default=32, help="fp precision to use, e.g. 32/16")
args = parser.parse_args()
print(vars(args))

logging.info("preparing the data module")
dm = CharDataModule(batch_size=args.batch_size, block_size=args.block_size, num_workers=args.num_workers)

logging.info("creating the model")
model = GPT(dm.train_dataset.vocab_size, args.block_size, n_layer=4, n_head=4, n_embd=128)

logging.info("preparing the learning rate schedule")
iter_tokens = args.batch_size * args.block_size # number of tokens backpropped in one iteration
epoch_tokens = math.ceil(len(dm.train_dataset) / args.batch_size) * iter_tokens
lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4, warmup_tokens=epoch_tokens//4,
                                         final_tokens=args.num_epochs*epoch_tokens)

t0 = time.time()
logging.info("training...")
trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0, callbacks=[lr_decay],
                     precision=args.precision)
trainer.fit(model, dm)
t1 = time.time()
logging.info("%d epochs took %fs, or %fs/epoch", args.num_epochs, t1 - t0, (t1-t0)/args.num_epochs)

logging.info("sampling:")
context = "O God, O God!"
x = torch.tensor([dm.train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
y = sample(model, x, 100, temperature=1.0, sample=True, top_k=None)[0]
completion = ''.join([dm.train_dataset.itos[int(i)] for i in y])
print(completion)
