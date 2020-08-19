import math
from typing import Optional
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class CharDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=64, block_size=128, num_workers=8):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = 0
        self.data_size = 0

    def setup(self, stage: Optional[str] = None):
        url = 'https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt'
        self.__download_file(url)

        text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
        self.train_dataset = CharDataset(text, self.block_size)
        self.vocab_size = self.train_dataset.vocab_size
        self.data_size = self.train_dataset.data_size

    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers)
        return dl

    def __download_file(self, url):
        import urllib
        testfile = urllib.URLopener()
        testfile.retrieve(url, 'input.txt')


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data_size = data_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

