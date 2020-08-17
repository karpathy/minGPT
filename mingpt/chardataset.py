import logging
import math
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class TextDataset(Dataset):
    data_size: int
    vocab_size: int


class CharDataset(TextDataset):
    def __init__(
        self, url: str, cache_dir: str, filename: str, block_size: int
    ) -> None:
        self.block_size = block_size
        path = Path(cache_dir) / filename
        if not path.exists():
            log.info(f"Downloading {url} -> {path}")
            r = requests.get(url, allow_redirects=True)
            data = r.content.decode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        else:
            log.info(f"Loading {path}")
            data = path.read_text()

        chars = list(set(data))
        self.data_size, self.vocab_size = len(data), len(chars)
        log.info(
            "data has %d characters, %d unique." % (self.data_size, self.vocab_size)
        )

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))

        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
