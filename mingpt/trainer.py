"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader

from mingpt.text_dataset import TextDataset
from mingpt.conf import TrainerConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_dataset: TextDataset,
        test_dataset: TextDataset,
        config: TrainerConfig,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            logger.info(f"Using device : {self.device}")
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=config.learning_rate, betas=config.betas
        )

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data, batch_size=config.batch_size, num_workers=config.num_workers
            )
            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            final_tokens = (
                                200
                                * len(self.train_dataset)
                                * self.train_dataset.block_size
                            )
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(max(1, final_tokens - config.warmup_tokens))
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                logger.info("test loss: %f", np.mean(losses))

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch("train")
            if self.test_dataset is not None:
                run_epoch("test")

            self.save_checkpoint()
