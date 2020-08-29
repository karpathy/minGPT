"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically. This is a
very basic Trainer class that will only train the model on up to one GPU.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# -----------------------------------------------------------------------------
import os
if int(os.environ.get('USE_LIGHTNING', 0)):
    import pytorch_lightning as pl
else:
    import mingpt.fake_lightning as pl
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class WarmupCosineLearningRateDecay(pl.Callback):
    """
    based on the number of tokens seen during training will adjust the learning rate:
    1. first it will start at zero and gradually ramp up to full learning rate
    2. then it will decay down with the cosine learning rate decay down until 10% of original
    """

    def __init__(self, learning_rate, warmup_tokens, final_tokens):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        # state in this class, will count number of tokens processed so far
        self.tokens = 0

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx=None, dataloader_idx=None):
        _, y = batch
        self.tokens += (y >= 0).sum()  # y == -100 is "ignore", so don't count these
        if self.tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
        else:
            # followed by cosine learning rate decay
            progress = float(self.tokens - self.warmup_tokens) / float(
                max(1, self.final_tokens - self.warmup_tokens))
            lr_mult = 0.1 + 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.learning_rate * lr_mult
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

class Trainer:

    def __init__(self, max_epochs, gpus=0, gradient_clip_val=None, ckpt_path=None, callbacks=None):
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_path = ckpt_path
        self.callbacks = [] if callbacks is None else callbacks
        self.model = None

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        logger.info("saving model checkpoint to %s", self.ckpt_path)
        torch.save(self.model.state_dict(), self.ckpt_path)

    def fit(self, model, train_loader, test_loader=None):
        self.model = model # bind model to the class here

        # ship model to gpu if possible
        device = 'cpu'
        if self.gpus > 0 and torch.cuda.is_available():
            logger.info("found CUDA device, shipping model to GPU")
            device = 'cuda'
            self.model = self.model.to(device)

        # preprare the optimizer
        optimizer = self.model.configure_optimizers()
        self.optimizers = [optimizer]

        def run_epoch(split):
            is_train = split == 'train'
            self.model.train(is_train)
            loader = train_loader if is_train else test_loader

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(device)
                y = y.to(device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    result = self.model.training_step((x, y))
                    loss = result.minimize
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    self.model.zero_grad()
                    loss.backward()
                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    optimizer.step()

                    # notify all relevant callbacks that a batch update ended. e.g. a callback may decay learning rate
                    for cb in self.callbacks:
                        if hasattr(cb, 'on_train_batch_end'):
                            cb.on_train_batch_end(self, None, (x, y))

                    # report progress
                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        for epoch in range(self.max_epochs):

            run_epoch('train')
            if test_loader is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = test_loader is None or test_loss < best_loss
            if self.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
