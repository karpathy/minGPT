"""
A manual, minimal and non-full-featured implementation of boilerplate training loop.
Intentionally made to have the same API as PyTorch Lightning, giving two benefits:
1) Everyone can inspect/hack this simple implementation for educational purposes
2) Everyone can run the full Lightning implementation when they just want to go FAST
"""

import os
import math
import logging

from tqdm import tqdm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class LightningModule(nn.Module):
    pass

class Callback:
    pass

# -----------------------------------------------------------------------------
"""
Simple Trainer object; Boilerplate that could apply to any arbitrary neural network,
so nothing here really has anything to do with GPT specifically. This is a
very basic Trainer class that will only train the model on up to one GPU.
"""

class Trainer:

    def __init__(self, max_epochs, gpus=0, gradient_clip_val=None, default_root_dir='.', callbacks=None,
                 precision=32, **kwargs):
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.callbacks = [] if callbacks is None else callbacks
        self.model = None

        if default_root_dir is not None:
            os.makedirs(default_root_dir, exist_ok = True)
            self.default_root_dir = default_root_dir

        if self.gpus > 1:
            logger.error("This simple Trainer does not support > 1 GPUs, will just use one.")

        if precision != 32:
            logger.error("This simple Trainer does not support non-fp32 precision, will use fp32")

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.default_root_dir, 'model.pt')
        logger.info("saving model checkpoint to %s", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.default_root_dir, 'model.pt')
        logger.info("loading model from %s", ckpt_path)
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)

    def eval_split_(self, dataloader, split):

        self.model.eval()
        use_gpu = self.gpus > 0 and torch.cuda.is_available()
        losses = []
        for it, (x, y) in enumerate(dataloader):
            # place data on the correct device
            if use_gpu:
                x, y = x.cuda(), y.cuda(non_blocking=True)
            # forward the model
            with torch.no_grad():
                if split == 'val':
                    result = self.model.validation_step((x, y))
                elif split == 'test':
                    result = self.model.test_step((x, y))
                losses.append(result['loss'].item())
        mean_loss = torch.mean(torch.tensor(losses)).item()

        logger.info("%s loss: %f", split, mean_loss)
        return mean_loss

    def test(self, test_dataloaders): # note we expect a list of dataloaders here
        self.load_checkpoint() # load the best checkpoint we found during optimization
        return self.eval_split_(test_dataloaders, 'test')

    def val(self, val_dataloader):
        return self.eval_split_(val_dataloader, 'val')

    def fit(self, model, train_dataloader, val_dataloader=None):
        self.model = model # bind model to the class here
        self.model.train()

        # ship model to gpu if possible
        use_gpu = self.gpus > 0 and torch.cuda.is_available()
        if use_gpu:
            logger.info("found CUDA device, shipping model to GPU")
            self.model.cuda()

        # prepare the optimizer
        optimizer = self.model.configure_optimizers()
        self.optimizers = [optimizer]

        # start the training loop
        best_val_loss = float('inf')
        for epoch in range(self.max_epochs):

            # do an epoch of training
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for it, (x, y) in pbar:

                # place data on the correct device
                if use_gpu:
                    x, y = x.cuda(), y.cuda(non_blocking=True)

                # forward the model
                result = self.model.training_step((x, y))
                loss = result['loss']

                # reset gradient
                for param in self.model.parameters():
                    param.grad = None # a faster alternative to model.zero_grad()

                # backward pass
                loss.backward()

                # clip the gradient to mitigate loss explosions
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                # update all parameters
                optimizer.step() # todo: use fused optimizer

                # notify all relevant callbacks that a batch update ended. e.g. a callback may decay learning rate
                for cb in self.callbacks:
                    if hasattr(cb, 'on_train_batch_end'):
                        cb.on_train_batch_end(self, None, (x, y))

                # report progress
                lr = optimizer.param_groups[0]['lr']
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            # calculate the current validation loss and checkpoint the model for early stopping
            if val_dataloader is not None:
                val_loss = self.val(val_dataloader)
                if (self.default_root_dir is not None) and (val_loss < best_val_loss):
                    best_val_loss = val_loss
                    self.save_checkpoint()
