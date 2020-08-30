"""
A manual, minimal and non-full-featured implementation of boilerplate training loop.
Intentionally made to have the same API as PyTorch Lightning, giving two benefits:
1) Everyone can inspect/hack this simple implementation for educational purposes
2) Everyone can run the full Lightning implementation when they just want to go FAST
"""

import math
import logging
import torch

from tqdm import tqdm
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class Result:
        """ very thin wrapper around a result of a train/val/test step of the model """
        def __init__(self, minimize, checkpoint_on):
            self.minimize = minimize

        def log(self, key, val):
            pass

class TrainResult(Result):
    pass

class LightningModule(nn.Module):
    """ a very thin wrapper on module that returns the model's device similar to Lightning """

    @property
    def device(self):
        return 'cuda' if next(self.parameters()).is_cuda else 'cpu'

    @device.setter
    def device(self, new_device):
        raise RuntimeError("Cannot be set directly, is derived based on the module's parameters' location")

class Callback:
    pass

class LightningDataModule:

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

# -----------------------------------------------------------------------------
"""
Simple Trainer object; Boilerplate that could apply to any arbitrary neural network,
so nothing here really has anything to do with GPT specifically. This is a
very basic Trainer class that will only train the model on up to one GPU.
"""

class Trainer:

    def __init__(self, max_epochs, gpus=0, gradient_clip_val=None, ckpt_path=None, callbacks=None,
                 precision=32, **kwargs):
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.ckpt_path = ckpt_path
        self.callbacks = [] if callbacks is None else callbacks
        self.model = None

        if self.gpus > 1:
            logger.error("This simple Trainer does not support > 1 GPUs, will just use one.")

        if precision != 32:
            logger.error("This simple Trainer does not support non-fp32 precision, will use fp32")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        logger.info("saving model checkpoint to %s", self.ckpt_path)
        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_checkpoint(self):
        logger.info("loading the best model checkpoint from %s", self.ckpt_path)
        state_dict = torch.load(self.ckpt_path)
        self.model.load_state_dict(state_dict)

    def fit(self, model, data_module):
        self.model = model # bind model to the class here

        # prepare the dataloaders for outputting batches
        data_module.prepare_data()
        data_module.setup('train')
        train_loader = data_module.train_dataloader()
        data_module.setup('val')
        val_loader = data_module.val_dataloader()
        data_module.setup('test')
        test_loader = data_module.test_dataloader()

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
            # set model into training or eval mode
            is_train = split == 'train'
            self.model.train(is_train)
            loader = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
            }[split]

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
                mean_loss = torch.mean(losses).item()
                logger.info("val loss: %f", mean_loss)
                return mean_loss

        best_val_loss = float('inf')
        for epoch in range(self.max_epochs):

            run_epoch('train')
            if val_loader is not None:
                val_loss = run_epoch('val')

            # supports early stopping based on the val loss, or just save always if no val set is provided
            good_model = val_loader is None or val_loss < best_val_loss
            if self.ckpt_path is not None and good_model:
                best_val_loss = val_loss
                self.save_checkpoint()

        # finally eval once on the test set at the end.
        self.load_checkpoint() # load the best model we had
        run_epoch('test')
