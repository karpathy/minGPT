import math

# -----------------------------------------------------------------------------
import os
if int(os.environ.get('USE_LIGHTNING', 0)):
    import pytorch_lightning as pl
else:
    import mingpt.fake_lightning as pl
# -----------------------------------------------------------------------------

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
