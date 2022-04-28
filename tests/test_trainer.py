import io
import tempfile
import unittest
from contextlib import redirect_stderr

import torch
from torch.utils.data import Dataset

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


class DummyDataset(Dataset):
    """
    Dummy Dataset for tests
    """

    def __init__(self, vocab_size, block_size, batch_size):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        return x, y


class TrainerTester(unittest.TestCase):
    def test_train(self):
        # create a model with "tiny" hyperparams
        vocab_size = 2
        block_size = 2
        n_layer = 1
        n_head = 1
        n_embd = 2
        tiny_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

        set_seed(42)  # to make the test is reproducible

        model = GPT(tiny_config)

        # create a trainer
        batch_size = 32
        dummy_dataset = DummyDataset(vocab_size, block_size, batch_size)
        tconf = TrainerConfig(
            max_epochs=1,
            batch_size=batch_size,
            lr_decay=True,
            warmup_tokens=2,
            final_tokens=4,
            num_workers=1,
        )

        trainer = Trainer(model, dummy_dataset, None, tconf)

        # capture training logs into variable with `contextlib.redirect_stderr`
        with redirect_stderr(io.StringIO()) as stderr:
            trainer.train()

        # `split("\r")[-1]` captures the last log state
        captured_log = stderr.getvalue().split("\r")[-1]
        expected_log_prefix = (
            "epoch 1 iter 0: train loss 0.70439. lr 3.000000e-05: 100%"
        )

        self.assertTrue(captured_log.startswith(expected_log_prefix))

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create a temporary path where model will be saved
            ckpt_path = f"{tmp_dir}/my_model"

            # create a model with "tiny" hyperparams
            vocab_size = 2
            block_size = 2
            n_layer = 1
            n_head = 1
            n_embd = 2
            tiny_config = GPTConfig(
                vocab_size=vocab_size,
                block_size=block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
            )

            model = GPT(tiny_config)

            # create a trainer
            batch_size = 32
            dummy_dataset = DummyDataset(vocab_size, block_size, batch_size)
            tconf = TrainerConfig(
                max_epochs=1,
                batch_size=batch_size,
                lr_decay=True,
                warmup_tokens=2,
                final_tokens=4,
                num_workers=1,
                ckpt_path=ckpt_path,
            )

            trainer = Trainer(model, dummy_dataset, None, tconf)
            trainer.save_checkpoint()

            # load saved model
            saved_model_state_dict = torch.load(ckpt_path)

            # make sure that parameters of original model & saved/loaded model are equal
            for key, tensor in model.state_dict().items():
                saved_tensor = saved_model_state_dict[key]
                self.assertTrue(torch.equal(tensor, saved_tensor))
