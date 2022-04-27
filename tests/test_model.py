import unittest

import torch

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import TrainerConfig


class GPTTester(unittest.TestCase):
    def test_forward(self):
        # create a tiny config with "tiny" hyperparams
        batch_size = 1
        vocab_size = 5
        block_size = 128
        n_layer = 4
        n_head = 4
        n_embd = 16
        tiny_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

        model = GPT(tiny_config)
        model.eval()
        input_ids = torch.zeros(
            batch_size, block_size, dtype=torch.long
        )  # `dtype=torch.long` because ids have to be of type long integers
        logits, _ = model(input_ids)

        self.assertEqual(logits.shape, (batch_size, block_size, vocab_size))

        # test illegal model block size
        illegal_block_size = block_size + 1
        input_ids_with_extrablock = torch.IntTensor(batch_size, illegal_block_size)

        self.assertRaises(AssertionError, model, input_ids_with_extrablock)

    def test_configure_optimizers(self):
        # create a tiny config with "tiny" hyperparams
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
        model.eval()

        trainer_config = TrainerConfig()
        optimizer = model.configure_optimizers(trainer_config)
        group_decay, group_nodecay = optimizer.param_groups

        # test param_groups' `weight_decay` attribute
        self.assertAlmostEqual(group_decay["weight_decay"], 0.1)
        self.assertAlmostEqual(group_nodecay["weight_decay"], 0.0)

        # test decay parameters
        n_params_group_decay = sum([param.numel() for param in group_decay["params"]])
        self.assertEqual(n_params_group_decay, 52)

        # test no_decay parameters
        n_params_group_no_decay = sum(
            [param.numel() for param in group_nodecay["params"]]
        )
        self.assertEqual(n_params_group_no_decay, 38)
