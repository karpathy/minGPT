"""
Some tests for minGPT
"""

import unittest
import torch
from mingpt.model import GPT

class GPT2Tester(unittest.TestCase):    

    def test_dtypes(self):
        """
            Dtype tests for GPT2 model
        """
        config_fp16 = GPT.get_default_config()
        config_fp16.merge_from_dict({'dtype':'float16', 'vocab_size':50257, 'block_size':1024})
        config_fp16.model_type = 'gpt2'

        config_fp32 = GPT.get_default_config()
        config_fp32.merge_from_dict({'vocab_size':50257, 'block_size':1024})
        config_fp32.model_type = 'gpt2'


        model_fp16 = GPT(config_fp16)
        model_fp32 = GPT(config_fp32)

        # Check whether the dtype has been checked correctly
        self.assertTrue(model_fp16.dtype == torch.float16)
        self.assertTrue(model_fp32.dtype == torch.float32)

        # Checck whether the memory footprint is half of the fp32 model
        self.assertTrue(model_fp16.get_memory_footprint() == model_fp32.get_memory_footprint() // 2)


if __name__ == '__main__':
    unittest.main()