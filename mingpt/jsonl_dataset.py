import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import GPT2Tokenizer



import torch
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = JSONL_Dataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 
    C.trainer.max_iters = 1
    C.trainer.batch_size = 4

    return C


class JSONL_Dataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 64
        return C

    def __init__(self, file_path, block_size):
        super().__init__()
        self.tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.file_path = os.path.normpath(file_path)
        self.block_size = block_size
        self.jsonlines = []
        with pd.read_json(self.file_path, lines=True, chunksize=1) as reader:
            reader
            for chunk in reader:
                chunklet = chunk['text'].tolist()[0]
                if isinstance(chunklet, str):
                    self.jsonlines.append(self.tokenizer(chunklet, padding='max_length', truncation=True, max_length=self.block_size)['input_ids'])
        

    def __len__(self):
        return len(self.jsonlines)

    def __getitem__(self, idx):
        token = self.jsonlines[idx]
        x = torch.tensor(token[:-1], dtype=torch.long)
        y = torch.tensor(token[1:], dtype=torch.long)
        return x, y
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_block_size(self):
        return self.block_size
    
    def text_to_token(self, str):
        return [self.tokenizer(str, padding='max_length', truncation=True, max_length=self.block_size)['input_ids']]
    
    def token_to_text(self, tokens):
        return self.tokenizer.decode(tokens)
            

if __name__ == "__main__":
    #file_path = os.path.abspath("/lustre/scratch/usr/dw87/pile_data_10.jsonl")
    file_path = os.path.join(Path().cwd(), "test_pile_file.jsonl")

     # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    train_dataset = JSONL_Dataset(file_path, 100)
    
    print(train_dataset.__getitem__(0))

    """
    iter = []
    loss = []

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
            loss.append(trainer.loss.item())
            iter.append(trainer.iter_num)

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                pass
                #sample from the model...
                #context = "o god o god"
                #x = torch.tensor(train_dataset.text_to_token(context), dtype=torch.long).to(trainer.device)
                #y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                #completion = ''.join(train_dataset.token_to_text(y))
                #print(completion)
            # save the latest model
            print("saving model")
            #ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            #torch.save(model.state_dict(), ckpt_path)
            model.save('my_model.pth')
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()


    plt.plot(iter, loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.savefig("Loss.png")
    """
    