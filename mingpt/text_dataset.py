from torch.utils.data import Dataset


class TextDataset(Dataset):
    data_size: int
    vocab_size: int
    block_size: int
