import torch.utils.data as data
import torch

class IndexDataset(data.Dataset):
    def __init__(self, indices_array):
        self.indices = torch.LongTensor(indices_array)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x = self.indices[idx]           # shape [1,H*W]
        return torch.LongTensor(x[0])  # supprime le premier axe → [H*W]
