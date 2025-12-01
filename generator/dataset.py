import torch.utils.data as data

class IndexDataset(data.Dataset):
    def __init__(self, indices_array):
        self.indices = torch.LongTensor(indices_array)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.indices[idx]

dataset = IndexDataset(all_indices)
loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
