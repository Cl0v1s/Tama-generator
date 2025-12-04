import glob
import io
import torch
from torch.utils.data import Dataset
from PIL import Image

class Tamadataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = glob.glob(root_dir + "/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)

        return image
