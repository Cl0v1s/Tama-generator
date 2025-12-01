import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, hidden=128, kernel_size=3):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, hidden)
        self.conv1 = nn.Conv2d(hidden, hidden, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(hidden, num_embeddings, 1)  # logits
    def forward(self, x):
        # x: [B,H,W], integer indices
        h = self.embed(x).permute(0,3,1,2) if x.dim()==3 else self.embed(x).permute(0,3,1,2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        logits = self.conv3(h)
        return logits
