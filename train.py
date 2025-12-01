import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import PixelCNN
from vae.vqvae import VQVAE
from vae.dataset import Tamadataset
from torchvision import transforms
from dataset import IndexDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_ema = True
H_prime = W_prime = 8
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 128,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}

vqvae = VQVAE(**model_args).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae.pt", map_location=device))
vqvae.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[1.0]*3),
])

raw_dataset = Tamadataset('./datasets/general', transform)
raw_loader = DataLoader(raw_dataset, batch_size=32, shuffle=False)

all_codes = []
with torch.no_grad():
    for images in raw_loader:
        images = images.to(device)
        _, _, _, encoding_indices = vqvae.quantize(images)      # [B, 64]
        codes_2d = encoding_indices.view(-1, H_prime, W_prime)   # [B, 8, 8]
        all_codes.append(codes_2d.cpu())

all_codes = torch.cat(all_codes, dim=0).long()   # [N, 8, 8]
print("Codes shape:", all_codes.shape)

# TensorDataset pour PixelCNN
train_dataset = TensorDataset(all_codes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

pixelcnn = PixelCNN(vqvae.vq.num_embeddings).to(device)
opt = optim.Adam(pixelcnn.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataset = TensorDataset(all_codes)  # ← Garde ça !
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(60):
    for batch in train_loader:
        x = batch[0].to(device)  # [B, 8, 8]
        
        logits = pixelcnn(x)     # <-- CORRECT
        loss = criterion(
            logits.reshape(-1, vqvae.vq.num_embeddings),
            x.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
torch.save(pixelcnn.state_dict(), f"./checkpoints/pixelcnn.pt")
