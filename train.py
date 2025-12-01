import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import PixelCNN
from vae.vqvae import VQVAE
from vae.dataset import Tamadataset
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}

vqvae = VQVAE(**model_args).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae_epoch_400.pt", map_location=device))

# ===== Hyperparamètres =====
batch_size = 32
num_epochs = 500
learning_rate = 1e-3
H_prime = W_prime = 8  # taille des codes latents du VQ-VAE

# Dataset
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
dataset = Tamadataset(root_dir="./datasets/general",
                      transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Extraire tous les codes latents
all_codes = []
vqvae.eval()
with torch.no_grad():
    for images in dataloader:
        # images: [B, C, H, W]
        _, _, _, encoding_indices = vqvae.quantize(images.to(device))
        # encoding_indices: [B, H'*W']
        codes_2d = encoding_indices.view(-1, H_prime, W_prime)
        all_codes.append(codes_2d)

all_codes = torch.cat(all_codes, dim=0)  # [N, H', W']

# Dataset & DataLoader
dataset = TensorDataset(all_codes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== Initialiser PixelCNN =====
pixelcnn = PixelCNN(num_embeddings=vqvae.vq.num_embeddings, hidden=model_args["num_hiddens"], kernel_size=7, n_layers=7)
pixelcnn.to(device)
pixelcnn.train()
optimizer = optim.Adam(pixelcnn.parameters(), lr=learning_rate)

# ===== Boucle d'entraînement =====
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        (codes,) = batch  # codes: [B,H',W']
        optimizer.zero_grad()
        logits = pixelcnn(codes)  # [B, num_embeddings, H', W']
        loss = F.cross_entropy(logits, codes)  # cross-entropy sur chaque pixel
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * codes.size(0)
    
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
torch.save(pixelcnn.state_dict(), f"./checkpoints/pixelcnn.pt")
