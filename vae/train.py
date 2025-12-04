import numpy as np
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from vqvae import VQVAE
import matplotlib.pyplot as plt
from dataset import Tamadataset

torch.set_printoptions(linewidth=160)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === VQVAE SETUP ============================================================
batch_size = 256
epochs = 10
learning_rate = 1e-3

num_embeddings = 4
embedding_dim = 2
downsampling = 2 
commitment_loss = 0.1

model = VQVAE(num_embeddings, embedding_dim, downsampling, commitment_loss).to(device)

# === DATASET SETUP ============================================================

workers = 0
# normalize = transforms.Normalize(mean=[0.5], std=[1.0])
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         normalize,
#     ]
# )
# train_dataset = Tamadataset('../datasets/general', transform)
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=workers,
# )
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8,       
    pin_memory=True,
    shuffle=True
)

# # === TRAINING SETUP ============================================================


# # === TRAIN LOOP ================================================================

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------
# Fonction de perplexité corrigée
# ------------------------------
def compute_perplexity_from_indices(encoding_indices, num_embeddings):
    """
    encoding_indices : (B, H, W)
    num_embeddings  : taille du codebook
    """
    # Aplatir : (B*H*W)
    flat = encoding_indices.reshape(-1)

    # Compter les occurrences de chaque code
    counts = torch.bincount(flat, minlength=num_embeddings).float()

    # Distribution
    probs = counts / counts.sum()

    # Perplexité
    perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
    
    return perplexity.item()

criterion = torch.nn.MSELoss()
for epoch in range(epochs):
    total_loss = 0
    total_perplexity = 0
    for imgs, _ in train_loader:
        imgs = imgs.float().to(device)
        optimizer.zero_grad()
        out, quantize_loss, encoding_indices = model(imgs)
        
        recon_loss = criterion(out, imgs)
        loss = recon_loss + quantize_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_perplexity += compute_perplexity_from_indices(encoding_indices, num_embeddings)

    avg_loss = total_loss / len(train_loader)
    avg_perplexity = total_perplexity / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")


# Save checkpoint
torch.save(model.state_dict(), f"./checkpoints/vqvae.pt")

with torch.no_grad():
    all_indices = []

    for images, _ in train_loader:
        images = images.float().to(device)
        # Le modèle doit renvoyer : output, quantize_loss, encoding_indices
        _, _, encoding_indices = model(images)
        # encoding_indices : (B, H, W)
        flat = encoding_indices.reshape(-1).cpu()
        all_indices.append(flat)
    # Concaténer tous les indices de tout le dataset
    all_indices = torch.cat(all_indices, dim=0)
# Compter l'utilisation des codes
counts = torch.bincount(all_indices, minlength=num_embeddings)
print("Utilisation des codes:", counts)
num_codes_used = (counts > 0).sum().item()
print(f"Nombre de codes uniques utilisés dans le codebook: {num_codes_used}")

# === Evaluation / Grid ===
model.eval()

# ------------------------------
# Dé-normalisation
# ------------------------------
def denormalize(x):
    return torch.clamp(x + 0.5, 0, 1)

# ------------------------------
# Reconstruction et affichage
# ------------------------------
model.eval()
with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images.float().to(device)
    
    out, loss, encoding_indices  = model(images)
    recon_loss = criterion(out, images)
    # Perplexité correcte
    perplexity = compute_perplexity_from_indices(encoding_indices, num_embeddings)

    print(f"Reconstruction Loss: {recon_loss.item():.4f}, VQ Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")

    # Dé-normalisation
    orig = denormalize(images[:8].cpu())
    reco = denormalize(out[:8].cpu())

# ------------------------------
# Grid de comparaison
# ------------------------------
grid = torch.cat([orig, reco], dim=0)  # shape: [16, C, H, W]
grid = grid.permute(0, 2, 3, 1).numpy()  # HWC pour matplotlib

fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for i in range(8):
    ax[0, i].imshow(grid[i])
    ax[0, i].axis("off")
    ax[1, i].imshow(grid[i+8])
    ax[1, i].axis("off")

plt.tight_layout()
plt.show()
