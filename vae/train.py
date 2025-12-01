import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from vqvae import VQVAE
import matplotlib.pyplot as plt
from dataset import Tamadataset

torch.set_printoptions(linewidth=160)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_ema = True
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
model = VQVAE(**model_args).to(device)

# Initialize dataset.
batch_size = 32
workers = 0
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
train_dataset = Tamadataset('../datasets/general', transform)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
)

# === TRAINING SETUP ============================================================
epochs = 400
learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print_every = 100

# === TRAIN LOOP ================================================================

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        # Forward pass through VQ-VAE
        # Ton modèle doit retourner :
        #   reconstructions, vq_loss, commitment_loss
        # selon la signature que tu lui as donnée.
        recon, vq_loss, commit_loss = model(images)

        # Reconstruction loss
        recon_loss = torch.mean((images - recon) ** 2)

        # Si tu suis la formule de VQ-VAE :
        # total_loss = recon_loss + vq_loss + beta * commit_loss
        # avec beta souvent entre 0.25 et 0.75
        beta = 0.25
        total_loss = recon_loss + commit_loss * beta
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # Logging
        if batch_idx % print_every == 0:
            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"[Batch {batch_idx}/{len(train_loader)}] "
                f"Reconstruction: {recon_loss.item():.4f} | "
                f"Commit: {commit_loss.item():.4f} | "
                f"Total: {total_loss.item():.4f}"
            )

    # Epoch summary
    print(f"===> Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # Save checkpoint
torch.save(model.state_dict(), f"./checkpoints/vqvae.pt")


model.eval()
images = next(iter(train_loader))
images = images.to(device)

with torch.no_grad():
    recon, _, _ = model(images)

# dé-normalisation
def denormalize(x):
    return torch.clamp(x * 1.0 + 0.5, 0, 1)

orig = denormalize(images[:8].cpu())
reco = denormalize(recon[:8].cpu())

# afficher
grid = torch.cat([orig, reco], dim=0)
grid = grid.permute(0, 2, 3, 1).numpy()

fig, ax = plt.subplots(2, 8, figsize=(12, 4))
for i in range(8):
    ax[0, i].imshow(grid[i])
    ax[0, i].axis("off")
    ax[1, i].imshow(grid[i+8])
    ax[1, i].axis("off")

plt.show()
