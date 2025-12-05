import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import PixelCNN
from vae.vqvae import VQVAE
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === HYPERPARAMS ============================================================
batch_size = 256
epochs = 50
learning_rate = 1e-4

num_embeddings = 8
embedding_dim = 4
downsampling = 2 
commitment_loss = 0.1

H_prime = W_prime = 7
# === MODELS ================================================================
vqvae = VQVAE(num_embeddings, embedding_dim, downsampling, commitment_loss).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae.pt", map_location=device))
vqvae.eval()  # important

pixelcnn = PixelCNN(num_embeddings=num_embeddings, hidden_dim=72, num_layers=14).to(device)
optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(
    root="./vae/data",
    train=True,
    download=True,
    transform=transform,
)

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8,       
    pin_memory=True,
    shuffle=True
)


for epoch in range(epochs):
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        with torch.no_grad():
            _, _, indices = vqvae(images)
        indices = indices.long()

        logits = pixelcnn(indices)
        loss = F.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, num_embeddings),
            indices.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs} - PixelCNN loss: {avg_loss:.4f}")

torch.save(pixelcnn.state_dict(), "./checkpoints/pixelcnn.pt")

