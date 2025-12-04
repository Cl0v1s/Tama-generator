import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import GatedPixelCNN
from vae.vqvae import VQVAE
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === HYPERPARAMS ============================================================
batch_size = 64
epochs = 10

model_args = {
    "in_channels": 1,        # MNIST grayscale
    "hidden_channels": 128,
    "embedding_dim": 76,     
    "num_embeddings": 56,    
    "commitment_cost": 0.1,
}
H_prime = W_prime = 7

# === MODELS ================================================================
vqvae = VQVAE(**model_args).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae.pt", map_location=device))
vqvae.eval()  # important

pixelcnn = GatedPixelCNN(model_args["num_embeddings"], model_args["embedding_dim"]).to(device)

# === DATASET ===============================================================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset_mnist = datasets.MNIST(
    root="./vae/data",
    train=True,
    download=True,
    transform=transform
)
train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=batch_size, shuffle=True)

# === EXTRAIRE TOUS LES CODES DU CODEBOOK ====================================
all_codes = []
with torch.no_grad():
    for images, _ in train_loader_mnist:
        images = images.to(device)
        _, _, _, z_q = vqvae(images)  # [B, D, H', W']

        B, D, H, W = z_q.shape
        # Flatten latents
        flat_zq = z_q.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Calcul des distances vers le codebook
        distances = (flat_zq**2).sum(dim=1, keepdim=True) + \
                    (vqvae.quantizer.embeddings.weight**2).sum(dim=1) - \
                    2 * torch.matmul(flat_zq, vqvae.quantizer.embeddings.weight.t())
        encoding_indices = torch.argmin(distances, dim=1)  # [B*H*W]

        codes_2d = encoding_indices.view(B, H, W)  # [B, H', W']
        all_codes.append(codes_2d.cpu())

all_codes = torch.cat(all_codes, dim=0).long()  # [N, H', W']
print("Codes shape:", all_codes.shape)
all_codes_flat = all_codes.view(-1)
print("Codebook min/max:", all_codes_flat.min(), all_codes_flat.max())
print("Number of unique codes:", len(all_codes_flat.unique()))
counts = torch.bincount(all_codes_flat - all_codes_flat.min())
print("Counts per code:", counts)

# === CREER DATASET PIXELCNN ================================================
train_dataset = TensorDataset(all_codes)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,     num_workers=8,       pin_memory=True  )

# === PIXELCNN TRAINING =====================================================
opt = optim.Adam(pixelcnn.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_embeddings = model_args["num_embeddings"]

for epoch in range(epochs):
    for batch in train_loader:
        x = batch[0].to(device)  # [B, H', W']
        
        logits = pixelcnn(x)     # [B, num_embeddings, H', W']
        
        loss = criterion(
            logits.reshape(-1, num_embeddings),
            x.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

torch.save(pixelcnn.state_dict(), "./checkpoints/pixelcnn.pt")

# === GENERATION ============================================================
pixelcnn.eval()
with torch.no_grad():
    generated = pixelcnn.generate()  # [B, H', W']
print("Shape of a sample code:", all_codes[0].shape)
print("Shape of generated sample:", generated.shape)
