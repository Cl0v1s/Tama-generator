import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import GatedPixelCNN
from vae.vqvae import VQVAE
from vae.dataset import Tamadataset
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
pixelcnn.load_state_dict(torch.load("./checkpoints/pixelcnn.pt", map_location=device))


# ------------------------------
# 1. Génération de codes avec PixelCNN
# ------------------------------
# pixelcnn.generate() doit retourner des indices d'embedding
codes = pixelcnn.generate(shape=(7, 7), temperature=0.1)
print("Codes générés:", codes.shape)

# ------------------------------
# 2. Transformer les indices en embeddings
# ------------------------------
# VectorQuantizer.embeddings contient la table d'embeddings
quantized = vqvae.quantizer.embeddings(codes)  # [B, H, W, embedding_dim]
quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]
print("Quantized:", quantized.shape)

# ------------------------------
# 3. Passer par le décodeur
# ------------------------------
images = vqvae.decoder(quantized)  # [B, 3, H_img, W_img]
print("Images reconstruites:", images.shape)

# ------------------------------
# 4. Affichage
# ------------------------------
img = images[0].cpu().detach().permute(1, 2, 0)  # [H_img, W_img, C]
# Normalisation de -1..1 -> 0..1
plt.imshow((img + 1) / 2)  # si sortie du modèle entre -1 et 1
plt.axis("off")
plt.show()
