import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from pixelcnn import PixelCNN
from vae.vqvae import VQVAE
from vae.dataset import Tamadataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_embeddings = 8
embedding_dim = 4
downsampling = 2 
commitment_loss = 0.1

H = W = 7
# === MODELS ================================================================
vqvae = VQVAE(num_embeddings, embedding_dim, downsampling, commitment_loss).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae.pt", map_location=device))
vqvae.eval()  # important

pixelcnn = PixelCNN(num_embeddings=num_embeddings, hidden_dim=72, num_layers=14).to(device)
pixelcnn.load_state_dict(torch.load("./checkpoints/pixelcnn.pt", map_location=device))

device = next(pixelcnn.parameters()).device
samples = torch.zeros((1, H, W), dtype=torch.long, device=device)

for y in range(H):
    for x in range(W):
        logits = pixelcnn(samples)   # (1, num_emb, H, W)
        probs = torch.softmax(logits[0, :, y, x], dim=0)
        samples[0, y, x] = torch.multinomial(probs, 1)

emb = vqvae.embedding(samples)                   # (B, H, W, C)
emb = emb.permute(0, 3, 1, 2)                  # (B, C, H, W)
z_q = vqvae.post_quant_conv(emb)
img = vqvae.decoder(z_q)

# img : Tensor généré par VQ-VAE, shape (1,1,H,W), valeurs [-1,1]
img_show = img.squeeze(0).squeeze(0).cpu().detach()  # (H,W)
img_show = (img_show + 1) / 2 * 255  # [0,255] pour PIL
img_show = img_show.numpy().astype('uint8')

# Convertir en image PIL
pil_img = Image.fromarray(img_show)

# Débruitage avec filtre médian (option la plus simple)
pil_img = pil_img.filter(ImageFilter.MedianFilter(size=5))
threshold_value = 180  # tu peux changer ce seuil
pil_img = pil_img.point(lambda p: 255 if p > threshold_value else 0)

# Afficher l'image débruitée
pil_img.show()
