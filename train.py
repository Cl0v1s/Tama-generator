# model = PixelCNN(num_embeddings=model.vq.num_embeddings).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(50):
#     for batch in loader:
#         batch = batch.to(device)  # [B,H,W]
#         optimizer.zero_grad()
#         logits = model(batch)     # [B,num_embeddings,H,W]
#         loss = F.cross_entropy(logits, batch)  # compare logits avec vrai indices
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch} loss: {loss.item():.4f}")

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vae.vqvae import VQVAE
import matplotlib.pyplot as plt
from vae.dataset import Tamadataset
from dataset import IndexDataset
from pixelcnn import PixelCNN
from torch.nn.functional import cross_entropy, softmax

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
    "num_embeddings": 10,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}




model_vqvae = VQVAE(**model_args).to(device)
model_vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae_epoch_600.pt", map_location=device))

def build(indices):
    global model_vqvae
    indices = torch.LongTensor(indices).to(device)  # shape [H*W] ou [1,H*W]
    H = W = int(np.sqrt(indices.numel()))  # si aplati
    indices = indices.view(1, H, W)
    embeddings = model_vqvae.vq.e_i_ts.transpose(0, 1)  # shape [num_embeddings, embedding_dim]
    z_q = embeddings[indices.view(-1)].float()  # [H*W, embedding_dim]
    z_q = z_q.view(1, H, W, -1)                # [1,H,W,D]
    z_q = z_q.permute(0, 3, 1, 2)              # [1,D,H,W] pour decoder

    with torch.no_grad():
        generated = model_vqvae.decoder(z_q)  # [1,C,H_orig,W_orig]

    # Dé-normalisation et conversion en image
    arr = ((generated[0].cpu().permute(1,2,0).numpy() + 1) * 0.5 * 255).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

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
train_dataset = Tamadataset('./datasets/general', transform)
all_indices = []
for img in train_dataset:
    img = img.unsqueeze(0).to(device)
    _, _, _, indices = model_vqvae.quantize(img)
    all_indices.append(indices.cpu().numpy())
all_indices = np.stack(all_indices)  # shape: [N, H*W]

train_dataset = IndexDataset(all_indices)
loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = PixelCNN(num_embeddings=model_vqvae.vq.num_embeddings).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for batch in loader:
        batch = batch.to(device)  # [B,H,W]
        optimizer.zero_grad()
        logits = model(batch)     # [B,num_embeddings,H,W]
        loss = cross_entropy(logits, batch)  # compare logits avec vrai indices
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item():.4f}")

num_downsampling_layers = model_args['num_downsampling_layers']  # 2
H = W = 32 // (2 ** num_downsampling_layers)
model.eval()
# indices = torch.LongTensor(train_dataset[0]).to(device)  # shape [H*W] ou [1,H*W]
# gen_indices[0,0,:] = indices[0,0,:]  # première ligne
gen_indices = torch.zeros(1,H,W,dtype=torch.long).to(device)
indices = torch.LongTensor(train_dataset[0]).to(device)  # shape [H*W] ou [1,H*W]
indices = indices.view(1,H,W)
gen_indices[0,0,:] = indices[0,0,:]  # première ligne

with torch.no_grad():
    for i in range(H):
        for j in range(W):
            logits = model(gen_indices)
            probs = torch.softmax(logits[0, :, i, j], dim=0)
            gen_indices[0,i,j] = torch.multinomial(probs, 1)

build(gen_indices[0])