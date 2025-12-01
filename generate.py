import torch
import torch.nn.functional as F
from pixelcnn import PixelCNN
from vae.vqvae import VQVAE
from torchvision.transforms import ToPILImage

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

def generate_image(pixelcnn, vqvae, H_prime=8, W_prime=8, num_embeddings=512, device='cuda'):
    """
    Génère une image à partir du PixelCNN + VQ-VAE.

    Args:
        pixelcnn : modèle PixelCNN entraîné
        vqvae : VQ-VAE entraîné
        H_prime, W_prime : taille des codes latents
        num_embeddings : nombre d'embeddings du VQ-VAE
        device : 'cuda' ou 'cpu'

    Returns:
        image: tensor [C,H,W] dans [0,1]
    """
    pixelcnn.eval()
    vqvae.eval()
    
    with torch.no_grad():
        # Tensor vide pour stocker les indices latents
        generated_codes = torch.zeros(1, H_prime, W_prime, dtype=torch.long, device=device)
        
        # Génération autoregressive pixel par pixel
        for i in range(H_prime):
            for j in range(W_prime):
                logits = pixelcnn(generated_codes)  # [1, num_embeddings, H', W']
                probs = F.softmax(logits[:, :, i, j], dim=1)  # probabilité du pixel courant
                # échantillonnage
                sampled_pixel = torch.multinomial(probs.squeeze(0), 1)
                generated_codes[0, i, j] = sampled_pixel

        # Convertir les codes en embeddings et décoder
        embeddings = vqvae.vq.e_i_ts.transpose(0, 1)[generated_codes]  # [1,H',W',D]
        embeddings = embeddings.permute(0, 3, 1, 2)  # [1,D,H',W']
        image_recon = vqvae.decoder(embeddings)  # [1,3,H,W]
        image_recon = torch.sigmoid(image_recon)  # optionnel, si sortie non bornée
        
        return image_recon[0]  # [3,H,W]

vqvae = VQVAE(**model_args).to(device)
vqvae.load_state_dict(torch.load("./vae/checkpoints/vqvae_epoch_400.pt", map_location=device))
pixelcnn = PixelCNN(num_embeddings=vqvae.vq.num_embeddings, hidden=model_args["num_hiddens"], kernel_size=7, n_layers=7)
pixelcnn.load_state_dict(torch.load("./checkpoints/pixelcnn.pt", map_location=device))

pixelcnn.to(device)
vqvae.to(device)

# Génération
image = generate_image(pixelcnn, vqvae, H_prime=8, W_prime=8, num_embeddings=512, device=device)

to_pil = ToPILImage()
pil_image = to_pil(image.cpu())
pil_image.show()
