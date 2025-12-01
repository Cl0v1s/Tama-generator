import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    """
    Convolution masquée pour PixelCNN.
    Type 'A' : masque le pixel courant pour la première couche.
    Type 'B' : masque le pixel courant pour les couches suivantes.
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        assert mask_type in ('A', 'B'), "mask_type doit être 'A' ou 'B'"
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc + (mask_type=='B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, hidden=128, kernel_size=7, n_layers=7):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, hidden)
        self.layers = nn.ModuleList()
        
        # Première couche : Mask-A
        self.layers.append(MaskedConv2d('A', hidden, hidden, kernel_size, padding=kernel_size//2))
        # Couches intermédiaires : Mask-B
        for _ in range(n_layers - 2):
            self.layers.append(MaskedConv2d('B', hidden, hidden, kernel_size, padding=kernel_size//2))
        # Dernière couche : logits
        self.layers.append(nn.Conv2d(hidden, num_embeddings, 1))

    def forward(self, x):
        """
        x: [B,H,W] (indices)
        output: [B, num_embeddings, H, W] (logits)
        """
        h = self.embed(x)  # [B,H,W,hidden]
        h = h.permute(0,3,1,2)  # [B,hidden,H,W]
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        logits = self.layers[-1](h)
        return logits
