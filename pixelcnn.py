import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """Convolution masquée de PixelCNN avec masque A ou B."""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        _, _, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2

        self.mask[:, :, yc, xc + (mask_type=='A'):] = 0
        self.mask[:, :, yc+1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, hidden_dim=64, num_layers=7):
        super().__init__()
        self.num_embeddings = num_embeddings
        
        layers = []
        
        # 1) Couche d'entrée MASQUÉE TYPE A (pas de dépendance au pixel courant)
        layers.append(MaskedConv2d(
            'A', 
            in_channels=num_embeddings, 
            out_channels=hidden_dim, 
            kernel_size=7, 
            padding=3
        ))
        layers.append(nn.ReLU())
        
        # 2) Couches masquées TYPE B
        for _ in range(num_layers):
            layers.append(MaskedConv2d(
                'B',
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.ReLU())
        
        # 3) Prédiction finale logits pour chaque codebook entry
        layers.append(nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, indices):
        """
        indices: (B, H, W) entiers → one-hot → PixelCNN
        """
        B, H, W = indices.shape
        
        # One-hot → shape (B, num_embeddings, H, W)
        x = F.one_hot(indices, num_classes=self.num_embeddings).permute(0, 3, 1, 2).float()

        logits = self.net(x)  # (B, num_embeddings, H, W)
        return logits
