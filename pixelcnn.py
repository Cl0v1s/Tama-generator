import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size must be odd"
        self.mask_type = mask_type
        self.residual = residual

        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)

        # Vertical stack
        self.vert_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, padding=padding_shp)

        # Vertical -> horizontal
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        # Horizontal stack
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, padding=padding_shp)

        # Residual link
        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        # Mask vertical
        self.vert_stack.weight.data[:, :, -1].zero_()
        # Mask horizontal
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def forward(self, x_v, x_h):

        if self.mask_type == 'A':
            self.make_causal()

        # Vertical stack
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert)

        # Horizontal stack
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz)

        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15):
        super().__init__()
        self.dim = dim

        # Input embedding (for discrete codes)
        self.embedding = nn.Embedding(input_dim, dim)

        # PixelCNN layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = (i != 0)
            self.layers.append(GatedMaskedConv2d(mask_type, dim, kernel, residual))

        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x):
        """
        x : [B, H, W] integers (codebook indices)
        """
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)

        return self.output_conv(x_h)

    @torch.no_grad()
    def generate(self, shape=(8, 8), batch_size=64, temperature=1):
        """
        Génère des indices de codebook à partir du PixelCNN.

        Args:
            shape (tuple): (H, W) taille de l'image codée
            batch_size (int)
            temperature (float): contrôle la diversité, <1 = plus conservatif, >1 = plus varié
        """
        device = next(self.parameters()).device
        x = torch.zeros((batch_size, *shape), dtype=torch.long, device=device)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x)  # [B, num_embeddings, H, W]
                # On prend les logits du pixel courant
                pixel_logits = logits[:, :, i, j] / temperature
                probs = F.softmax(pixel_logits, dim=-1)
                # Multinomial sampling
                x[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return x