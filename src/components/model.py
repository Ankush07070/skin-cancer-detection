import torch
import torch.nn as nn
import torchvision.models as models


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # 🔥 LIGHTWEIGHT CNN (IMPORTANT CHANGE)
        base_model = models.mobilenet_v2(pretrained=True)

        # Take feature extractor only
        self.cnn = base_model.features   # output channels = 1280

        # 🔥 REDUCE DIMENSION (CRITICAL)
        self.reduce_dim = nn.Conv2d(1280, 256, kernel_size=1)

        # 🔥 LIGHT TRANSFORMER
        self.transformer = TransformerBlock(embed_dim=256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)   # (B, 1280, H, W)

        x = self.reduce_dim(x)  # (B, 256, H, W)

        B, C, H, W = x.shape

        # Flatten → sequence
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, 256)

        x = self.transformer(x)

        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)

        x = self.classifier(x)

        return x