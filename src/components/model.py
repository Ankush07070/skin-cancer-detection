import torch
import torch.nn as nn
import torchvision.models as models
from src.logger import logging
from src.exception import CustomException   
import sys



class TransformerBlock(nn.Module):
    try:
            def __init__(self, embed_dim=2048, num_heads=8):
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
                # Self-attention
                attn_output, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_output)

                # Feed forward
                ffn_output = self.ffn(x)
                x = self.norm2(x + ffn_output)

                return x
    except Exception as e:
        logging.error(f"Error in TransformerBlock: {str(e)}")
        raise CustomException(e, sys)

# main  modeel class
class SkinCancerModel(nn.Module):
    try:
        def __init__(self, num_classes=7):
            super().__init__()

            # Pretrained ResNet50
            self.cnn = models.resnet50(pretrained=True)

            # Remove final layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

            self.transformer = TransformerBlock(embed_dim=2048)

            self.pool = nn.AdaptiveAvgPool1d(1)

            self.classifier = nn.Linear(2048, num_classes)

        def forward(self, x):
            # CNN feature extraction
            x = self.cnn(x)  
            # Shape: (B, 2048, H, W)

            B, C, H, W = x.shape

            # Flatten spatial dims → sequence
            x = x.view(B, C, H * W).permute(0, 2, 1)
            # Shape: (B, N, 2048)

            # Transformer
            x = self.transformer(x)

            # Pool
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)

            # Classification
            x = self.classifier(x)

            return x
    except Exception as e:
        logging.error(f"Error in SkinCancerModel: {str(e)}")
        raise CustomException(e, sys)