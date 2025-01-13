import torch
import torch.nn as nn

class AVAttention2D(nn.Module):
    def __init__(self, embed_dim=256):
        super(AVAttention2D, self).__init__()
        
        # Sigmoid activation for attention map
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio, visual):
        """
        Args:
            audio (torch.Tensor): Audio input of shape (B, 256)
            visual (torch.Tensor): Visual input of shape (B, 256, 64, 64)

        Returns:
            torch.Tensor: Attention-modulated visual features of shape (B, 256, 64, 64)
        """
        B, C, H, W = visual.shape

        # Project audio features to match the embedding space
        audio_proj = audio.view(B, C)  # (B, 256)

        # Reshape for broadcasting
        audio_proj = audio_proj.view(B, C, 1, 1)  # (B, 256, 1, 1)

        # Compute dot product at each spatial location
        attention_map = (visual * audio_proj).sum(dim=1, keepdim=True)  # (B, 1, 64, 64)

        # Apply sigmoid activation to normalize the attention map
        attention_map = self.sigmoid(attention_map)  # (B, 1, 64, 64)

        return attention_map
