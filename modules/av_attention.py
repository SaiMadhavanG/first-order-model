import torch
import torch.nn as nn

class AVAttention2D(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super(AVAttention2D, self).__init__()
        self.attn_audio_to_visual = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_visual_to_audio = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)  # Fuse the two outputs
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, audio, visual):
        # audio, visual: (B, 256, 64, 64)
        B, C, H, W = audio.shape
        spatial_dim = H * W

        # Flatten spatial dimensions
        audio_flat = audio.view(B, C, spatial_dim).permute(0, 2, 1)  # (B, 4096, 256)
        visual_flat = visual.view(B, C, spatial_dim).permute(0, 2, 1)  # (B, 4096, 256)

        # Cross-attention
        attn_audio_to_visual, _ = self.attn_audio_to_visual(audio_flat, visual_flat, visual_flat)  # (B, 4096, 256)
        attn_visual_to_audio, _ = self.attn_visual_to_audio(visual_flat, audio_flat, audio_flat)  # (B, 4096, 256)

        # Fuse attention outputs
        fused = torch.cat([attn_audio_to_visual, attn_visual_to_audio], dim=-1)  # (B, 4096, 512)
        fused = self.proj(fused)  # (B, 4096, 256)
        fused = self.norm(fused)

        # Reshape back to (B, 256, 64, 64)
        fused = fused.permute(0, 2, 1).view(B, C, H, W)  # (B, 256, 64, 64)
        return fused
