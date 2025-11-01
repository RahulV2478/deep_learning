from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    """
    Simple MLP that flattens left/right lane boundary points and predicts future waypoints.
    Input:  track_left  (B, n_track, 2), track_right (B, n_track, 2)
    Output: waypoints   (B, n_waypoints, 2)
    """
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden: Tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        in_dim = (n_track * 2) * 2  # left/right, each has (n_track,2)
        out_dim = n_waypoints * 2

        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        b, nt, c = track_left.shape  # (B, n_track, 2)
        assert nt == self.n_track and c == 2, "track_left shape mismatch"
        x = torch.cat([track_left, track_right], dim=1)        # (B, 2*n_track, 2)
        x = x.reshape(b, -1)                                   # (B, 4*n_track)
        pred = self.net(x)                                     # (B, 2*n_waypoints)
        return pred.view(b, self.n_waypoints, 2)               # (B, n_waypoints, 2)


class TransformerPlanner(nn.Module):
    """
    Perceiver-style planner:
      - Encode boundary points with small MLP to d_model
      - Concatenate left/right sets; add a learned side embedding
      - Use learned waypoint queries as 'latent array' (queries)
      - One or more TransformerDecoder layers for cross-attention
      - Project each query to (x,y) waypoint
    """
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Encode (x,y) to d_model
        self.point_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Side/type embedding: left=0, right=1
        self.side_embed = nn.Embedding(2, d_model)

        # Waypoint queries (latent array)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer decoder (queries attend over encoded boundary points)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project per-query feature to (x,y)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            track_left:  (B, n_track, 2)
            track_right: (B, n_track, 2)
        Returns:
            (B, n_waypoints, 2)
        """
        B, nL, _ = track_left.shape
        B2, nR, _ = track_right.shape
        assert B == B2 and nL == self.n_track and nR == self.n_track

        # Encode points
        left_feats = self.point_encoder(track_left)                        # (B, n_track, d)
        right_feats = self.point_encoder(track_right)                      # (B, n_track, d)

        # Add side embeddings
        left_feats  = left_feats  + self.side_embed.weight[0]              # (d,)
        right_feats = right_feats + self.side_embed.weight[1]              # (d,)

        # Concatenate memory: (B, 2*n_track, d_model)
        memory = torch.cat([left_feats, right_feats], dim=1)

        # Waypoint queries: (B, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Cross-attention: queries attend to memory
        dec = self.decoder(tgt=queries, memory=memory)                     # (B, n_waypoints, d_model)

        # Predict (x,y) per query
        out = self.head(dec)                                               # (B, n_waypoints, 2)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C, H//p, p, W//p, p) -> (B, H//p * W//p, C * p * p)
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return self.projection(x)  # (B, N, D)


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block:
      LN -> MHA -> +res -> LN -> MLP -> +res
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        hidden = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        # MLP
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class ViTPlanner(nn.Module):
    """
    Vision Transformer planner:
      - Patch embed image to tokens
      - Add learnable positional embeddings
      - Run several encoder blocks
      - Pool (mean) and predict (x,y) waypoints
    """
    def __init__(
        self,
        n_waypoints: int = 3,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        img_h: int = 96,
        img_w: int = 128,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.patch = PatchEmbedding(h=img_h, w=img_w, patch_size=patch_size, in_channels=3, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Head to (n_waypoints, 2) — project pooled representation
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_waypoints * 2),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize input to approx zero mean / unit var (per channel)
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]  # (B,3,96,128)

        # Patches + positional embeddings
        x = self.patch(x) + self.pos_embed  # (B, N, D)

        # Encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Global average pooling over tokens
        x = x.mean(dim=1)  # (B, D)

        # Predict (B, n_waypoints, 2)
        out = self.head(x).view(x.size(0), self.n_waypoints, 2)
        return out


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return str(output_path)


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024