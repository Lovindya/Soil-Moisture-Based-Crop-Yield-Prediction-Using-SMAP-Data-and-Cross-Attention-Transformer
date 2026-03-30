import torch
import torch.nn as nn

class FeatureCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, context=None):
        x = self.norm1(x + self.self_attn(x, x, x)[0])

        if context is not None:
            x = self.norm2(x + self.cross_attn(x, context, context)[0])

        # Feed-forward
        x = self.norm3(x + self.ffn(x))
        return x


class CornYieldTransformer(nn.Module):
    def __init__(self, n_features, emb_dim=128, num_heads=8, num_blocks=2, dropout=0.1):
        super().__init__()

        self.feature_emb = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(n_features)])

        self.pos_emb = nn.Parameter(torch.randn(n_features, emb_dim))

        self.blocks = nn.ModuleList([
            FeatureCrossAttentionBlock(emb_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, x, context=None):
        """
        x: (batch_size, n_features)
        context: optional tensor (batch_size, seq_len_context, emb_dim)
        """
        tokens = [emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_emb)]
        x_tokens = torch.stack(tokens, dim=1) 

        x_tokens = x_tokens + self.pos_emb.unsqueeze(0)

        for block in self.blocks:
            x_tokens = block(x_tokens, context)

        pooled = x_tokens.mean(dim=1)

        out = self.predictor(pooled)
        return out

def predict(X_tensor, device):
    model = CornYieldTransformer(
        n_features=X_tensor.shape[1],
        emb_dim=128,
        num_heads=8,
        num_blocks=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load("models/best_cross_attention_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        return model(X_tensor).cpu().numpy()