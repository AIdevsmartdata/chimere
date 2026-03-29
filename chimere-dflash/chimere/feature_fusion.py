"""
Extract hidden states from target model layers and fuse them
via linear projection for KV injection into drafter layers.
"""
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Fuse k target model hidden states and project per-layer KV pairs.
    DFlash key insight: inject into EVERY drafter layer (not just layer 1).
    """

    def __init__(self, config):
        super().__init__()
        self.num_feature_layers = config.num_feature_layers
        self.target_hidden_size = config.target_hidden_size
        self.fusion_dim = config.fusion_dim

        self.fuse_proj = nn.Linear(
            config.num_feature_layers * config.target_hidden_size,
            config.fusion_dim,
            bias=False,
        )

        self.kv_projections = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "k_proj": nn.Linear(
                            config.fusion_dim,
                            config.drafter_num_heads * config.drafter_head_dim,
                            bias=False,
                        ),
                        "v_proj": nn.Linear(
                            config.fusion_dim,
                            config.drafter_num_heads * config.drafter_head_dim,
                            bias=False,
                        ),
                    }
                )
                for _ in range(config.drafter_num_layers)
            ]
        )

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: list of k tensors [batch, seq_len, target_hidden]
        Returns:
            fused: [batch, seq_len, fusion_dim]
            kv_pairs: list of (K, V) for each drafter layer
        """
        import torch

        concatenated = torch.cat(hidden_states_list, dim=-1)
        fused = self.fuse_proj(concatenated)

        kv_pairs = []
        for layer_projs in self.kv_projections:
            k = layer_projs["k_proj"](fused)
            v = layer_projs["v_proj"](fused)
            kv_pairs.append((k, v))

        return fused, kv_pairs
