import torch
import torch.nn as nn
from .transformer_layers import PoemLayoutDecoderLayer

class LayoutDecoder(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, num_layers: int, num_heads: int, ff_size: int = None, dropout: float = 0.1):
        super(LayoutDecoder, self).__init__()
        if ff_size is None:
            ff_size = hidden_size * 4
        self.bb_size = bb_size
        # Renamed from 'layers' to 'layers' to be a clear nn.ModuleList attribute
        self.layers = nn.ModuleList([
            PoemLayoutDecoderLayer(
                hidden_size=hidden_size,
                bb_size=bb_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        # The output size of each layer is [B, T, hidden_size + bb_size] after concatenation inside the layer
        # So, the final layer norm input size is hidden_size + bb_size
        # Note: The original PoemLayoutDecoderLayer did not return concatenated output.
        # It returned (layout_out, text_out) separately.
        # So, the final concatenation and layer norm should happen here in LayoutDecoder.
        # Final output size will be [B, T, hidden_size + bb_size]
        self.layer_norm = nn.LayerNorm(hidden_size + bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, layout_embed: torch.Tensor, text_features: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor):
        # layout_embed: [B, T, bb_size]
        # text_features: [B, L_text, hidden_size]
        B, T, bb_size_dim = layout_embed.shape
        B_text, L_text, hidden_size_dim = text_features.shape

        # Expand text features to match layout sequence length T
        # A simple way: repeat the [CLS] token (or mean pool) T times
        # Or use interpolation if alignment is critical
        # Here we use the first token (often [CLS]) and repeat
        text_repr_for_layout = text_features[:, 0:1, :].expand(-1, T, -1) # [B, T, hidden_size]

        layout_x = layout_embed # [B, T, bb_size]
        text_x = text_repr_for_layout # [B, T, hidden_size]

        # Iterate through the stack of layers
        for layer in self.layers: # self.layers is the nn.ModuleList
            # Call the fixed PoemLayoutDecoderLayer
            # Input: layout_x [B, T, bb_size], text_x [B, T, hidden_size], text_memory [B, L_text, hidden_size], src_mask, trg_mask
            # Output: new_layout_x [B, T, bb_size], new_text_x [B, T, hidden_size]
            layout_x, text_x = layer(layout_x, text_x, text_features, src_mask, trg_mask)

        # After all layers, concatenate the final streams
        # Order: [text_features_stream, layout_features_stream]
        output = torch.cat([text_x, layout_x], dim=-1) # [B, T, hidden_size + bb_size]
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output # [B, T, hidden_size + bb_size]
