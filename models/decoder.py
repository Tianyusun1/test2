import torch
import torch.nn as nn
from .transformer_layers import PoemLayoutDecoderLayer

class LayoutDecoder(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, num_layers: int, num_heads: int, ff_size: int = None, dropout: float = 0.1):
        super(LayoutDecoder, self).__init__()
        if ff_size is None:
            ff_size = hidden_size * 4
        self.bb_size = bb_size
        
        self.layers = nn.ModuleList([
            PoemLayoutDecoderLayer(
                hidden_size=hidden_size,
                bb_size=bb_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # The final layer norm input size is hidden_size + bb_size
        self.layer_norm = nn.LayerNorm(hidden_size + bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    # [MODIFIED] Added spatial_bias parameter
    def forward(self, layout_embed: torch.Tensor, text_features: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor, spatial_bias: torch.Tensor = None):
        # layout_embed: [B, T, bb_size]
        # text_features: [B, L_text, hidden_size]
        B, T, bb_size_dim = layout_embed.shape
        B_text, L_text, hidden_size_dim = text_features.shape

        # === 关键修复区域 ===
        # 1. 移除 BUG：原代码错误地将 [CLS] token 重复 T 次作为 text_x 的初始值。
        # 2. 修复：将 text_x 初始化为零向量 (Zero Tensor)，避免 [CLS] token 污染自注意力上下文。
        
        # text_x 流在初始时应为零，其信息将通过 PoemLayoutDecoderLayer 内部的 Cross-Attention 从 text_features 中获取。
        text_x = torch.zeros(B, T, hidden_size_dim, device=layout_embed.device) 
        
        layout_x = layout_embed # [B, T, bb_size]
        # 原有的 BUG 赋值语句 text_x = text_repr_for_layout 已被移除。
        # ==================

        # Iterate through the stack of layers
        for layer in self.layers:
            # layout_x 和 text_x 都是 Query，text_features 是 Memory
            # [MODIFIED] Pass spatial_bias to each layer
            layout_x, text_x = layer(layout_x, text_x, text_features, src_mask, trg_mask, spatial_bias=spatial_bias)

        # After all layers, concatenate the final streams
        # Order: [text_features_stream, layout_features_stream]
        output = torch.cat([text_x, layout_x], dim=-1) # [B, T, hidden_size + bb_size]
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output # [B, T, hidden_size + bb_size]