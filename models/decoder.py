import torch
import torch.nn as nn
from typing import Optional, Tuple
from .transformer_layers import PoemLayoutDecoderLayer

class LayoutDecoder(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, num_layers: int, num_heads: int, ff_size: int = None, dropout: float = 0.1):
        super(LayoutDecoder, self).__init__()
        if ff_size is None:
            ff_size = hidden_size * 4
        self.bb_size = bb_size
        self.hidden_size = hidden_size
        
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
        # 因为最后我们将 text_x 和 layout_x 进行了拼接
        self.layer_norm = nn.LayerNorm(hidden_size + bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                layout_embed: torch.Tensor, 
                text_features: torch.Tensor, 
                src_mask: Optional[torch.Tensor], 
                trg_mask: Optional[torch.Tensor], 
                spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            layout_embed: [B, T, bb_size] - 布局的 Embedding (Geometry)
            text_features: [B, L_text, hidden_size] - 文本编码器的输出 (Semantics)
            src_mask: 针对 text_features 的掩码
            trg_mask: 针对 layout 序列的自回归掩码
            spatial_bias: [B, n_heads, T, T] - 相对位置偏置 (V5.2 Feature)
        """
        B, T, _ = layout_embed.shape
        
        # === 关键修复区域 (V5.3 Fix) ===
        # 原问题：全 0 初始化导致第一层 Text 流缺乏语义引导。
        # 修正：使用 BERT 的 [CLS] token (Index 0) 初始化 text_x 流。
        
        # 1. 提取 [CLS] token: [B, L_text, H] -> [B, 1, H]
        # 确保 text_features 不为空
        if text_features.size(1) == 0:
             raise ValueError("text_features sequence length is 0, cannot extract CLS token.")
             
        cls_token = text_features[:, 0, :].unsqueeze(1)
        
        # 2. 扩展到序列长度 T: [B, 1, H] -> [B, T, H]
        # 使用 clone() 确保后续 inplace 操作安全，避免梯度问题
        text_x = cls_token.expand(-1, T, -1).clone()
        
        layout_x = layout_embed # [B, T, bb_size]
        # ==================

        # Iterate through the stack of layers
        for layer in self.layers:
            # layout_x 和 text_x 作为 Query 传入，text_features 作为 Memory
            # [MODIFIED] Pass spatial_bias to each layer explicitly
            layout_x, text_x = layer(
                layout_x=layout_x, 
                text_x=text_x, 
                memory=text_features, 
                src_mask=src_mask, 
                trg_mask=trg_mask, 
                spatial_bias=spatial_bias
            )

        # After all layers, concatenate the final streams
        # 最终输出融合了语义流(text_x)和几何流(layout_x)
        # Order: [text_features_stream, layout_features_stream]
        output = torch.cat([text_x, layout_x], dim=-1) # [B, T, hidden_size + bb_size]
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output # [B, T, hidden_size + bb_size]