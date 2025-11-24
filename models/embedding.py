import torch
import torch.nn as nn

class PoemLayoutEmbedding(nn.Module):
    """
    将类别 ID 和 YOLO 格式坐标 (cx, cy, w, h) 转换为向量嵌入。
    """
    def __init__(self, num_classes: int, bb_size: int, dropout: float = 0.1):
        super(PoemLayoutEmbedding, self).__init__()
        self.bb_size = bb_size
        # 注意: num_classes 必须是 布局元素数 + 特殊 token 数
        self.cls_embed = nn.Embedding(num_classes, bb_size - 4) # -4 for raw bbox
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_ids: torch.Tensor, bboxes: torch.Tensor):
        """
        Args:
            cls_ids: [B, T] (类别索引, 0-based)
            bboxes: [B, T, 4] (cx, cy, w, h, 归一化)
        Returns:
            embeddings: [B, T, bb_size]
        """
        cls_emb = self.cls_embed(cls_ids) # [B, T, bb_size - 4]
        # 将原始 bbox 坐标拼接到类别嵌入后面
        embeddings = torch.cat([cls_emb, bboxes], dim=-1) # [B, T, bb_size]
        return self.dropout(embeddings)