# File: tianyusun1/test2/test2-2.0/models/poem2layout.py (QUERY-BASED FIXED)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
# 注意：不再导入 PoemLayoutEmbedding，因为我们不再使用复杂的离散序列嵌入
from .decoder import LayoutDecoder

class Poem2LayoutGenerator(nn.Module):
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, 
                 decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, 
                 reg_loss_weight: float = 1.0, iou_loss_weight: float = 1.0, area_loss_weight: float = 1.0,
                 **kwargs): # 忽略多余的旧参数 (如 num_bbox_bins)
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes # 9
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        
        # Loss weights
        self.reg_loss_weight = reg_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.area_loss_weight = area_loss_weight
        
        # 1. Text Encoder
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # 2. Object Query Embedding (核心修改)
        # 将 KG 提供的类别 ID (Queries) 映射为向量，作为 Layout Stream 的初始输入
        # 词表大小: num_classes + 2 (0:PAD, 1:Unused, 2-10:Objects)
        # 输入维度: bb_size (因为 LayoutDecoder 期望 layout 流是 bb_size)
        self.obj_class_embedding = nn.Embedding(num_classes + 2, bb_size, padding_idx=0)
        
        # 3. Spatial Bias Embedding
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)
        
        # 4. KG Feature Projection (用于增强文本)
        self.kg_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. Decoder (复用现有的 LayoutDecoder)
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # Decoder 输出是 cat([text_stream, layout_stream]) -> hidden_size + bb_size
        decoder_output_size = hidden_size + bb_size
        
        # 6. Prediction Head (只保留回归头)
        # 输入: Decoder Output -> 输出: [cx, cy, w, h] (Sigmoid 归一化)
        self.reg_head = nn.Sequential(
            nn.Linear(decoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )

    def construct_spatial_bias(self, cls_ids, kg_spatial_matrix):
        """构建空间注意力偏置 (通用逻辑)"""
        if kg_spatial_matrix is None:
            return None
            
        B, S = cls_ids.shape
        
        # 映射 ID: 2-10 -> 0-8
        map_ids = cls_ids - 2 
        gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
        
        # Gather 关系 ID
        b_idx = torch.arange(B, device=cls_ids.device).view(B, 1, 1).expand(-1, S, S)
        row_idx = gather_ids.view(B, S, 1).expand(-1, -1, S)
        col_idx = gather_ids.view(B, 1, S).expand(-1, S, -1)
        
        rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] # [B, S, S]
        
        # Mask 掉无效物体 (PAD=0, map_ids < 0)
        is_valid_obj = (map_ids >= 0)
        valid_pair_mask = is_valid_obj.unsqueeze(2) & is_valid_obj.unsqueeze(1)
        rel_ids = rel_ids.masked_fill(~valid_pair_mask, 0)
        
        # Embedding
        spatial_bias = self.spatial_bias_embedding(rel_ids) # [B, S, S, H]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2).contiguous() # [B, H, S, S]
        
        return spatial_bias

    def forward(self, input_ids, attention_mask, kg_class_ids, padding_mask, kg_spatial_matrix=None):
        """
        Query-Based Forward Pass.
        Args:
            input_ids: [B, L] BERT 输入
            attention_mask: [B, L] BERT Mask
            kg_class_ids: [B, S] KG 提取的物体类别 ID (Queries)
            padding_mask: [B, S] True 表示 PAD (用于 trg_mask)
            kg_spatial_matrix: [B, 9, 9] 空间关系矩阵
        """
        # 1. Text Encoding
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 
        # [B, L, H]

        # 2. Query Embedding (Layout Stream Init)
        # [B, S] -> [B, S, bb_size]
        layout_embed = self.obj_class_embedding(kg_class_ids)
        
        # 3. Masks
        # src_mask (Text): [B, 1, 1, L]
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # trg_mask (Layout): [B, 1, 1, S] 
        # Query-Based 不需自回归 Mask，只需 Padding Mask
        # padding_mask 是 [B, S]，我们需要广播到 [B, 1, 1, S] 并转为 float mask
        # PyTorch attention 通常接受 boolean mask (True=ignore) 或 float additive mask
        # 这里 LayoutDecoder 内部逻辑可能期望 float mask (0 for keep, -inf for ignore)
        if padding_mask is not None:
            trg_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            trg_mask = trg_mask.masked_fill(padding_mask, float('-inf'))
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
        else:
            trg_mask = None

        # 4. Spatial Bias
        spatial_bias = self.construct_spatial_bias(kg_class_ids, kg_spatial_matrix)

        # 5. Decoder
        # 输入: layout_embed (Queries), text_encoded (Keys/Values)
        decoder_output = self.layout_decoder(
            layout_embed, 
            text_encoded, 
            src_mask, 
            trg_mask,
            spatial_bias=spatial_bias
        ) 
        # Output: [B, S, hidden_size + bb_size]

        # 6. Prediction
        # 直接回归坐标 [B, S, 4]，使用 Sigmoid 归一化到 [0, 1]
        pred_boxes = torch.sigmoid(self.reg_head(decoder_output))
        
        # 兼容旧接口返回 (pred_cls, pred_bbox_ids, pred_coord_float, pred_count)
        # 这里 cls 和 bbox_ids 不再需要，填 None
        return None, None, pred_boxes, None

    def get_loss(self, pred_cls, pred_bbox_ids, pred_boxes, pred_count, layout_seq, layout_mask, num_boxes, target_coords_gt=None):
        """
        Args:
            pred_boxes: [B, S, 4] 预测框
            layout_mask: [B, S] 这里实际上是 loss_mask (1=有GT, 0=无GT)
            target_coords_gt: [B, S, 4] 真实框
        """
        # 兼容 Trainer 调用，这里 layout_mask 传入的就是 loss_mask
        loss_mask = layout_mask 
        target_boxes = target_coords_gt
        
        # 确保 Mask 形状匹配
        if loss_mask.dim() == 1: # 如果意外展平了
             loss_mask = loss_mask.view(pred_boxes.shape[0], -1)
        
        # 计算有效物体数量 (防止除零)
        num_valid = loss_mask.sum().clamp(min=1)

        # 1. Reg Loss (Smooth L1)
        loss_reg = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none') # [B, S, 4]
        # 对每个框求平均，然后应用 Mask
        loss_reg = (loss_reg.mean(dim=-1) * loss_mask).sum() / num_valid
        
        # 2. IoU Loss
        iou_loss = self._compute_iou_loss(pred_boxes, target_boxes, loss_mask)
        
        # 3. Area Loss
        pred_w, pred_h = pred_boxes[..., 2], pred_boxes[..., 3]
        tgt_w, tgt_h = target_boxes[..., 2], target_boxes[..., 3]
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        
        loss_area = F.smooth_l1_loss(pred_area, tgt_area, reduction='none')
        loss_area = (loss_area * loss_mask).sum() / num_valid
        
        # Total Loss
        total_loss = self.reg_loss_weight * loss_reg + \
                     self.iou_loss_weight * iou_loss + \
                     self.area_loss_weight * loss_area
                     
        # 为了兼容 Trainer 的返回值解包 (需要 7 个值)
        # 填 0 补位: cls_loss, coord_loss, count_loss
        return total_loss, torch.tensor(0.0, device=total_loss.device), torch.tensor(0.0, device=total_loss.device), \
               loss_reg, iou_loss, torch.tensor(0.0, device=total_loss.device), loss_area

    def _compute_iou_loss(self, pred, target, mask):
        """计算带 Mask 的 IoU Loss"""
        # pred, target: [B, S, 4] (cx, cy, w, h)
        # Convert to corners
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        tgt_x1 = target[..., 0] - target[..., 2] / 2
        tgt_y1 = target[..., 1] - target[..., 3] / 2
        tgt_x2 = target[..., 0] + target[..., 2] / 2
        tgt_y2 = target[..., 1] + target[..., 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        pred_area = pred[..., 2] * pred[..., 3]
        tgt_area = target[..., 2] * target[..., 3]
        union_area = pred_area + tgt_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        # Loss = 1 - IoU
        loss = 1.0 - iou
        
        # Apply Mask
        loss = (loss * mask).sum() / (mask.sum().clamp(min=1))
        return loss