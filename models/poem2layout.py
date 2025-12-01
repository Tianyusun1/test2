# File: tianyusun1/test2/test2-2.0/models/poem2layout.py (V4.0: LEARNABLE PRIORS + GATED FUSION + 8x8 GRID)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .decoder import LayoutDecoder

# === [NEW] 1. 门控融合模块 ===
class GatedFusion(nn.Module):
    """
    动态融合 Content Features 和 Position Features。
    F_final = alpha * F_content + (1 - alpha) * F_pos
    其中 alpha 是由输入动态决定的门控值 (0~1)。
    """
    def __init__(self, hidden_size):
        super().__init__()
        # 门控网络：输入 concat(content, pos)，输出 1 个 scalar
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, content, pos):
        # content, pos: [B, T, hidden_size]
        combined = torch.cat([content, pos], dim=-1)
        alpha = self.gate_net(combined) # [B, T, 1]
        
        fused = alpha * content + (1 - alpha) * pos
        return self.norm(fused)

# === [NEW] 2. 可学习关系先验网络 ===
class RelationPriorNet(nn.Module):
    """
    从 KG 关系矩阵中直接学习位置编码，而不只是依赖手工 Grid。
    输入：当前物体与其他物体的关系 ID 列表
    输出：该物体隐含的位置 Embedding
    """
    def __init__(self, num_relations, embedding_dim, hidden_size):
        super().__init__()
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        # 聚合器：将一个物体与其他所有物体的关系特征聚合
        self.aggregator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size) 
        )
        
    def forward(self, kg_spatial_matrix):
        """
        Args:
            kg_spatial_matrix: [B, S, S] 关系矩阵 (LongTensor)
        Returns:
            learned_pos: [B, S, hidden_size]
        """
        # 1. Embed 关系: [B, S, S] -> [B, S, S, dim]
        rels = self.rel_embedding(kg_spatial_matrix)
        
        # 2. 聚合 (Aggregation)
        # 对于物体 i (Row i)，它与所有 j 的关系共同决定了它的位置
        # 例如：i above j, i inside k -> i 应该在上方且被包含
        # 对最后两个维度 (S, dim) 进行 Mean Pooling，得到 [B, S, dim]
        aggregated = rels.mean(dim=2) 
        
        # 3. 映射到 Hidden Size
        learned_pos = self.aggregator(aggregated)
        return learned_pos

# === 主模型更新 ===
class Poem2LayoutGenerator(nn.Module):
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, 
                 decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, 
                 reg_loss_weight: float = 1.0, iou_loss_weight: float = 1.0, area_loss_weight: float = 1.0,
                 relation_loss_weight: float = 5.0,  # 关系约束权重
                 overlap_loss_weight: float = 2.0,   # 重叠惩罚权重
                 size_loss_weight: float = 2.0,      # 尺寸先验权重
                 **kwargs): 
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes # 9
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        
        # Loss weights
        self.reg_loss_weight = reg_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.area_loss_weight = area_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.size_loss_weight = size_loss_weight
        
        # 1. Text Encoder
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # 2. Object Query Embedding (Content)
        self.obj_class_embedding = nn.Embedding(num_classes + 2, bb_size, padding_idx=0)
        
        # 3. Spatial Bias Embedding (Attention Bias)
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)

        # === [MODIFIED] 4. Hybrid Position Encoders ===
        # A. 手工 Grid Encoder (8x8 -> 64)
        self.grid_encoder = nn.Sequential(
            nn.Linear(64, bb_size), # [MODIFIED] 25 -> 64
            nn.ReLU(),
            nn.Linear(bb_size, bb_size),
            nn.Dropout(dropout)
        )
        
        # B. [NEW] 可学习先验网络 (Relation -> Position)
        # 让模型自己去理解 "Above" 到底意味着什么样的位置 Embedding
        self.learnable_prior_net = RelationPriorNet(
            num_relations=self.num_spatial_relations,
            embedding_dim=32,
            hidden_size=bb_size # 输出维度与 bb_size 对齐
        )
        
        # C. [NEW] 门控融合
        self.fusion_gate = GatedFusion(bb_size)
        # ==============================================
        
        # 5. KG Feature Projection
        self.kg_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 6. Decoder
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # Decoder 输出
        decoder_output_size = hidden_size + bb_size
        
        # 7. Prediction Head
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
        map_ids = cls_ids - 2 
        gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
        
        b_idx = torch.arange(B, device=cls_ids.device).view(B, 1, 1).expand(-1, S, S)
        row_idx = gather_ids.view(B, S, 1).expand(-1, -1, S)
        col_idx = gather_ids.view(B, 1, S).expand(-1, S, -1)
        
        rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] 
        
        is_valid_obj = (map_ids >= 0)
        valid_pair_mask = is_valid_obj.unsqueeze(2) & is_valid_obj.unsqueeze(1)
        rel_ids = rel_ids.masked_fill(~valid_pair_mask, 0)
        
        spatial_bias = self.spatial_bias_embedding(rel_ids) 
        spatial_bias = spatial_bias.permute(0, 3, 1, 2).contiguous() 
        
        return spatial_bias

    def forward(self, input_ids, attention_mask, kg_class_ids, padding_mask, 
                kg_spatial_matrix=None, location_grids=None):
        """Query-Based Forward Pass."""
        
        # 1. Text Encoding
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 

        # 2. Query Embedding (Content)
        # kg_class_ids: [B, T] -> [B, T, bb_size]
        content_embed = self.obj_class_embedding(kg_class_ids)
        
        # === [MODIFIED] 3. Robust Position Injection ===
        # 准备位置特征，初始化为 0
        pos_feat = torch.zeros_like(content_embed)
        
        # A. 来自手工 Grid 的位置信息 (Explicit Handcrafted)
        if location_grids is not None:
            # location_grids: [B, T, 8, 8]
            B, T, H, W = location_grids.shape
            
            # Flatten: [B, T, 64]
            grid_flat = location_grids.view(B, T, -1).to(content_embed.device)
            
            # Encode: [B, T, 64] -> [B, T, bb_size]
            handcrafted_pos = self.grid_encoder(grid_flat) # [B, T, bb_size]
            pos_feat = pos_feat + handcrafted_pos

        # B. 来自可学习网络的位置信息 (Implicit Learnable)
        if kg_spatial_matrix is not None:
            # 重新提取当前序列的关系矩阵 [B, T, T] (因为传入的是全类别的 9x9)
            B, T = kg_class_ids.shape
            map_ids = kg_class_ids - 2 
            gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
            
            b_idx = torch.arange(B, device=kg_class_ids.device).view(B, 1, 1).expand(-1, T, T)
            row_idx = gather_ids.view(B, T, 1).expand(-1, -1, T)
            col_idx = gather_ids.view(B, 1, T).expand(-1, T, -1)
            
            # 当前 batch 中物体 T 与物体 T 之间的关系 ID
            seq_rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] # [B, T, T]
            
            learned_pos = self.learnable_prior_net(seq_rel_ids) # [B, T, bb_size]
            pos_feat = pos_feat + learned_pos # 叠加两种位置信息

        # C. 门控融合 (Gated Fusion)
        # 不再是简单的 layout_embed = content + pos
        # 而是让模型决定怎么融合
        layout_embed = self.fusion_gate(content_embed, pos_feat)
        # ===============================================
        
        # 4. Masks
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if padding_mask is not None:
            trg_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            trg_mask = trg_mask.masked_fill(padding_mask, float('-inf'))
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)
        else:
            trg_mask = None

        # 5. Spatial Bias (Attention Bias from KG Matrix)
        spatial_bias = self.construct_spatial_bias(kg_class_ids, kg_spatial_matrix)

        # 6. Decoder
        decoder_output = self.layout_decoder(
            layout_embed, 
            text_encoded, 
            src_mask, 
            trg_mask,
            spatial_bias=spatial_bias
        ) 

        # 7. Prediction
        pred_boxes = torch.sigmoid(self.reg_head(decoder_output))
        
        return None, None, pred_boxes, None

    def get_loss(self, pred_cls, pred_bbox_ids, pred_boxes, pred_count, layout_seq, layout_mask, num_boxes, target_coords_gt=None, kg_spatial_matrix=None, kg_class_weights=None):
        """
        计算损失 (保持不变)
        """
        loss_mask = layout_mask 
        target_boxes = target_coords_gt
        
        if loss_mask.dim() == 1:
             loss_mask = loss_mask.view(pred_boxes.shape[0], -1)
        
        num_valid = loss_mask.sum().clamp(min=1)

        # 1. Reg Loss
        loss_reg = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none') 
        loss_reg = (loss_reg.mean(dim=-1) * loss_mask).sum() / num_valid
        
        # 2. IoU Loss
        loss_iou = self._compute_iou_loss(pred_boxes, target_boxes, loss_mask)
        
        # 3. Area Loss
        pred_w, pred_h = pred_boxes[..., 2], pred_boxes[..., 3]
        tgt_w, tgt_h = target_boxes[..., 2], target_boxes[..., 3]
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        loss_area = F.smooth_l1_loss(pred_area, tgt_area, reduction='none')
        loss_area = (loss_area * loss_mask).sum() / num_valid
        
        # 4. Relation Constraint Loss
        loss_relation = self._compute_relation_loss(pred_boxes, loss_mask, kg_spatial_matrix)

        # 5. Overlap Penalty
        loss_overlap = self._compute_overlap_loss(pred_boxes, loss_mask, kg_spatial_matrix)

        # 6. Size Prior Loss
        loss_size_prior = self._compute_size_loss(pred_boxes, loss_mask, num_boxes, kg_class_weights)

        # Total Loss
        total_loss = self.reg_loss_weight * loss_reg + \
                     self.iou_loss_weight * loss_iou + \
                     self.area_loss_weight * loss_area + \
                     self.relation_loss_weight * loss_relation + \
                     self.overlap_loss_weight * loss_overlap + \
                     self.size_loss_weight * loss_size_prior
                     
        return total_loss, loss_relation, loss_overlap, \
               loss_reg, loss_iou, loss_size_prior, loss_area

    def _compute_iou_loss(self, pred, target, mask):
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        tgt_x1 = target[..., 0] - target[..., 2] / 2
        tgt_y1 = target[..., 1] - target[..., 3] / 2
        tgt_x2 = target[..., 0] + target[..., 2] / 2
        tgt_y2 = target[..., 1] + target[..., 3] / 2
        
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        pred_area = pred[..., 2] * pred[..., 3]
        tgt_area = target[..., 2] * target[..., 3]
        union_area = pred_area + tgt_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        loss = (1.0 - iou) * mask
        return loss.sum() / (mask.sum().clamp(min=1))

    def _compute_relation_loss(self, pred_boxes, mask, kg_spatial_matrix):
        if kg_spatial_matrix is None:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, S, _ = pred_boxes.shape
        count = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) < 2: continue
            
            for i in valid_indices:
                for j in valid_indices:
                    if i == j: continue
                    rel_id = kg_spatial_matrix[b, i, j].item()
                    if rel_id == 0: continue

                    box_a = pred_boxes[b, i]
                    box_b = pred_boxes[b, j]
                    
                    if rel_id in [1, 5]: # ABOVE / ON_TOP
                        dist = box_a[1] - box_b[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    elif rel_id == 2: # BELOW
                        dist = box_b[1] - box_a[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    elif rel_id == 3: # INSIDE
                        a_x1, a_y1 = box_a[0]-box_a[2]/2, box_a[1]-box_a[3]/2
                        a_x2, a_y2 = box_a[0]+box_a[2]/2, box_a[1]+box_a[3]/2
                        b_x1, b_y1 = box_b[0]-box_b[2]/2, box_b[1]-box_b[3]/2
                        b_x2, b_y2 = box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2
                        
                        l_inside = F.relu(b_x1 - a_x1) + F.relu(a_x2 - b_x2) + \
                                   F.relu(b_y1 - a_y1) + F.relu(a_y2 - b_y2)
                        if l_inside > 0:
                            loss += l_inside
                            count += 1

        if count > 0:
            return loss / count
        return loss

    def _compute_overlap_loss(self, pred_boxes, mask, kg_spatial_matrix):
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, S, _ = pred_boxes.shape
        count = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) < 2: continue
            
            boxes = pred_boxes[b, valid_indices]
            N = boxes.shape[0]
            
            x1 = boxes[:, 0] - boxes[:, 2]/2
            y1 = boxes[:, 1] - boxes[:, 3]/2
            x2 = boxes[:, 0] + boxes[:, 2]/2
            y2 = boxes[:, 1] + boxes[:, 3]/2
            areas = boxes[:, 2] * boxes[:, 3]
            
            inter_x1 = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
            inter_y1 = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
            inter_x2 = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
            inter_y2 = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
            
            inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            union_area = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
            iou_mat = inter_area / (union_area + 1e-6)
            
            ignore_overlap = torch.eye(N, device=pred_boxes.device).bool()
            
            if kg_spatial_matrix is not None:
                for i_local, i_global in enumerate(valid_indices):
                    for j_local, j_global in enumerate(valid_indices):
                        if i_local == j_local: continue
                        rel = kg_spatial_matrix[b, i_global, j_global].item()
                        if rel in [3, 4]:
                            ignore_overlap[i_local, j_local] = True
                            ignore_overlap[j_local, i_local] = True
            
            triu_mask = torch.triu(torch.ones(N, N, device=pred_boxes.device), diagonal=1).bool()
            target_mask = triu_mask & (~ignore_overlap)
            
            if target_mask.sum() > 0:
                bad_iou = iou_mat[target_mask]
                loss += F.relu(bad_iou - 0.1).sum()
                count += target_mask.sum()

        if count > 0:
            return loss / count
        return loss

    def _compute_size_loss(self, pred_boxes, mask, num_boxes, weights=None):
        pred_areas = pred_boxes[..., 2] * pred_boxes[..., 3] # [B, S]
        
        if num_boxes is None:
            N = mask.sum(dim=1).clamp(min=1).float() # [B]
        else:
            N = num_boxes.float().clamp(min=1)
            
        base_expected_area = (0.5 / torch.sqrt(N)).unsqueeze(1) 
        
        if weights is not None:
            target_areas = base_expected_area * weights
        else:
            target_areas = base_expected_area.expand_as(pred_areas)
            
        loss = F.smooth_l1_loss(pred_areas, target_areas, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss