# File: tianyusun1/test2/test2-2.0/models/poem2layout.py (WEIGHT AWARE SIZE LOSS)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .decoder import LayoutDecoder

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
        
        # 2. Object Query Embedding
        self.obj_class_embedding = nn.Embedding(num_classes + 2, bb_size, padding_idx=0)
        
        # 3. Spatial Bias Embedding
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)
        
        # 4. KG Feature Projection
        self.kg_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. Decoder
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # Decoder 输出
        decoder_output_size = hidden_size + bb_size
        
        # 6. Prediction Head
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

    def forward(self, input_ids, attention_mask, kg_class_ids, padding_mask, kg_spatial_matrix=None):
        """Query-Based Forward Pass."""
        # 1. Text Encoding
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 

        # 2. Query Embedding
        layout_embed = self.obj_class_embedding(kg_class_ids)
        
        # 3. Masks
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if padding_mask is not None:
            trg_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            trg_mask = trg_mask.masked_fill(padding_mask, float('-inf'))
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)
        else:
            trg_mask = None

        # 4. Spatial Bias
        spatial_bias = self.construct_spatial_bias(kg_class_ids, kg_spatial_matrix)

        # 5. Decoder
        decoder_output = self.layout_decoder(
            layout_embed, 
            text_encoded, 
            src_mask, 
            trg_mask,
            spatial_bias=spatial_bias
        ) 

        # 6. Prediction
        pred_boxes = torch.sigmoid(self.reg_head(decoder_output))
        
        return None, None, pred_boxes, None

    def get_loss(self, pred_cls, pred_bbox_ids, pred_boxes, pred_count, layout_seq, layout_mask, num_boxes, target_coords_gt=None, kg_spatial_matrix=None, kg_class_weights=None):
        """
        Args:
            kg_spatial_matrix: [B, 9, 9] 空间关系矩阵
            kg_class_weights: [B, S] 物体权重 (1.0=Explicit, 0.5=Implicit) [NEW]
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

        # 6. Size Prior Loss (Updated with Weights)
        # [MODIFIED] 传入 kg_class_weights
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
        """计算带 Mask 的 IoU Loss"""
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
        """计算空间关系违反损失"""
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
                    
                    # 1/5: ABOVE/ON_TOP
                    if rel_id in [1, 5]:
                        dist = box_a[1] - box_b[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    # 2: BELOW
                    elif rel_id == 2:
                        dist = box_b[1] - box_a[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    # 3: INSIDE
                    elif rel_id == 3:
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
        """惩罚不合理的重叠"""
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
        """
        尺寸先验 (Weighted): 
        1. Base Area ~ 0.5 / sqrt(N)
        2. Element Area ~ Base * Weight (Implicit=0.5 -> Smaller)
        """
        # pred_boxes: [B, S, 4], area = w * h
        pred_areas = pred_boxes[..., 2] * pred_boxes[..., 3] # [B, S]
        
        if num_boxes is None:
            N = mask.sum(dim=1).clamp(min=1).float() # [B]
        else:
            N = num_boxes.float().clamp(min=1)
            
        # 1. 计算基准面积 (Shape [B, 1])
        base_expected_area = (0.5 / torch.sqrt(N)).unsqueeze(1) 
        
        # 2. 结合 KG 权重
        if weights is not None:
            # weights: [B, S] (1.0 或 0.5)
            target_areas = base_expected_area * weights
        else:
            target_areas = base_expected_area.expand_as(pred_areas)
            
        # 3. 计算 Element-wise Loss
        loss = F.smooth_l1_loss(pred_areas, target_areas, reduction='none')
        
        # 只计算 mask=1 的部分
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        
        return loss