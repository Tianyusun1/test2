# models/poem2layout.py (V5.0: R-GAT + AESTHETIC LOSS)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .decoder import LayoutDecoder

# === 1. 门控融合模块 (保持不变) ===
class GatedFusion(nn.Module):
    """
    动态融合 Content Features 和 Position Features。
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, content, pos):
        combined = torch.cat([content, pos], dim=-1)
        alpha = self.gate_net(combined)
        fused = alpha * content + (1 - alpha) * pos
        return self.norm(fused)

# === [NEW] 2. 图神经网络关系先验 (R-GAT) ===
# [升级 V5.0] 替换了原有的简单 RelationPriorNet
class GraphRelationPriorNet(nn.Module):
    """
    [升级版 V5.0] Relational Graph Attention Network (R-GAT)
    引入 Attention 机制，根据语义相似度和关系类型动态计算邻居权重。
    """
    def __init__(self, num_relations, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # 线性变换
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        
        # 关系嵌入：将每种关系映射为向量，融入 Attention
        self.rel_embed_k = nn.Embedding(num_relations, hidden_dim)
        self.rel_embed_v = nn.Embedding(num_relations, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features, spatial_matrix):
        """
        Args:
            node_features: [B, T, D]
            spatial_matrix: [B, T, T] (Relation IDs)
        """
        B, T, D = node_features.shape
        H = self.num_heads
        d_k = self.head_dim

        # 1. 线性投影 & 分头 [B, T, H, d_k]
        q = self.q_proj(node_features).view(B, T, H, d_k)
        k = self.k_proj(node_features).view(B, T, H, d_k)
        v = self.v_proj(node_features).view(B, T, H, d_k)

        # 2. 准备关系嵌入
        # [B, T, T] -> [B, T, T, D] -> [B, T, T, H, d_k]
        r_k = self.rel_embed_k(spatial_matrix).view(B, T, T, H, d_k)
        r_v = self.rel_embed_v(spatial_matrix).view(B, T, T, H, d_k)

        # 3. 计算 Attention Scores: Q * (K + R_k)^T
        # q: [B, T, 1, H, d_k]
        # k_prime: [B, 1, T, H, d_k] (包含节点特征和关系特征)
        q = q.unsqueeze(2) # [B, T, 1, H, d_k]
        k_prime = k.unsqueeze(1) + r_k # [B, 1, T, H, d_k]
        
        # Attention Logic
        # [B, T, 1, H, d_k] * [B, 1, T, H, d_k] -> [B, T, T, H] (点积求和)
        scores = (q * k_prime).sum(dim=-1) / (d_k ** 0.5)
        
        attn_weights = torch.softmax(scores, dim=2) # [B, T, T, H]
        attn_weights = self.dropout(attn_weights)

        # 4. 聚合: Weights * (V + R_v)
        v_prime = v.unsqueeze(1) + r_v # [B, 1, T, H, d_k]
        
        # Weighted Sum: [B, T, T, H, 1] * [B, 1, T, H, d_k] -> [B, T, T, H, d_k] -> Sum dim 2 -> [B, T, H, d_k]
        agg = (attn_weights.unsqueeze(-1) * v_prime).sum(dim=2)
        
        # 5. 合并多头
        agg = agg.view(B, T, D)
        output = self.out_proj(agg)
        
        # 6. 残差 + Norm
        output = output + node_features
        output = self.norm(output)
        
        # 最后的激活
        output = self.activation(output)
        
        return output

# === 3. 布局变换编码器 (保持不变) ===
class LayoutTransformerEncoder(nn.Module):
    """
    使用 Transformer Encoder 来编码 GT 布局。
    能捕捉框与框之间的排布关系（Arrangement），让隐变量 z 包含构图信息。
    """
    def __init__(self, input_dim=4, hidden_size=768, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 50, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, boxes, mask=None):
        """
        boxes: [B, T, 4]
        mask: [B, T] (True for padding)
        """
        B, T, _ = boxes.shape
        x = self.input_proj(boxes)
        
        # Add Positional Embedding
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        else:
            x = x + self.pos_embed[:, :self.pos_embed.size(1), :]
            
        # Transformer Encoding
        # mask: True 为 Padding，直接传给 src_key_padding_mask
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Pooling: 获取全局特征 (Masked Mean Pooling)
        if mask is not None:
            # mask: True is padding, need to invert for multiplication (0 for padding)
            valid_mask = (~mask).unsqueeze(-1).float() # [B, T, 1]
            sum_feat = (x * valid_mask).sum(dim=1)
            count = valid_mask.sum(dim=1).clamp(min=1e-6)
            global_feat = sum_feat / count
        else:
            global_feat = x.mean(dim=1)
            
        return global_feat

# === 主模型更新 ===
class Poem2LayoutGenerator(nn.Module):
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, 
                 decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, 
                 reg_loss_weight: float = 1.0, iou_loss_weight: float = 1.0, area_loss_weight: float = 1.0,
                 relation_loss_weight: float = 5.0,  
                 overlap_loss_weight: float = 2.0,   
                 size_loss_weight: float = 2.0,
                 # [NEW V5.0] 新增审美损失权重
                 alignment_loss_weight: float = 0.5,
                 balance_loss_weight: float = 0.5,
                 latent_dim: int = 32,               
                 **kwargs): 
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes # 9
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.latent_dim = latent_dim
        
        # Loss weights
        self.reg_loss_weight = reg_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.area_loss_weight = area_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.size_loss_weight = size_loss_weight
        # [NEW]
        self.alignment_loss_weight = alignment_loss_weight
        self.balance_loss_weight = balance_loss_weight
        
        # 1. Text Encoder
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # 2. Object Query Embedding (Content)
        self.obj_class_embedding = nn.Embedding(num_classes + 2, bb_size, padding_idx=0)
        
        # 3. Spatial Bias Embedding (Attention Bias)
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)

        # === 4. Hybrid Position Encoders ===
        # A. 手工 Grid Encoder
        self.grid_encoder = nn.Sequential(
            nn.Linear(64, bb_size), 
            nn.ReLU(),
            nn.Linear(bb_size, bb_size),
            nn.Dropout(dropout)
        )
        
        # B. [升级] GNN Relation Prior (R-GAT)
        self.gnn_prior = GraphRelationPriorNet(
            num_relations=self.num_spatial_relations,
            input_dim=bb_size, 
            hidden_dim=bb_size,
            num_heads=4, # 默认4头
            dropout=dropout
        )
        
        # C. 门控融合
        self.fusion_gate = GatedFusion(bb_size)

        # === 5. CVAE Components ===
        # D. Layout Transformer Encoder
        self.layout_encoder = LayoutTransformerEncoder(
            input_dim=4,
            hidden_size=hidden_size,
            num_layers=2,
            nhead=4,
            dropout=dropout
        )
        
        # VAE Heads
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)
        
        # E. Latent Projector
        self.z_proj = nn.Linear(latent_dim, bb_size)
        # ==============================================
        
        # 6. KG Feature Projection
        self.kg_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 7. Decoder
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # Decoder 输出
        decoder_output_size = hidden_size + bb_size
        
        # 8. Prediction Head
        self.reg_head = nn.Sequential(
            nn.Linear(decoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )

    def reparameterize(self, mu, logvar):
        """CVAE Reparameterization Trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def construct_spatial_bias(self, cls_ids, kg_spatial_matrix):
        """构建空间注意力偏置"""
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
                kg_spatial_matrix=None, location_grids=None, target_boxes=None):
        """
        Query-Based Forward Pass (CVAE + GNN Enhanced).
        """
        
        # 1. Text Encoding
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 

        # 2. Query Embedding (Content)
        content_embed = self.obj_class_embedding(kg_class_ids) # [B, T, bb_size]
        
        # === 3. CVAE Logic: Latent Z Injection ===
        if target_boxes is not None:
            # [Training Mode]: 使用 Transformer Encode GT Layout -> Z
            # padding_mask: True 为 padding
            global_layout_feat = self.layout_encoder(target_boxes, mask=padding_mask) 
            
            mu = self.mu_head(global_layout_feat)
            logvar = self.logvar_head(global_layout_feat)
            z = self.reparameterize(mu, logvar)
        else:
            # [Inference Mode]: Sample Z from Prior N(0, I)
            B = input_ids.shape[0]
            mu = None
            logvar = None
            z = torch.randn(B, self.latent_dim, device=input_ids.device)
            
        z_feat = self.z_proj(z).unsqueeze(1) # [B, 1, bb_size]
        # ===============================================

        # 4. Position Features Construction
        pos_feat = torch.zeros_like(content_embed)
        
        # A. Handcrafted Grid
        if location_grids is not None:
            B, T, H, W = location_grids.shape
            grid_flat = location_grids.view(B, T, -1).to(content_embed.device)
            handcrafted_pos = self.grid_encoder(grid_flat) 
            pos_feat = pos_feat + handcrafted_pos

        # B. [升级] GNN Relation Prior (R-GAT)
        if kg_spatial_matrix is not None:
            B, T = kg_class_ids.shape
            map_ids = kg_class_ids - 2 
            gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
            b_idx = torch.arange(B, device=kg_class_ids.device).view(B, 1, 1).expand(-1, T, T)
            row_idx = gather_ids.view(B, T, 1).expand(-1, -1, T)
            col_idx = gather_ids.view(B, 1, T).expand(-1, T, -1)
            seq_rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] # [B, T, T]
            
            # 使用 GNN 进行推理: 输入 content_embed (语义) + 关系矩阵
            learned_pos = self.gnn_prior(content_embed, seq_rel_ids)
            
            pos_feat = pos_feat + learned_pos 

        # C. Fusion
        layout_embed = self.fusion_gate(content_embed, pos_feat)
        layout_embed = layout_embed + z_feat 

        # 5. Masks
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if padding_mask is not None:
            trg_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            trg_mask = trg_mask.masked_fill(padding_mask, float('-inf'))
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)
        else:
            trg_mask = None

        # 6. Spatial Bias
        spatial_bias = self.construct_spatial_bias(kg_class_ids, kg_spatial_matrix)

        # 7. Decoder
        decoder_output = self.layout_decoder(
            layout_embed, 
            text_encoded, 
            src_mask, 
            trg_mask,
            spatial_bias=spatial_bias
        ) 

        # 8. Prediction
        pred_boxes = torch.sigmoid(self.reg_head(decoder_output))
        
        return mu, logvar, pred_boxes, decoder_output

    def get_loss(self, pred_cls, pred_bbox_ids, pred_boxes, pred_count, layout_seq, layout_mask, num_boxes, target_coords_gt=None, kg_spatial_matrix=None, kg_class_weights=None):
        """计算重建损失 (V5.0: 加入审美 Loss)"""
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

        # 7. [NEW] Aesthetic Losses
        loss_alignment = self._compute_alignment_loss(pred_boxes, loss_mask)
        loss_balance = self._compute_balance_loss(pred_boxes, loss_mask)

        # Total Loss
        total_loss = self.reg_loss_weight * loss_reg + \
                     self.iou_loss_weight * loss_iou + \
                     self.area_loss_weight * loss_area + \
                     self.relation_loss_weight * loss_relation + \
                     self.overlap_loss_weight * loss_overlap + \
                     self.size_loss_weight * loss_size_prior + \
                     self.alignment_loss_weight * loss_alignment + \
                     self.balance_loss_weight * loss_balance
                     
        return total_loss, loss_relation, loss_overlap, \
               loss_reg, loss_iou, loss_size_prior, loss_area, \
               loss_alignment, loss_balance

    # === 原有 Loss 计算函数 ===
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
        pred_areas = pred_boxes[..., 2] * pred_boxes[..., 3] 
        if num_boxes is None:
            N = mask.sum(dim=1).clamp(min=1).float() 
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

    # === [NEW] Aesthetic Loss Functions ===
    
    def _compute_alignment_loss(self, pred_boxes, mask):
        """
        [审美] 对齐损失：鼓励框的边缘与其他框对齐。
        """
        B, N, _ = pred_boxes.shape
        loss = torch.tensor(0.0, device=pred_boxes.device)
        count = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            num_valid = len(valid_indices)
            if num_valid < 2: continue
            
            boxes = pred_boxes[b, valid_indices] # [K, 4]
            
            # 提取关键坐标
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2
            
            x_vals = torch.stack([left, cx, right], dim=1) # [K, 3]
            y_vals = torch.stack([top, cy, bottom], dim=1) # [K, 3]
            
            def min_dist_loss(vals):
                v1 = vals.unsqueeze(1) # [K, 1, 3]
                v2 = vals.unsqueeze(0) # [1, K, 3]
                diff = torch.abs(v1.unsqueeze(3) - v2.unsqueeze(2)).view(num_valid, num_valid, -1)
                eye_mask = torch.eye(num_valid, device=vals.device).bool().unsqueeze(-1)
                diff = diff.masked_fill(eye_mask, 100.0)
                min_dists, _ = diff.min(dim=2) 
                min_dists, _ = min_dists.min(dim=1) 
                return min_dists.mean()

            loss += min_dist_loss(x_vals) + min_dist_loss(y_vals)
            count += 1
            
        if count > 0:
            return loss / count
        return loss

    def _compute_balance_loss(self, pred_boxes, mask):
        """
        [审美] 平衡损失：鼓励布局重心接近中心。
        """
        B, N, _ = pred_boxes.shape
        loss = torch.tensor(0.0, device=pred_boxes.device)
        count = 0
        target_center = 0.5
        
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) == 0: continue
            
            boxes = pred_boxes[b, valid_indices]
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            areas = w * h
            total_area = areas.sum().clamp(min=1e-6)
            
            center_x = (cx * areas).sum() / total_area
            center_y = (cy * areas).sum() / total_area
            
            dist_sq = (center_x - target_center)**2 + (center_y - target_center)**2
            loss += dist_sq
            count += 1
            
        if count > 0:
            return loss / count
        return loss