# File: tianyusun1/test2/test2-4.0/models/poem2layout.py (V4.5: GNN + TRANSFORMER ENCODER)

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

# === [NEW] 2. 图神经网络关系先验 (GNN Relation Prior) ===
# [升级] 替换了原有的简单 RelationPriorNet
class GraphRelationPriorNet(nn.Module):
    """
    使用 RGCN (关系图卷积) 进行空间推理。
    它接收物体的语义特征 (Content Embed) 和关系矩阵，
    通过消息传递机制推导出每个物体的隐含位置。
    """
    def __init__(self, num_relations, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # 核心：为每种关系定义一个独立的变换矩阵
        # Shape: [num_relations, input_dim, hidden_dim]
        self.rel_weights = nn.Parameter(torch.Tensor(num_relations, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.rel_weights)
        
        # 处理自身的变换 (Self-loop)
        self.self_weight = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features, spatial_matrix):
        """
        Args:
            node_features: [B, T, D] (Content Embeddings，即“我是什么”)
            spatial_matrix: [B, T, T] (Relation IDs，即“我们什么关系”)
        Returns:
            learned_pos: [B, T, D]
        """
        B, T, _ = node_features.shape
        
        # 初始化聚合结果
        agg_features = torch.zeros(B, T, self.hidden_dim, device=node_features.device)
        
        # === 核心逻辑：按关系类型聚合消息 (Message Passing) ===
        # 遍历每一种关系 r (从1开始，0是无关系/Self)
        for r in range(1, self.num_relations):
            # 1. 找出当前 batch 中所有属于关系 r 的边
            mask = (spatial_matrix == r).float() # [B, T, T]
            
            # 归一化因子 (计算度数)
            degree = mask.sum(dim=2, keepdim=True).clamp(min=1e-6)
            mask_norm = mask / degree
            
            # 2. 聚合邻居特征
            # [B, T, T] x [B, T, D] -> [B, T, D]
            neighbors = torch.bmm(mask_norm, node_features)
            
            # 3. 应用关系特有的变换权重 W_r
            w_r = self.rel_weights[r] # [D, D]
            transformed = torch.matmul(neighbors, w_r)
            
            # 4. 累加到总特征
            agg_features = agg_features + transformed
            
        # === 加上自身特征 (Self-Loop) ===
        self_feat = self.self_weight(node_features)
        output = agg_features + self_feat
        
        # 后处理
        output = self.activation(self.norm(output))
        return self.dropout(output)

# === [NEW] 3. 布局变换编码器 (Layout Transformer Encoder) ===
# [升级] 替换了 CVAE 中简单的 MLP Encoder
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
        
        # B. [升级] GNN 关系先验网络
        # 输入维度为 bb_size (因为输入的是 content_embed)
        self.gnn_prior = GraphRelationPriorNet(
            num_relations=self.num_spatial_relations,
            input_dim=bb_size, 
            hidden_dim=bb_size,
            dropout=dropout
        )
        
        # C. 门控融合
        self.fusion_gate = GatedFusion(bb_size)

        # === 5. CVAE Components ===
        # D. [升级] Layout Transformer Encoder
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

        # B. [升级] GNN Relation Prior (RGCN)
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
        """计算重建损失 (保持不变)"""
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