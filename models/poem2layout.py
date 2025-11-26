# File: tianyusun1/test2/test2-2.0/models/poem2layout.py (FIXED)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .embedding import PoemLayoutEmbedding
from .decoder import LayoutDecoder

class Poem2LayoutGenerator(nn.Module):
    # num_classes: 实际的布局元素类别数 (例如：9)
    def __init__(self, bert_path: str, num_classes: int, num_bbox_bins: int, bbox_embed_dim: int, hidden_size: int = 768, bb_size: int = 64, decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, coord_loss_weight: float = 1.0, iou_loss_weight: float = 0.1, reg_loss_weight: float = 1.0, cls_loss_weight: float = 1.0, count_loss_weight: float = 1.0, area_loss_weight: float = 1.0, class_weights: torch.Tensor = None): 
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes 
        self.num_bbox_bins = num_bbox_bins 
        self.bbox_embed_dim = bbox_embed_dim 
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.coord_loss_weight = coord_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.reg_loss_weight = reg_loss_weight 
        self.cls_loss_weight = cls_loss_weight 
        self.count_loss_weight = count_loss_weight 
        self.area_loss_weight = area_loss_weight 
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Special tokens: BOS (0), EOS (1)
        self.bos_token_id = 0
        self.eos_token_id = 1
        total_vocab_size = self.num_element_classes + 2 
        
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # KG 特征投影层
        self.kg_projection = nn.Sequential(
            nn.Linear(self.num_element_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # [NEW] 空间关系偏置嵌入层
        # 输入: 关系ID (0-6), 输出: 一个向量 (长度=decoder_heads)，用于添加到 Attention Scores
        # num_relations = 7 (none, above, below, inside, surrounds, on_top, near)
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)
        
        # 数量预测头
        self.count_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1) 
        )
        
        # 实例化 Layout Embedding
        self.layout_embedding = PoemLayoutEmbedding(
            total_vocab_size=total_vocab_size, 
            num_bbox_bins=num_bbox_bins,
            bb_size=bb_size,
            cls_embed_dim=bb_size - 4 * bbox_embed_dim, 
            bbox_embed_dim=bbox_embed_dim
        )
        
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        decoder_output_size = hidden_size + bb_size
        
        # 预测头
        self.cls_head = nn.Linear(decoder_output_size, self.num_element_classes + 1) 
        
        self.cx_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.cy_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.w_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.h_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        
        self.reg_head = nn.Linear(decoder_output_size, 4)

    # --- NEW: 提取出来的空间偏置构建逻辑，供 forward 和 inference 复用 ---
    def construct_spatial_bias(self, cls_ids, kg_spatial_matrix):
        """
        根据当前生成的类别序列和全局空间矩阵构建 Attention Bias。
        Args:
            cls_ids: [B, S] 当前序列的类别 ID (原始 2-10, BOS/EOS)
            kg_spatial_matrix: [B, 9, 9] 知识图谱空间矩阵
        Returns:
            spatial_bias: [B, num_heads, S, S] or None
        """
        if kg_spatial_matrix is None:
            return None
            
        B, S = cls_ids.shape
        
        # 1. 映射 cls_ids (2-10) 到 KG 索引 (0-8)
        # BOS(0) -> -2, EOS(1) -> -1. 只有 >=0 的才是有效物体
        map_ids = cls_ids - 2 
        
        # 2. 为 gather 准备索引 (Clamp 负值以避免越界，稍后 mask 掉)
        gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1) # [B, S]
        
        # 3. Gather 操作
        b_idx = torch.arange(B, device=cls_ids.device).view(B, 1, 1).expand(-1, S, S)
        row_idx = gather_ids.view(B, S, 1).expand(-1, -1, S)
        col_idx = gather_ids.view(B, 1, S).expand(-1, S, -1)
        
        # [B, S, S] - 获取每对 Token 之间的关系 ID
        rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] 
        
        # 4. Mask 掉无效元素 (BOS/EOS)
        is_valid_obj = (map_ids >= 0) # [B, S]
        valid_pair_mask = is_valid_obj.unsqueeze(2) & is_valid_obj.unsqueeze(1) # [B, S, S]
        rel_ids = rel_ids.masked_fill(~valid_pair_mask, 0) # 0 是 'none' 关系
        
        # 5. Embedding & Reshape
        spatial_bias = self.spatial_bias_embedding(rel_ids) # [B, S, S, num_heads]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2).contiguous() # [B, num_heads, S, S]
        
        return spatial_bias

    # **修改: forward 增加 kg_spatial_matrix 参数**
    def forward(self, input_ids, attention_mask, layout_seq, kg_vectors, kg_spatial_matrix=None):
        # layout_seq: [B, S]
        batch_size, seq_len = layout_seq.shape
        if seq_len % 5 != 0:
            raise ValueError(f"Layout sequence length {seq_len} must be a multiple of 5.")
        num_elements = seq_len // 5

        # 1. BERT 文本特征
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 
        
        # 数量预测
        pred_count = self.count_head(text_encoded[:, 0, :]) 

        # 2. KG 特征注入
        kg_feat = self.kg_projection(kg_vectors) 
        text_encoded = text_encoded + kg_feat.unsqueeze(1)

        # Reshape layout_seq
        reshaped_seq = layout_seq.view(batch_size, num_elements, 5)
        cls_ids = reshaped_seq[:, :, 0].long()      # [B, num_elements]
        bbox_ids = reshaped_seq[:, :, 1:5].long()   # [B, num_elements, 4]

        # Layout Embedding
        layout_embed = self.layout_embedding(cls_ids, bbox_ids) 

        # Masks
        trg_mask = self.generate_square_subsequent_mask(num_elements).to(layout_embed.device)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) 

        # =================================================
        # [NEW] 构建空间 Attention Bias (调用 helper 方法)
        # =================================================
        spatial_bias = self.construct_spatial_bias(cls_ids, kg_spatial_matrix)

        # 传入 spatial_bias 到解码器
        decoder_output = self.layout_decoder(
            layout_embed, 
            text_encoded, 
            src_mask, 
            trg_mask,
            spatial_bias=spatial_bias # <<< NEW Arg
        ) 

        # Predictions
        pred_cls = self.cls_head(decoder_output) 
        
        pred_cx_id = self.cx_head(decoder_output) 
        pred_cy_id = self.cy_head(decoder_output)
        pred_w_id = self.w_head(decoder_output)
        pred_h_id = self.h_head(decoder_output)
        
        pred_bbox_ids = torch.stack([pred_cx_id, pred_cy_id, pred_w_id, pred_h_id], dim=2) 
        pred_coord_float = torch.sigmoid(self.reg_head(decoder_output)) 

        return pred_cls, pred_bbox_ids, pred_coord_float, pred_count

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _detokenize_target_ids(self, target_bbox_ids: torch.Tensor) -> torch.Tensor:
        num_bins = self.num_bbox_bins
        range_divisor = num_bins - 1
        detok_boxes = target_bbox_ids.float() / range_divisor
        return detok_boxes

    def _detokenize_pred_ids(self, pred_bbox_ids: torch.Tensor) -> torch.Tensor:
        pred_ids = pred_bbox_ids.argmax(dim=-1) 
        num_bins = self.num_bbox_bins
        range_divisor = num_bins - 1
        detok_boxes = pred_ids.float() / range_divisor
        return detok_boxes

    def _compute_iou_loss(self, pred_boxes: torch.Tensor, valid_mask: torch.Tensor, iou_threshold: float = 0.5):
        B, S, _ = pred_boxes.shape
        w_half = pred_boxes[..., 2:3] / 2
        h_half = pred_boxes[..., 3:4] / 2
        boxes = torch.cat([
            pred_boxes[..., 0:1] - w_half, 
            pred_boxes[..., 1:2] - h_half, 
            pred_boxes[..., 0:1] + w_half, 
            pred_boxes[..., 1:2] + h_half 
        ], dim=-1).clamp(0., 1.) 

        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]) 
        area_i = area.unsqueeze(2) 
        area_j = area.unsqueeze(1) 
        
        xmin_max = torch.max(boxes.unsqueeze(2)[..., 0], boxes.unsqueeze(1)[..., 0]) 
        ymin_max = torch.max(boxes.unsqueeze(2)[..., 1], boxes.unsqueeze(1)[..., 1]) 
        xmax_min = torch.min(boxes.unsqueeze(2)[..., 2], boxes.unsqueeze(1)[..., 2]) 
        ymax_min = torch.min(boxes.unsqueeze(2)[..., 3], boxes.unsqueeze(1)[..., 3]) 

        inter_w = (xmax_min - xmin_max).clamp(min=0) 
        inter_h = (ymax_min - ymin_max).clamp(min=0) 
        intersection = inter_w * inter_h 

        union = area_i + area_j - intersection 
        iou = intersection / (union + 1e-6) 
        
        valid_mask_i = valid_mask.unsqueeze(2) 
        valid_mask_j = valid_mask.unsqueeze(1) 
        pairwise_mask = valid_mask_i & valid_mask_j 
        
        diag_mask = torch.eye(S, device=iou.device).bool().unsqueeze(0).expand(B, -1, -1) 
        pairwise_mask = pairwise_mask & (~diag_mask) 
        
        penalty = (iou - iou_threshold).clamp(min=0.) ** 2 
        
        masked_penalty = penalty * pairwise_mask.float()
        num_valid_pairs = pairwise_mask.sum().clamp(min=1) 
        iou_loss = masked_penalty.sum() / num_valid_pairs

        return iou_loss

    def get_loss(self, pred_cls, pred_bbox_ids, pred_coord_float, pred_count, target_layout_seq, target_layout_mask, target_num_boxes, target_coords_gt=None):
        batch_size, seq_len_decoded = target_layout_seq.shape 
        num_elements_decoded = seq_len_decoded // 5 

        reshaped_target = target_layout_seq.view(batch_size, num_elements_decoded, 5)
        target_cls_ids = reshaped_target[:, :, 0].long()
        target_bbox_ids = reshaped_target[:, :, 1:5].long()

        reshaped_mask = target_layout_mask.view(batch_size, num_elements_decoded, 5)
        cls_mask = reshaped_mask[:, :, 0].bool() 

        pred_seq_len = pred_cls.size(1)
        target_seq_len = target_cls_ids.size(1)

        if pred_seq_len != target_seq_len:
            min_seq_len = min(pred_seq_len, target_seq_len)
            pred_cls = pred_cls[:, :min_seq_len, :] 
            pred_bbox_ids = pred_bbox_ids[:, :min_seq_len, :, :] 
            pred_coord_float = pred_coord_float[:, :min_seq_len, :] 
            target_cls_ids = target_cls_ids[:, :min_seq_len] 
            target_bbox_ids = target_bbox_ids[:, :min_seq_len, :] 
            cls_mask = cls_mask[:, :min_seq_len]
            # [NEW] Handle GT float target truncation
            if target_coords_gt is not None:
                target_coords_gt = target_coords_gt[:, :min_seq_len, :]
            
        # 1. Cls Loss
        valid_cls_target_mask = (target_cls_ids >= 1) & (target_cls_ids <= 10) 
        target_cls_ids_mapped = torch.clamp(target_cls_ids - 1, min=0, max=self.num_element_classes) 
        
        final_cls_mask = cls_mask & valid_cls_target_mask 
        final_cls_mask_flat = final_cls_mask.view(-1)
        
        pred_cls_flat = pred_cls.view(-1, pred_cls.size(-1)) 
        target_cls_ids_flat = target_cls_ids_mapped.view(-1) 
        
        pred_cls_valid = pred_cls_flat[final_cls_mask_flat]
        target_cls_valid = target_cls_ids_flat[final_cls_mask_flat]
        
        cls_loss_per_element = F.cross_entropy(
            pred_cls_valid, 
            target_cls_valid, 
            reduction='none'
        )
        
        if self.class_weights is not None:
            weights = self.class_weights[target_cls_valid] 
            cls_loss_weighted = cls_loss_per_element * weights
        else:
            cls_loss_weighted = cls_loss_per_element
        
        cls_loss = cls_loss_weighted.sum() / final_cls_mask_flat.sum().clamp(min=1) 

        # 2. Coord Loss
        valid_coord_target_mask = (target_cls_ids >= 2) & (target_cls_ids <= 10) 
        coord_mask_flat = (cls_mask & valid_coord_target_mask).unsqueeze(-1).expand(-1, -1, 4).flatten() 
        
        pred_bbox_flat = pred_bbox_ids.flatten(0, 2) 
        target_bbox_flat = target_bbox_ids.flatten()
        
        pred_bbox_valid = pred_bbox_flat[coord_mask_flat]
        target_bbox_valid = target_bbox_flat[coord_mask_flat]
        
        coord_loss = F.cross_entropy(pred_bbox_valid, target_bbox_valid, reduction='sum')
        num_valid_tokens = (cls_mask & valid_coord_target_mask).sum().clamp(min=1) * 4 
        coord_loss = coord_loss / num_valid_tokens
        
        # 3. Reg Loss (MODIFIED: Use GT float if available)
        if target_coords_gt is not None:
            target_coord_float = target_coords_gt.detach()
        else:
            # Fallback for inference/validation if gt not provided
            target_coord_float = self._detokenize_target_ids(target_bbox_ids) 
            
        reg_loss_per_coord = F.smooth_l1_loss(pred_coord_float, target_coord_float, reduction='none') 
        reg_loss_mask = (cls_mask & valid_coord_target_mask).unsqueeze(-1).float() 
        reg_loss = reg_loss_per_coord * reg_loss_mask
        reg_loss = reg_loss.sum() / num_valid_tokens
        
        # 4. IoU Loss
        iou_loss = self._compute_iou_loss(
            pred_boxes=pred_coord_float, 
            valid_mask=(cls_mask & valid_coord_target_mask), 
            iou_threshold=0.5
        )

        # 5. Area Loss (MODIFIED: Use GT float)
        
        # Calculate target area (w * h)
        target_w = target_coord_float[..., 2]
        target_h = target_coord_float[..., 3]
        target_area = target_w * target_h
        
        # Calculate predicted area (w * h)
        pred_w = pred_coord_float[..., 2]
        pred_h = pred_coord_float[..., 3]
        pred_area = pred_w * pred_h 

        # Use Smooth L1 to penalize the difference between predicted and target area
        area_loss_per_element = F.smooth_l1_loss(pred_area, target_area.detach(), reduction='none') 

        # Reuse existing masks for valid elements
        area_loss_mask = (cls_mask & valid_coord_target_mask).float() 
        num_valid_elements = area_loss_mask.sum().clamp(min=1)
        
        area_loss = area_loss_per_element * area_loss_mask
        area_loss = area_loss.sum() / num_valid_elements
        
        # 6. Count Loss
        target_count = target_num_boxes.float().unsqueeze(1) 
        count_loss = F.smooth_l1_loss(pred_count, target_count)
        
        # 7. Total Loss
        total_loss = (self.cls_loss_weight * cls_loss) + \
                     (self.coord_loss_weight * coord_loss) + \
                     (self.reg_loss_weight * reg_loss) + \
                     (self.iou_loss_weight * iou_loss) + \
                     (self.count_loss_weight * count_loss) + \
                     (self.area_loss_weight * area_loss) 
        
        return total_loss, cls_loss, coord_loss, reg_loss, iou_loss, count_loss, area_loss