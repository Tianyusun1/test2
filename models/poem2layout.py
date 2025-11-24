import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .embedding import PoemLayoutEmbedding
from .decoder import LayoutDecoder

class Poem2LayoutGenerator(nn.Module):
    # num_classes: 实际的布局元素类别数 (例如：9)
    def __init__(self, bert_path: str, num_classes: int, num_bbox_bins: int, bbox_embed_dim: int, hidden_size: int = 768, bb_size: int = 64, decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, coord_loss_weight: float = 1.0, iou_loss_weight: float = 0.1, class_weights: torch.Tensor = None):
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes 
        self.num_bbox_bins = num_bbox_bins 
        self.bbox_embed_dim = bbox_embed_dim 
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.coord_loss_weight = coord_loss_weight
        self.iou_loss_weight = iou_loss_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Special tokens: BOS (0), EOS (1)
        self.bos_token_id = 0
        self.eos_token_id = 1
        total_vocab_size = self.num_element_classes + 2 
        
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # --- NEW: KG 特征投影层 ---
        # 将 9维 的视觉向量投影到 BERT 的 hidden_size (768)
        # num_element_classes = 9
        self.kg_projection = nn.Sequential(
            nn.Linear(self.num_element_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # -------------------------
        
        # 实例化 Layout Embedding，传入所有 BBox 相关的配置参数
        self.layout_embedding = PoemLayoutEmbedding(
            total_vocab_size=total_vocab_size, 
            num_bbox_bins=num_bbox_bins,
            bb_size=bb_size,
            cls_embed_dim=bb_size - 4 * bbox_embed_dim, # 动态计算类别嵌入维度
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
        
        # 分类头仍预测布局元素的数量 (9 类)
        self.cls_head = nn.Linear(decoder_output_size, self.num_element_classes)
        
        # 4 个 BBox 坐标的分类头
        self.cx_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.cy_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.w_head = nn.Linear(decoder_output_size, self.num_bbox_bins)
        self.h_head = nn.Linear(decoder_output_size, self.num_bbox_bins)

    # --- Updated Forward to accept kg_vectors ---
    def forward(self, input_ids, attention_mask, layout_seq, kg_vectors):
        # layout_seq: [B, S] 现包含整数 ID (类别 ID, cx_id, cy_id, w_id, h_id)
        batch_size, seq_len = layout_seq.shape
        if seq_len % 5 != 0:
            raise ValueError(f"Layout sequence length {seq_len} must be a multiple of 5.")
        num_elements = seq_len // 5

        # 1. 获取 BERT 文本特征 [B, L_text, H]
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 

        # 2. --- NEW: 注入 KG 特征 ---
        # kg_vectors: [B, 9] -> projection -> [B, H]
        kg_feat = self.kg_projection(kg_vectors) 
        
        # 将 KG 特征加到文本特征的每一个 token 上 (Broadcasting)
        # [B, L_text, H] + [B, 1, H] -> [B, L_text, H]
        text_encoded = text_encoded + kg_feat.unsqueeze(1)
        # ---------------------------

        # Reshape layout_seq to separate cls and 4 BBox token IDs (all Long)
        reshaped_seq = layout_seq.view(batch_size, num_elements, 5)
        cls_ids = reshaped_seq[:, :, 0].long()      # [B, num_elements]
        bbox_ids = reshaped_seq[:, :, 1:5].long()   # [B, num_elements, 4]

        # 使用 BBox IDs 进行嵌入
        layout_embed = self.layout_embedding(cls_ids, bbox_ids) # [B, num_elements, bb_size]

        trg_mask = self.generate_square_subsequent_mask(num_elements).to(layout_embed.device)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) 

        decoder_output = self.layout_decoder(layout_embed, text_encoded, src_mask, trg_mask) # [B, num_elements, H + bb_size]

        # Cls prediction
        pred_cls = self.cls_head(decoder_output) # [B, num_elements, num_element_classes]
        
        # 4 BBox Token Predictions (Classification)
        pred_cx_id = self.cx_head(decoder_output) # [B, T, num_bbox_bins]
        pred_cy_id = self.cy_head(decoder_output)
        pred_w_id = self.w_head(decoder_output)
        pred_h_id = self.h_head(decoder_output)
        
        # 堆叠预测结果，用于损失计算
        # 形状: [B, T, 4, num_bbox_bins]
        pred_bbox_ids = torch.stack([pred_cx_id, pred_cy_id, pred_w_id, pred_h_id], dim=2) 

        return pred_cls, pred_bbox_ids

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _detokenize_pred_ids(self, pred_bbox_ids: torch.Tensor) -> torch.Tensor:
        """
        将预测的 BBox Token Logits 转换回连续的浮点坐标。
        """
        # 1. 找到概率最高的 bin ID
        pred_ids = pred_bbox_ids.argmax(dim=-1) # [B, S, 4]
        
        # 2. 反量化公式
        num_bins = self.num_bbox_bins
        range_divisor = num_bins - 1
        
        detok_boxes = pred_ids.float() / range_divisor
        
        return detok_boxes

    def _compute_iou_loss(self, pred_boxes: torch.Tensor, valid_mask: torch.Tensor, iou_threshold: float = 0.5):
        """
        IoU 排斥损失计算。
        NOTE: pred_boxes 必须是浮点值 (detokenized)。
        """
        B, S, _ = pred_boxes.shape
        
        # 1. Convert cx,cy,w,h to xmin,ymin,xmax,ymax
        w_half = pred_boxes[..., 2:3] / 2
        h_half = pred_boxes[..., 3:4] / 2
        boxes = torch.cat([
            pred_boxes[..., 0:1] - w_half,  # xmin
            pred_boxes[..., 1:2] - h_half,  # ymin
            pred_boxes[..., 0:1] + w_half,  # xmax
            pred_boxes[..., 1:2] + h_half   # ymax
        ], dim=-1).clamp(0., 1.) # [B, S, 4]

        # 2. Compute pairwise IoU
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]) # [B, S]
        area_i = area.unsqueeze(2) # [B, S, 1]
        area_j = area.unsqueeze(1) # [B, 1, S]
        
        # Intersections
        xmin_max = torch.max(boxes.unsqueeze(2)[..., 0], boxes.unsqueeze(1)[..., 0]) # [B, S, S]
        ymin_max = torch.max(boxes.unsqueeze(2)[..., 1], boxes.unsqueeze(1)[..., 1]) # [B, S, S]
        xmax_min = torch.min(boxes.unsqueeze(2)[..., 2], boxes.unsqueeze(1)[..., 2]) # [B, S, S]
        ymax_min = torch.min(boxes.unsqueeze(2)[..., 3], boxes.unsqueeze(1)[..., 3]) # [B, S, S]

        inter_w = (xmax_min - xmin_max).clamp(min=0) # [B, S, S]
        inter_h = (ymax_min - ymin_max).clamp(min=0) # [B, S, S]
        intersection = inter_w * inter_h # [B, S, S]

        # Union
        union = area_i + area_j - intersection # [B, S, S]

        # IoU
        iou = intersection / (union + 1e-6) # [B, S, S]
        
        # 3. Apply mask and calculate penalty
        valid_mask_i = valid_mask.unsqueeze(2) # [B, S, 1]
        valid_mask_j = valid_mask.unsqueeze(1) # [B, 1, S]
        pairwise_mask = valid_mask_i & valid_mask_j # [B, S, S]
        
        # Mask out self-IoU (diagonal: i == j) which is always 1.
        diag_mask = torch.eye(S, device=iou.device).bool().unsqueeze(0).expand(B, -1, -1) # [B, S, S]
        pairwise_mask = pairwise_mask & (~diag_mask) 
        
        # Apply IoU Repulsion Penalty: penalty = max(0, IoU - threshold)^2
        penalty = (iou - iou_threshold).clamp(min=0.) ** 2 # [B, S, S]
        
        # Apply the final mask
        masked_penalty = penalty * pairwise_mask.float()
        
        # Average over all non-diagonal, valid pairs
        num_valid_pairs = pairwise_mask.sum().clamp(min=1) 
        iou_loss = masked_penalty.sum() / num_valid_pairs

        return iou_loss

    def get_loss(self, pred_cls, pred_bbox_ids, target_layout_seq, target_layout_mask):
        """
        Calculates the combined loss for class prediction and coordinate prediction,
        NOW using Cross-Entropy for 4 BBox token predictions.
        """
        batch_size, seq_len_decoded = target_layout_seq.shape 
        num_elements_decoded = seq_len_decoded // 5 

        # --- Reshape target sequence and mask ---
        reshaped_target = target_layout_seq.view(batch_size, num_elements_decoded, 5)
        
        # Target values are now token IDs (LongTensor)
        target_cls_ids = reshaped_target[:, :, 0].long()     # [B, S] (2-10)
        target_bbox_ids = reshaped_target[:, :, 1:5].long()  # [B, S, 4] (0-999)

        reshaped_mask = target_layout_mask.view(batch_size, num_elements_decoded, 5)
        cls_mask = reshaped_mask[:, :, 0].bool() # [B, S]

        # --- Defensive: Align sequence lengths ---
        pred_seq_len = pred_cls.size(1)
        target_seq_len = target_cls_ids.size(1)

        if pred_seq_len != target_seq_len:
            min_seq_len = min(pred_seq_len, target_seq_len)
            pred_cls = pred_cls[:, :min_seq_len, :] 
            pred_bbox_ids = pred_bbox_ids[:, :min_seq_len, :, :] # [B, min_S, 4, num_bins]
            target_cls_ids = target_cls_ids[:, :min_seq_len] 
            target_bbox_ids = target_bbox_ids[:, :min_seq_len, :] # [B, min_S, 4]
            cls_mask = cls_mask[:, :min_seq_len] 
            
        # === Core Processing: Map Target ID and Mask ===
        valid_cls_target_mask = (target_cls_ids >= 2) & (target_cls_ids <= 10) # [B, S]
        # Map original IDs (2-10) to internal IDs (0-8)
        target_cls_ids_mapped = torch.clamp(target_cls_ids - 2, min=0, max=self.num_element_classes - 1)
        final_cls_mask = cls_mask & valid_cls_target_mask # [B, S]
        
        # --- 1. Classification Loss (Cls ID - remains Cross-Entropy) ---
        pred_cls_flat = pred_cls.view(-1, pred_cls.size(-1)) 
        target_cls_ids_flat = target_cls_ids_mapped.view(-1) 
        final_cls_mask_flat = final_cls_mask.view(-1) 
        
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


        # --- 2. Coordinate Loss (BBox Token IDs - NOW Cross-Entropy) ---
        # Reshape for Cross-Entropy: pred should be [N, C] and target [N]
        
        # FIX: Flatten only B, S, and 4 dimensions to get [B*S*4, num_bins]
        pred_bbox_flat = pred_bbox_ids.flatten(0, 2) # [B*S*4, num_bins]

        # target_bbox_ids: [B, S, 4] -> [B*S*4]
        target_bbox_flat = target_bbox_ids.flatten()
        
        # Expand mask for all 4 coordinates
        # final_cls_mask: [B, S] -> [B, S, 4] -> [B*S*4]
        coord_mask_flat = final_cls_mask.unsqueeze(-1).expand(-1, -1, 4).flatten() 
        
        pred_bbox_valid = pred_bbox_flat[coord_mask_flat]
        target_bbox_valid = target_bbox_flat[coord_mask_flat]
        
        # BBox token loss is standard CE.
        coord_loss = F.cross_entropy(
            pred_bbox_valid,
            target_bbox_valid,
            reduction='sum'
        )
        # Normalize by the total number of valid BBox tokens (4 * num_valid_elements)
        num_valid_tokens = final_cls_mask_flat.sum().clamp(min=1) * 4 
        coord_loss = coord_loss / num_valid_tokens

        # --- 3. IoU Repulsion Loss (requires detokenization) ---
        
        # Get float coordinates by detokenizing the current prediction logits
        pred_boxes_float = self._detokenize_pred_ids(pred_bbox_ids) # [B, S, 4] float
        
        iou_loss = self._compute_iou_loss(
            pred_boxes=pred_boxes_float, # Use detokenized floats
            valid_mask=final_cls_mask,
            iou_threshold=0.5
        )
        
        # --- 4. Combine Losses ---
        total_loss = cls_loss + self.coord_loss_weight * coord_loss + self.iou_loss_weight * iou_loss
        
        return total_loss, cls_loss, coord_loss, iou_loss