import torch
import torch.nn as nn
from transformers import BertModel
from .embedding import PoemLayoutEmbedding
from .decoder import LayoutDecoder

class Poem2LayoutGenerator(nn.Module):
    # num_classes: 实际的布局元素类别数 (例如：9)
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, coord_loss_weight: float = 1.0, iou_loss_weight: float = 0.1, class_weights: torch.Tensor = None):
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes 
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.coord_loss_weight = coord_loss_weight
        # --- NEW: IoU Loss Weight ---
        self.iou_loss_weight = iou_loss_weight
        
        # --- NEW: Class Weights for Cross Entropy ---
        if class_weights is not None:
            # Move class weights to device when model is moved
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        # ------------------------------------------

        # Special tokens: BOS (0), EOS (1)
        self.bos_token_id = 0
        self.eos_token_id = 1
        
        # 类别总数 = 9 (元素) + 2 (BOS/EOS) = 11。
        total_vocab_size = self.num_element_classes + 2 
        
        self.text_encoder = BertModel.from_pretrained(bert_path)
        # 确保 PoemLayoutEmbedding 使用了完整的词汇表大小
        self.layout_embedding = PoemLayoutEmbedding(num_classes=total_vocab_size, bb_size=bb_size)
        
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        decoder_output_size = hidden_size + bb_size
        # 分类头仍应预测布局元素的数量 (9 类)
        self.cls_head = nn.Linear(decoder_output_size, self.num_element_classes)
        self.coord_head = nn.Sequential(
            nn.Linear(decoder_output_size, 4),
            nn.Sigmoid() # Ensure coords are in [0,1]
        )

    def forward(self, input_ids, attention_mask, layout_seq):
        # layout_seq: [B, S] where S = 5 * num_elements
        batch_size, seq_len = layout_seq.shape
        if seq_len % 5 != 0:
            raise ValueError(f"Layout sequence length {seq_len} must be a multiple of 5.")
        num_elements = seq_len // 5

        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # [B, L_text, H]

        # Reshape layout_seq to separate cls and bbox
        reshaped_seq = layout_seq.view(batch_size, num_elements, 5)
        cls_ids = reshaped_seq[:, :, 0].long() # [B, num_elements]
        bboxes = reshaped_seq[:, :, 1:5].float() # [B, num_elements, 4]

        layout_embed = self.layout_embedding(cls_ids, bboxes) # [B, num_elements, bb_size]

        # Create causal mask for self-attention on the layout sequence
        trg_mask = self.generate_square_subsequent_mask(num_elements).to(layout_embed.device) # [num_elements, num_elements]
        # Create mask for text-attention (to ignore padding in text sequence)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_text]

        decoder_output = self.layout_decoder(layout_embed, text_encoded, src_mask, trg_mask) # [B, num_elements, H + bb_size]

        pred_cls = self.cls_head(decoder_output) # [B, num_elements, num_classes]
        pred_coord = self.coord_head(decoder_output) # [B, num_elements, 4]

        return pred_cls, pred_coord

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _compute_iou_loss(self, pred_boxes: torch.Tensor, valid_mask: torch.Tensor, iou_threshold: float = 0.5):
        """
        Computes an IoU repulsion loss term to penalize excessive box overlap.
        (Implementation details remain the same as previous step for brevity)
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

    def get_loss(self, pred_cls, pred_coord, target_layout_seq, target_layout_mask):
        """
        Calculates the combined loss for class prediction and coordinate prediction,
        including a new IoU Repulsion Loss term.
        """
        batch_size, seq_len_decoded = target_layout_seq.shape 
        num_elements_decoded = seq_len_decoded // 5 

        # --- Reshape target sequence and mask ---
        reshaped_target = target_layout_seq.view(batch_size, num_elements_decoded, 5)
        target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements_decoded]
        target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements_decoded, 4]

        reshaped_mask = target_layout_mask.view(batch_size, num_elements_decoded, 5)
        cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements_decoded]

        # --- Defensive: Align sequence lengths ---
        pred_seq_len = pred_cls.size(1)
        target_seq_len = target_cls_ids.size(1)

        if pred_seq_len != target_seq_len:
            min_seq_len = min(pred_seq_len, target_seq_len)
            pred_cls = pred_cls[:, :min_seq_len, :] 
            pred_coord = pred_coord[:, :min_seq_len, :] 
            target_cls_ids = target_cls_ids[:, :min_seq_len] 
            target_coords = target_coords[:, :min_seq_len, :] 
            cls_mask = cls_mask[:, :min_seq_len] 

        # === 核心处理: 映射目标 ID 和掩码 ===
        valid_cls_target_mask = (target_cls_ids >= 2) & (target_cls_ids <= 10) # [B, S]
        target_cls_ids_mapped = torch.clamp(target_cls_ids - 2, min=0, max=self.num_element_classes - 1)
        final_cls_mask = cls_mask & valid_cls_target_mask # [B, S]
        
        # --- 1. Classification Loss (引入类别加权) ---
        pred_cls_flat = pred_cls.view(-1, pred_cls.size(-1)) 
        target_cls_ids_flat = target_cls_ids_mapped.view(-1) 
        final_cls_mask_flat = final_cls_mask.view(-1) 
        
        # 筛选出需要计算损失的预测和目标
        pred_cls_valid = pred_cls_flat[final_cls_mask_flat]
        target_cls_valid = target_cls_ids_flat[final_cls_mask_flat]
        
        # 使用 reduction='none' 获取每个有效元素的损失
        cls_loss_per_element = nn.functional.cross_entropy(
            pred_cls_valid, 
            target_cls_valid, 
            reduction='none'
        )
        
        # NEW: 应用类别权重 (治本措施)
        if self.class_weights is not None:
            # 根据目标类别 ID (0-8) 查找对应的权重
            weights = self.class_weights[target_cls_valid]
            cls_loss_weighted = cls_loss_per_element * weights
        else:
            cls_loss_weighted = cls_loss_per_element
        
        # 最终损失：加权损失求和并除以有效元素总数
        cls_loss = cls_loss_weighted.sum() / final_cls_mask_flat.sum().clamp(min=1) 


        # --- 2. Coordinate Loss ---
        coord_loss = nn.functional.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, S, 4]
        coord_mask = final_cls_mask.unsqueeze(-1).float() # [B, S, 1]
        
        coord_loss = coord_loss * coord_mask
        num_valid_coords = final_cls_mask.sum().clamp(min=1) * 4 
        coord_loss = coord_loss.sum() / num_valid_coords

        # --- 3. IoU Repulsion Loss (NEW - 治本措施) ---
        iou_loss = self._compute_iou_loss(
            pred_boxes=pred_coord,
            valid_mask=final_cls_mask,
            iou_threshold=0.5
        )
        
        # --- 4. Combine Losses ---
        total_loss = cls_loss + self.coord_loss_weight * coord_loss + self.iou_loss_weight * iou_loss
        
        # 返回所有损失，以便 Trainer 打印日志
        return total_loss, cls_loss, coord_loss, iou_loss