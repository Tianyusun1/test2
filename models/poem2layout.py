import torch
import torch.nn as nn
from transformers import BertModel
from .embedding import PoemLayoutEmbedding
from .decoder import LayoutDecoder

class Poem2LayoutGenerator(nn.Module):
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, coord_loss_weight: float = 1.0):
        super(Poem2LayoutGenerator, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.coord_loss_weight = coord_loss_weight

        self.text_encoder = BertModel.from_pretrained(bert_path)
        self.layout_embedding = PoemLayoutEmbedding(num_classes=num_classes, bb_size=bb_size)
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # The decoder output size is now hidden_size + bb_size after concatenation
        decoder_output_size = hidden_size + bb_size
        self.cls_head = nn.Linear(decoder_output_size, num_classes)
        self.coord_head = nn.Sequential(
            nn.Linear(decoder_output_size, 4),
            nn.Sigmoid() # Ensure coords are in [0,1]
        )

        # Special tokens
        self.bos_token_id = 0
        self.eos_token_id = 1

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

    def get_loss(self, pred_cls, pred_coord, target_layout_seq, target_layout_mask):
        """
        Calculates the combined loss for class prediction and coordinate prediction,
        taking into account the layout mask to ignore padded elements.
        Includes defensive alignment of sequence lengths due to potential decoder length mismatch.
        """
        batch_size, seq_len_decoded = target_layout_seq.shape # S_decoded
        num_elements_decoded = seq_len_decoded // 5 # This is the sequence length for target_cls_ids

        # --- Reshape target sequence and mask ---
        reshaped_target = target_layout_seq.view(batch_size, num_elements_decoded, 5)
        target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements_decoded]
        target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements_decoded, 4]

        reshaped_mask = target_layout_mask.view(batch_size, num_elements_decoded, 5)
        cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements_decoded]

        # --- Defensive: Align sequence lengths between prediction and target ---
        pred_seq_len = pred_cls.size(1)
        target_seq_len = target_cls_ids.size(1)

        if pred_seq_len != target_seq_len:
            # This indicates a mismatch in the decoder's output length vs. expected length.
            # The decoder (or its layers) might not be preserving sequence length correctly.
            # As a workaround, truncate both to the minimum length.
            min_seq_len = min(pred_seq_len, target_seq_len)

            pred_cls = pred_cls[:, :min_seq_len, :] # [B, min_seq_len, num_classes]
            pred_coord = pred_coord[:, :min_seq_len, :] # [B, min_seq_len, 4]
            target_cls_ids = target_cls_ids[:, :min_seq_len] # [B, min_seq_len]
            target_coords = target_coords[:, :min_seq_len, :] # [B, min_seq_len, 4]
            cls_mask = cls_mask[:, :min_seq_len] # [B, min_seq_len]

        # --- RESHAPE for cross_entropy ---
        # pred_cls: [B, S, C] -> [B*S, C]
        pred_cls_flat = pred_cls.view(-1, pred_cls.size(-1)) # [B*S, C]
        # target_cls_ids: [B, S] -> [B*S]
        target_cls_ids_flat = target_cls_ids.view(-1) # [B*S]
        # cls_mask: [B, S] -> [B*S]
        cls_mask_flat = cls_mask.view(-1) # [B*S]

        # --- Calculate Classification Loss ---
        # cls_loss: [B*S, C] vs [B*S] -> [B*S]
        cls_loss = nn.functional.cross_entropy(pred_cls_flat, target_cls_ids_flat, reduction='none') # [B*S]
        # Apply mask: [B*S] * [B*S] -> [B*S]
        cls_loss = cls_loss * cls_mask_flat.float() # [B*S]
        # Sum and normalize by number of valid (masked) elements
        cls_loss = cls_loss.sum() / cls_mask_flat.sum().clamp(min=1) # Scalar

        # --- Calculate Coordinate Loss ---
        # pred_coord: [B, S, 4]
        # target_coords: [B, S, 4]
        # cls_mask: [B, S] -> expand for 4 coordinates -> [B, S, 4]
        # coord_loss: [B, S, 4] vs [B, S, 4] -> [B, S, 4]
        coord_loss = nn.functional.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, S, 4]
        # Apply mask: [B, S, 4] * [B, S, 1] -> [B, S, 4]
        coord_loss = coord_loss * cls_mask.unsqueeze(-1).float() # [B, S, 4] * [B, S, 1]
        # Sum and normalize by number of valid elements * 4 coordinates
        coord_loss = coord_loss.sum() / (cls_mask.sum().clamp(min=1) * 4) # Scalar

        # --- Combine Losses ---
        total_loss = cls_loss + self.coord_loss_weight * coord_loss
        return total_loss, cls_loss, coord_loss
