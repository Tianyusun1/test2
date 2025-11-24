import torch
import numpy as np
import math
# Note: We assume Poem2LayoutGenerator and its dependencies are importable
# via sys.path manipulation in the calling script (e.g., scripts/infer.py)
# from models.poem2layout import Poem2LayoutGenerator # Import is implicit

def generate_square_subsequent_mask(sz, device='cuda'):
    """Generates a square subsequent mask for causal attention."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements: int = 20, device: str = 'cuda'):
    """
    Greedy decoding to generate a layout sequence from a poem.
    
    Args:
        model: Trained Poem2LayoutGenerator model. Requires attributes:
               .text_encoder, .layout_embedding, .layout_decoder, .cls_head, .coord_head,
               .bos_token_id (0), .eos_token_id (1).
        tokenizer: BERT tokenizer
        poem: Input poem string
        max_elements: Maximum number of elements to generate
        device: Device to run on
    Returns:
        layout: List of (cls_id, cx, cy, w, h), where cls_id is 2-10 range.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    # 1. Setup text encoding and masks
    with torch.no_grad():
        inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Encode text
        text_features = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_text]

        # 2. Initialize sequences
        batch_size = 1
        # Start with BOS token (ID 0)
        generated_cls_ids = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
        # BBoxes for BOS token are placeholders (all zero)
        generated_bboxes = torch.zeros((batch_size, 1, 4), dtype=torch.float, device=device)

        # 3. Autoregressive Loop
        for step in range(max_elements):
            current_num_elements = generated_cls_ids.size(1)

            # --- Forward Pass ---
            layout_embed = model.layout_embedding(generated_cls_ids, generated_bboxes)
            trg_mask = generate_square_subsequent_mask(current_num_elements, device=device) # [T, T]
            decoder_output = model.layout_decoder(layout_embed, text_features, src_mask, trg_mask)
            
            # Get predictions for the last element (which is the prediction for the T-th step, based on T-1 inputs)
            last_output = decoder_output[:, -1, :] 
            next_cls_logits = model.cls_head(last_output)
            next_coord = model.coord_head(last_output)

            # Greedy selection: ArgMax
            next_cls_id = next_cls_logits.argmax(dim=-1) # [B]

            # --- Stopping Condition (EOS) ---
            # EOS check: If the predicted class is the EOS token (ID 1), stop.
            is_eos = (next_cls_id == model.eos_token_id)
            if is_eos.all():
                break

            # Update sequences
            generated_cls_ids = torch.cat([generated_cls_ids, next_cls_id.unsqueeze(1)], dim=1)
            generated_bboxes = torch.cat([generated_bboxes, next_coord.unsqueeze(1)], dim=1)

    # 4. Final Extraction and Mapping
    # Skip BOS token (first element) and include up to the last generated element
    final_cls_ids = generated_cls_ids[0, 1:] 
    final_bboxes = generated_bboxes[0, 1:]

    layout = []
    for cls_id_tensor, (cx, cy, w, h) in zip(final_cls_ids, final_bboxes):
        # The classification head outputs class indices 0-8 for the elements.
        internal_cls_id = cls_id_tensor.item()
        
        # Map back to original IDs: 0->2, 1->3, ..., 8->10
        # CRITICAL: We only map the real layout elements (0-8)
        if internal_cls_id < 0 or internal_cls_id >= model.num_element_classes:
             # Should not happen if EOS is handled correctly, but good for safety
             continue 
        
        # Map back to original IDs (2-10).
        original_cls_id = internal_cls_id + 2
        
        layout.append((original_cls_id, cx.item(), cy.item(), w.item(), h.item()))

    return layout

# Keep auxiliary function for mask generation
def generate_square_subsequent_mask(sz, device='cuda'):
    """Generates a square subsequent mask for causal attention."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask