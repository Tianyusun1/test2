import torch
from transformers import BertTokenizer
# Note: We assume Poem2LayoutGenerator and its dependencies are importable
# via sys.path manipulation in the calling script (e.g., scripts/infer.py)
# from models.poem2layout import Poem2LayoutGenerator

def generate_square_subsequent_mask(sz, device='cuda'):
    """Generates a square subsequent mask for causal attention."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements: int = 20, device: str = 'cuda'):
    """
    Greedy decoding to generate a layout sequence from a poem.
    Args:
        model: Trained Poem2LayoutGenerator model
        tokenizer: BERT tokenizer
        poem: Input poem string
        max_elements: Maximum number of elements to generate
        device: Device to run on
    Returns:
        layout: List of (cls_id, cx, cy, w, h)
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Encode text
        text_features = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # [B, L_text, H]
        # Create source mask for text-attention
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_text]

        # Initialize sequences
        batch_size = 1
        # Assuming model has bos_token_id defined (e.g., model.bos_token_id = 0)
        generated_cls_ids = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
        generated_bboxes = torch.zeros((batch_size, 1, 4), dtype=torch.float, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_elements):
            current_num_elements = generated_cls_ids.size(1)

            # Embed current layout sequence
            layout_embed = model.layout_embedding(generated_cls_ids, generated_bboxes) # [B, T, bb_size]

            # Generate causal mask for self-attention on layout
            trg_mask = generate_square_subsequent_mask(current_num_elements, device=device) # [T, T]

            # Decode
            decoder_output = model.layout_decoder(layout_embed, text_features, src_mask, trg_mask) # [B, T, H + bb_size]

            # Get predictions for the last element
            last_output = decoder_output[:, -1, :] # [B, H + bb_size]
            next_cls_logits = model.cls_head(last_output) # [B, num_classes]
            next_coord = model.coord_head(last_output) # [B, 4]

            # Greedy selection: ArgMax
            next_cls_id = next_cls_logits.argmax(dim=-1) # [B]

            # Check for EOS (assuming model has eos_token_id defined, e.g., model.eos_token_id = 1)
            # Note: Since cls_ids are mapped internally (0-8), EOS token ID should also be mapped.
            # If original EOS was 1, internal EOS is 1-2=-1 (invalid). If original EOS was 10, internal EOS is 10-2=8.
            # Let's assume internal EOS ID is 8 (corresponding to original ID 10).
            # This is a strong assumption. A better way is to define an explicit internal EOS ID in the model.
            # For now, let's just use argmax and see if EOS appears naturally.
            # If you want to stop on a specific internal ID, define it clearly.
            # Let's assume internal EOS ID is 8 for this example.
            # If internal IDs are 0-8, and none of them represent EOS, then stopping condition is max_elements.
            # For now, let's remove EOS check and rely on max_elements.
            # is_eos = (next_cls_id == model.eos_token_id_internal) # Needs definition in model
            # finished |= is_eos
            # if finished.all():
            #     break

            # Update sequences
            generated_cls_ids = torch.cat([generated_cls_ids, next_cls_id.unsqueeze(1)], dim=1) # [B, T+1]
            generated_bboxes = torch.cat([generated_bboxes, next_coord.unsqueeze(1)], dim=1) # [B, T+1, 4]

        # Extract final generated sequence (skip BOS token)
        final_cls_ids = generated_cls_ids[0, 1:] # [num_generated_elements]
        final_bboxes = generated_bboxes[0, 1:] # [num_generated_elements, 4]

        layout = []
        for cls_id_tensor, (cx, cy, w, h) in zip(final_cls_ids, final_bboxes):
            # Convert tensor to int
            internal_cls_id = cls_id_tensor.item()
            # Map back from internal 0~8 to original 2~10
            original_cls_id = internal_cls_id + 2
            layout.append((original_cls_id, cx.item(), cy.item(), w.item(), h.item()))

        return layout
