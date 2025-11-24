import torch
import numpy as np
import math
# Note: We assume Poem2LayoutGenerator and its dependencies are importable
# via sys.path manipulation in the calling script (e.g., scripts/infer.py)
# from models.poem2layout import Poem2LayoutGenerator # Import is implicit

# --- NEW: Import KG ---
# 假设 models/kg.py 位于搜索路径中
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Warning] Could not import PoetryKnowledgeGraph.")
# --------------------

# --- NEW: Import BBoxTokenizer ---
# 假设 data/utils 中的 BBoxTokenizer 可以被导入
try:
    from data.utils import BBoxTokenizer
except ImportError:
    # 临时处理，如果导入失败则打印警告
    print("[Warning] Could not import BBoxTokenizer. Make sure data/utils.py is accessible.")
    class BBoxTokenizer:
        def __init__(self, num_bins): pass
        def tokenize(self, value): return 0
        def detokenize(self, bin_id): return 0.0
# --------------------------------

def generate_square_subsequent_mask(sz, device='cuda'):
    """Generates a square subsequent mask for causal attention."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements: int = 20, device: str = 'cuda'):
    """
    Greedy decoding to generate a layout sequence from a poem (using discrete BBox tokens and KG).
    
    Args:
        model: Trained Poem2LayoutGenerator model. 
        # ... (rest of args)
    Returns:
        layout: List of (cls_id, cx, cy, w, h), where cls_id is 2-10 range.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    # --- Initialize BBox Tokenizer ---
    try:
        num_bins = model.num_bbox_bins
    except AttributeError:
        raise AttributeError("Model attribute 'num_bbox_bins' not found. Ensure Poem2LayoutGenerator is initialized correctly.")
        
    bbox_tokenizer = BBoxTokenizer(num_bins=num_bins)
    range_divisor = num_bins - 1
    # --------------------------------

    # --- NEW: 提取 KG 向量 (关键修改) ---
    # 1. 实例化 KG
    pkg = PoetryKnowledgeGraph() 
    # 2. 提取特征 (原始 9 维 CPU Tensor)
    kg_vector_raw = pkg.extract_visual_feature_vector(poem) 
    # 3. 调整形状并移动到设备 (供模型使用)
    kg_vector = kg_vector_raw.unsqueeze(0).to(device) 

    # NEW: 打印 KG 向量 (用于调试)
    print("\n-------------------- KG DEBUG (Inference) ---------------------")
    print(f"Inference Poem: '{poem}'")
    # 内部 ID 0-8 对应原始 ID 2-10 (2:mountain, 3:water, ..., 10:animal)
    print(f"KG Vector (ID 2-10): {kg_vector_raw.tolist()}") 
    print("---------------------------------------------------------------")
    # -------------------------------------

    # 1. Setup text encoding and masks
    with torch.no_grad():
        inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Encode text: [B, L_text, H]
        text_features = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # --- NEW: 注入 KG 特征 (与 model.forward 逻辑一致) ---
        # kg_feat: [1, 9] -> projection -> [1, H]
        kg_feat = model.kg_projection(kg_vector) # <--- 使用移动到设备的 kg_vector
        # Augment text features: [1, L_text, H] + [1, 1, H] -> [1, L_text, H]
        text_features = text_features + kg_feat.unsqueeze(1)
        # -----------------------------------------------------
        
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_text]

        # 2. Initialize sequences
        batch_size = 1
        generated_cls_ids = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
        generated_bboxes = torch.zeros((batch_size, 1, 4), dtype=torch.long, device=device) 

        # 3. Autoregressive Loop
        for step in range(max_elements):
            current_num_elements = generated_cls_ids.size(1)

            # --- Forward Pass ---
            layout_embed = model.layout_embedding(generated_cls_ids, generated_bboxes) 
            trg_mask = generate_square_subsequent_mask(current_num_elements, device=device) # [T, T]
            
            # 使用增强后的 text_features
            decoder_output = model.layout_decoder(layout_embed, text_features, src_mask, trg_mask)
            
            # Get predictions for the last element 
            last_output = decoder_output[:, -1, :] 
            next_cls_logits = model.cls_head(last_output)
            
            # Get BBox Token Logits from 4 heads
            next_cx_logits = model.cx_head(last_output)
            next_cy_logits = model.cy_head(last_output)
            next_w_logits = model.w_head(last_output)
            next_h_logits = model.h_head(last_output)

            # Greedy selection for classes
            next_cls_id = next_cls_logits.argmax(dim=-1) # [B]
            
            # Greedy selection for BBox IDs (Argmax on Logits)
            next_cx_id = next_cx_logits.argmax(dim=-1) # [B]
            next_cy_id = next_cy_logits.argmax(dim=-1)
            next_w_id = next_w_logits.argmax(dim=-1)
            next_h_id = next_h_logits.argmax(dim=-1)
            
            next_bbox_ids = torch.stack([next_cx_id, next_cy_id, next_w_id, next_h_id], dim=1)

            # --- Stopping Condition (EOS) ---
            is_eos = (next_cls_id == model.eos_token_id)
            if is_eos.all():
                break

            # Update sequences (generated_bboxes MUST BE LONG IDs)
            generated_cls_ids = torch.cat([generated_cls_ids, next_cls_id.unsqueeze(1)], dim=1)
            generated_bboxes = torch.cat([generated_bboxes, next_bbox_ids.unsqueeze(1)], dim=1) # <--- Use LongTensor IDs

        # 4. Final Extraction and Mapping
        final_cls_ids = generated_cls_ids[0, 1:] 
        final_bbox_ids = generated_bboxes[0, 1:] # [N, 4] LongTensor of IDs

        # Detokenize BBox IDs to float coordinates
        final_bboxes_float = final_bbox_ids.float() / range_divisor # [N, 4] FloatTensor

        layout = []
        for cls_id_tensor, bbox_float_tensor in zip(final_cls_ids, final_bboxes_float):
            internal_cls_id = cls_id_tensor.item()
            
            if internal_cls_id < 0 or internal_cls_id >= model.num_element_classes:
                continue 
            
            original_cls_id = internal_cls_id + 2
            
            cx, cy, w, h = bbox_float_tensor.tolist()
            layout.append((original_cls_id, cx, cy, w, h))

        return layout