# File: tianyusun1/test2/test2-4.0/inference/greedy_decode.py (V4.2: CVAE COMPATIBLE)

import torch
import numpy as np

# --- Import KG & Location ---
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None

# [NEW] 导入位置生成器
try:
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] Could not import LocationSignalGenerator. Make sure models/location.py is accessible.")
    LocationSignalGenerator = None
# -----------------

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda', mode='greedy', top_k=3):
    """
    Query-Based decoding with Location Guidance & CVAE Diversity.
    
    1. Use KG to determine *what* objects are in the poem (Queries).
    2. Use LocationGenerator to determine *roughly where* they should be (Grid Signal).
    3. Use Model (CVAE Decoder) to determine *exactly where* they are.
       - Since target_boxes is NOT provided, the model samples latent 'z' from Prior N(0, I).
       - This introduces diversity: calling this function multiple times yields different layouts.
    
    Args:
        model: Trained Poem2LayoutGenerator (V4.2+).
        tokenizer: BertTokenizer.
        poem: Input string.
        mode: 'greedy' or 'sample' (Controls LocationGenerator behavior).
        top_k: Top-K sampling parameter (for LocationGenerator).
    Returns:
        layout: List of (cls_id, cx, cy, w, h).
    """
    if PoetryKnowledgeGraph is None:
        return []

    model.eval()
    device = torch.device(device)
    model.to(device)
    
    # 1. 实例化组件
    pkg = PoetryKnowledgeGraph()
    
    # [NEW] 实例化位置生成器 (使用 8x8 Grid)
    if LocationSignalGenerator is not None:
        location_gen = LocationSignalGenerator(grid_size=8)
    else:
        location_gen = None
    
    # 2. KG 提取内容
    # 提取视觉特征向量
    kg_vector = pkg.extract_visual_feature_vector(poem)
    
    # 转为物体 ID 列表 (2-10)
    existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
    kg_class_ids = (existing_indices + 2).tolist()
    
    if not kg_class_ids:
        return []
        
    # 3. 准备模型输入 Tensor
    # Content Queries: [1, S]
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    # Spatial Matrix (Bias): [1, 9, 9]
    kg_spatial_matrix_raw = pkg.extract_spatial_matrix(poem)
    kg_spatial_matrix = kg_spatial_matrix_raw.unsqueeze(0).to(device) 
    
    # === [NEW] 生成位置引导信号 (Location Grids) ===
    location_grids_tensor = None
    
    if location_gen is not None:
        # 初始化画布状态 (8x8)
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grids_list = []
        
        for i, cls_id in enumerate(kg_class_ids):
            matrix_idx = cls_id - 2
            
            # 获取关系 (注意: kg_spatial_matrix_raw 是 CPU Tensor)
            row = kg_spatial_matrix_raw[matrix_idx]
            col = kg_spatial_matrix_raw[:, matrix_idx]
            
            # [MODIFIED] 使用传入的 mode 和 top_k 参数控制 Grid 的随机性
            signal, current_occupancy = location_gen.infer_stateful_signal(
                i, row, col, current_occupancy, 
                mode=mode, top_k=top_k 
            )
            grids_list.append(signal)
            
        # Stack & Batch Dimension: [1, S, 8, 8]
        location_grids_tensor = torch.stack(grids_list).unsqueeze(0).to(device)
    # ===============================================
    
    # 文本编码
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Padding Mask
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    # 4. 单次前向传播 (One-Shot Prediction)
    with torch.no_grad():
        # [CVAE Update] 
        # Model returns: mu, logvar, pred_boxes, decoder_output
        # We implicitly pass target_boxes=None, triggering z ~ N(0, 1) sampling inside model.
        _, _, pred_boxes, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, 
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix,
            location_grids=location_grids_tensor
        )
        
    # 5. 格式化输出
    layout = []
    boxes_flat = pred_boxes[0].cpu().tolist()
    
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cx, cy, w, h = box
        layout.append((float(cls_id), cx, cy, w, h))
        
    return layout