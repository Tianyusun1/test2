# File: tianyusun1/test2/test2-2.0/inference/greedy_decode.py (FINAL QUERY-BASED FIXED)

import torch
import numpy as np

# --- Import KG ---
# 必须确保 models/kg.py 存在且可导入
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None
# -----------------

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda'):
    """
    Query-Based decoding:
    1. Use KG to determine *what* objects are in the poem (Queries).
    2. Use Model to determine *where* they are (Layout), guided by Text and Spatial Matrix.
    
    Args:
        model: Trained Poem2LayoutGenerator (Query-Based version).
        tokenizer: BertTokenizer.
        poem: Input string.
    Returns:
        layout: List of (cls_id, cx, cy, w, h).
    """
    if PoetryKnowledgeGraph is None:
        return []

    model.eval()
    device = torch.device(device)
    model.to(device)
    
    # 1. 实例化 KG 并提取内容
    pkg = PoetryKnowledgeGraph()
    
    # 提取视觉特征向量 (Multi-hot vector of shape [9])
    kg_vector = pkg.extract_visual_feature_vector(poem)
    
    # 将向量转换为具体的物体类别 ID 列表
    # vector index 0 -> internal ID 0 -> Class ID 2 (Mountain)
    # 我们需要 Input ID (词表索引 2-10)
    # vector indices: 0-8. Input IDs: 2-10.
    existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
    kg_class_ids = (existing_indices + 2).tolist()
    
    # Debug 打印
    # print(f"Poem: {poem}")
    # print(f"KG Detected Classes: {kg_class_ids}")
    
    if not kg_class_ids:
        # 如果 KG 没提取到任何物体，返回空布局
        return []
        
    # 2. 准备模型输入 Tensor
    # [1, S]
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    # 提取空间关系矩阵 (用于 Spatial Bias)
    kg_spatial_matrix_raw = pkg.extract_spatial_matrix(poem)
    kg_spatial_matrix = kg_spatial_matrix_raw.unsqueeze(0).to(device) # [1, 9, 9]
    
    # 文本编码
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Padding Mask (推理时 batch=1 且无 padding，全为 False)
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    # 3. 单次前向传播 (One-Shot Prediction)
    with torch.no_grad():
        # 模型返回: (None, None, pred_boxes, None)
        _, _, pred_boxes, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, # Queries
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix # Bias
        )
        # pred_boxes: [1, S, 4]
        
    # 4. 格式化输出
    layout = []
    # 取出第一个样本的结果，转为 list
    boxes_flat = pred_boxes[0].cpu().tolist()
    
    # 将类别 ID 和预测的坐标组合
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cx, cy, w, h = box
        # 确保输出格式为 (cls_id, cx, cy, w, h)
        # cls_id 已经是 2-10 的范围
        layout.append((float(cls_id), cx, cy, w, h))
        
    return layout