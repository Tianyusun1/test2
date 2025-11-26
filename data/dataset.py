import os
import torch
import pandas as pd
import yaml # NEW: 导入 yaml 用于读取配置
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np # <<< NEW: 导入 numpy 用于几何增强

# --- NEW: 从同级目录的 utils.py 导入 BBoxTokenizer ---
from .utils import BBoxTokenizer 

# --- NEW: 导入知识图谱模型 ---
# 确保您的 models/kg.py 已经更新包含 extract_visual_feature_vector 方法
from models.kg import PoetryKnowledgeGraph
# ---------------------------

# 类别定义（保持一致）
# 模型预期: 0: BOS, 1: EOS, 2-10: 布局元素
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

# --- 辅助函数：读取配置中的 num_bbox_bins ---
def _load_num_bins_from_config():
    """尝试从 configs/default.yaml 中读取 num_bbox_bins，否则使用默认值 1000。"""
    try:
        # 假设 config 文件在项目根目录下的 configs/default.yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config['model']['num_bbox_bins']
    except Exception:
        # 如果读取失败，使用默认值
        return 1000
# ---------------------------------------------


class PoegraphLayoutDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        labels_dir: str,
        bert_model_path: str = "/home/610-sty/huggingface/bert-base-chinese",
        max_layout_length: int = 30, 
        max_text_length: int = 64, 
        preload: bool = False
    ):
        super().__init__()
        self.xlsx_path = xlsx_path
        self.labels_dir = Path(labels_dir)
        self.max_layout_length = max_layout_length
        self.max_text_length = max_text_length
        
        # 实际元素类别数 (9)，但总词汇量为 11 (含 BOS/EOS)
        self.num_classes = 9 

        # --- NEW: 初始化知识图谱 ---
        print("Initializing Knowledge Graph...")
        self.pkg = PoetryKnowledgeGraph()
        print("✅ Knowledge Graph initialized.")
        # --------------------------

        # --- NEW: 初始化 BBox Tokenizer ---
        num_bins = _load_num_bins_from_config()
        self.bbox_tokenizer = BBoxTokenizer(num_bins=num_bins)
        print(f"Dataset initialized with BBox bins: {num_bins}")
        # ----------------------------------

        # 加载 Excel（自动读取表头 'image', 'poem'）
        df = pd.read_excel(xlsx_path)
        self.data = []

        for _, row in df.iterrows():
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            img_stem = Path(raw_img_name).stem
            label_path = self.labels_dir / f"{img_stem}.txt"

            if not label_path.exists():
                continue

            # 读取并验证标注
            boxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # 检查坐标归一化和类别有效性
                        if cls_id in VALID_CLASS_IDS and \
                           0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            
                            # 保持原始 ID (2~10)
                            cls_id_float = float(cls_id)
                            # 存储原始的浮点数，在 __getitem__ 中进行 tokenize
                            boxes.append((cls_id_float, cx, cy, w, h)) 
                            
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    # boxes 仍存储浮点数
                    'boxes': boxes[:max_layout_length] 
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")

        # 初始化 BERT tokenizer（模型加载由 Trainer 负责）
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

# [data/dataset.py]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        boxes = sample['boxes'] 

        # 1. 文本编码
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # 2. KG 特征提取
        kg_vector = self.pkg.extract_visual_feature_vector(poem)
        kg_spatial_matrix = self.pkg.extract_spatial_matrix(poem)

        # 3. 布局处理
        layout_seq_ids = []
        target_boxes_float = []  # <--- NEW: 存储真实的浮点坐标用于回归损失
        
        # 几何增强
        apply_aug = np.random.rand() < 0.5 
        
        for cls_id_float, cx, cy, w, h in boxes:
            if apply_aug:
                noise_magnitude = 0.01 
                noise = np.random.uniform(-noise_magnitude, noise_magnitude, size=4)
                cx = np.clip(cx + noise[0], 0.0, 1.0).item()
                cy = np.clip(cy + noise[1], 0.0, 1.0).item()
                w = np.clip(w + noise[2], 0.01, 1.0).item()
                h = np.clip(h + noise[3], 0.01, 1.0).item()
            
            # 存储无损浮点值 [cx, cy, w, h]
            target_boxes_float.append([cx, cy, w, h]) 

            # BBox 离散化
            cx_id = self.bbox_tokenizer.tokenize(cx)
            cy_id = self.bbox_tokenizer.tokenize(cy)
            w_id = self.bbox_tokenizer.tokenize(w)
            h_id = self.bbox_tokenizer.tokenize(h)
            
            layout_seq_ids.extend([cls_id_float, cx_id, cy_id, w_id, h_id])

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'layout_seq': layout_seq_ids, 
            'num_boxes': len(boxes),
            'kg_vector': kg_vector,
            'kg_spatial_matrix': kg_spatial_matrix,
            'target_boxes': target_boxes_float # <--- NEW: 返回浮点列表
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles padding for ids and stacking for float boxes.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    num_boxes_list = [item['num_boxes'] for item in batch]
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    kg_spatial_matrices = torch.stack([item['kg_spatial_matrix'] for item in batch])

    # 1. Pad Layout IDs (Discrete)
    max_seq_len = max(len(item['layout_seq']) for item in batch)
    if max_seq_len == 0: max_seq_len = 5
    
    layout_seqs_padded = []
    layout_masks = []
    
    for item in batch:
        seq = item['layout_seq']
        pad_len = max_seq_len - len(seq)
        layout_seqs_padded.append(seq + [0.0] * pad_len)
        layout_masks.append([1.0] * len(seq) + [0.0] * pad_len)
        
    # 2. Pad Target Boxes (Continuous Float)
    # layout_seq 长度是 5 * num_elements, 所以 boxes 数量是 max_seq_len // 5
    max_num_boxes = max_seq_len // 5
    target_boxes_padded = []
    
    for item in batch:
        boxes = item['target_boxes'] # list of [cx, cy, w, h]
        pad_count = max_num_boxes - len(boxes)
        # Pad with zeros
        padded_boxes = boxes + [[0.0, 0.0, 0.0, 0.0]] * pad_count
        target_boxes_padded.append(padded_boxes)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'layout_seq': torch.tensor(layout_seqs_padded, dtype=torch.long), 
        'layout_mask': torch.tensor(layout_masks, dtype=torch.float32), 
        'num_boxes': torch.tensor(num_boxes_list, dtype=torch.long),
        'kg_vector': kg_vectors,
        'kg_spatial_matrix': kg_spatial_matrices,
        'target_boxes': torch.tensor(target_boxes_padded, dtype=torch.float32) # <--- NEW: [B, N, 4]
    }
# ========================
# 简单验证脚本 - 保持不变
# ========================
if __name__ == "__main__":
    # ... (验证脚本)
    pass