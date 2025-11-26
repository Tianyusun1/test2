# File: tianyusun1/test2/test2-2.0/data/dataset.py (FINAL QUERY-BASED FIXED)

import os
import torch
import pandas as pd
import yaml 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np 

# --- 导入知识图谱模型 ---
from models.kg import PoetryKnowledgeGraph
# ---------------------------

# 类别定义
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

def _load_num_bins_from_config():
    """尝试从 configs/default.yaml 中读取 num_bbox_bins"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config['model']['num_bbox_bins']
    except Exception:
        return 1000

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
        self.max_layout_length = max_layout_length # 这里指最大物体数量
        self.max_text_length = max_text_length
        
        self.num_classes = 9 

        print("Initializing Knowledge Graph...")
        self.pkg = PoetryKnowledgeGraph()
        print("✅ Knowledge Graph initialized.")
        
        # 加载 Excel
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
                        if len(parts) != 5: continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        if cls_id in VALID_CLASS_IDS and \
                           0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            boxes.append((float(cls_id), cx, cy, w, h)) 
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    'boxes': boxes # List[(cls, cx, cy, w, h)]
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        gt_boxes = sample['boxes'] # List[(cls_id, cx, cy, w, h)]

        # 1. 文本编码
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # 2. KG 提取内容 (Query Generation)
        # extract_visual_feature_vector 返回 [9] 的 multi-hot
        kg_vector = self.pkg.extract_visual_feature_vector(poem)
        kg_spatial_matrix = self.pkg.extract_spatial_matrix(poem)
        
        # 将 vector (0/1) 转为具体的类别 ID 列表 (2-10)
        # indices: 0-8 -> class ids: 2-10
        existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
        kg_class_ids = (existing_indices + 2).tolist() 
        
        # 如果 KG 没提取到任何东西，为了防止报错，加一个 PAD(0) 或特殊处理
        # 这里我们假设至少有一个，如果没有，给一个占位符 0 (PAD)
        if not kg_class_ids:
            kg_class_ids = [0]

        # 3. GT 对齐 (Alignment Logic)
        # 我们要构建 target_boxes，其长度必须与 kg_class_ids 一致
        target_boxes = []
        loss_mask = [] # 1.0: 有GT, 0.0: 无GT(漏标)

        # 将 GT 按类别分组: {class_id: [[cx,cy,w,h], ...]}
        gt_dict = {}
        for item in gt_boxes:
            cid, cx, cy, w, h = item
            cid = int(cid)
            if cid not in gt_dict: gt_dict[cid] = []
            gt_dict[cid].append([cx, cy, w, h])

        # 遍历 KG 要求的每个物体，去 GT 里找
        for k_cls in kg_class_ids:
            k_cls = int(k_cls)
            if k_cls == 0: # PAD
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                loss_mask.append(0.0)
                continue

            if k_cls in gt_dict and len(gt_dict[k_cls]) > 0:
                # 找到了 -> 取出一个作为 Target
                # 策略: pop(0) 取第一个匹配的，避免重复匹配同一个 GT
                box = gt_dict[k_cls].pop(0)
                
                # [几何增强] 仅在训练时对 GT 坐标做微小扰动
                # 这里为了简化，直接在 getitem 里做
                # 也可以在 trainer 里做，这里加上简单的噪声
                noise = np.random.uniform(-0.01, 0.01, size=4)
                box_aug = [
                    np.clip(box[0] + noise[0], 0.0, 1.0),
                    np.clip(box[1] + noise[1], 0.0, 1.0),
                    np.clip(box[2] + noise[2], 0.01, 1.0),
                    np.clip(box[3] + noise[3], 0.01, 1.0)
                ]
                target_boxes.append(box_aug)
                loss_mask.append(1.0)
            else:
                # KG 说有，但 GT 没标 -> Target 填 0，Mask 设为 0 (不计算 Loss)
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                loss_mask.append(0.0)

        # 限制最大长度
        if len(kg_class_ids) > self.max_layout_length:
            kg_class_ids = kg_class_ids[:self.max_layout_length]
            target_boxes = target_boxes[:self.max_layout_length]
            loss_mask = loss_mask[:self.max_layout_length]

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'kg_class_ids': torch.tensor(kg_class_ids, dtype=torch.long),
            'target_boxes': torch.tensor(target_boxes, dtype=torch.float32),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
            'kg_spatial_matrix': kg_spatial_matrix,
            'kg_vector': kg_vector,
            'num_boxes': torch.tensor(len(gt_boxes), dtype=torch.long) # 仅用于记录
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle variable length kg_class_ids.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kg_spatial_matrices = torch.stack([item['kg_spatial_matrix'] for item in batch])
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])

    # Find max length in this batch
    lengths = [len(item['kg_class_ids']) for item in batch]
    max_len = max(lengths)
    if max_len == 0: max_len = 1 # 防止空 batch

    batched_class_ids = []
    batched_target_boxes = []
    batched_loss_mask = []
    batched_padding_mask = [] # For Transformer (True = Padding)

    for item in batch:
        cur_len = len(item['kg_class_ids'])
        pad_len = max_len - cur_len
        
        # 1. Pad Class IDs (PAD=0)
        # [S] -> [Max_S]
        padded_ids = torch.cat([
            item['kg_class_ids'], 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batched_class_ids.append(padded_ids)
        
        # 2. Pad Target Boxes
        # [S, 4] -> [Max_S, 4]
        padded_boxes = torch.cat([
            item['target_boxes'], 
            torch.zeros((pad_len, 4), dtype=torch.float32)
        ])
        batched_target_boxes.append(padded_boxes)
        
        # 3. Pad Loss Mask
        # [S] -> [Max_S]
        padded_loss_mask = torch.cat([
            item['loss_mask'], 
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_loss_mask.append(padded_loss_mask)
        
        # 4. Create Padding Mask (True where padded)
        # [Max_S]
        pad_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[cur_len:] = True
        batched_padding_mask.append(pad_mask)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'kg_class_ids': torch.stack(batched_class_ids),      # [B, S]
        'target_boxes': torch.stack(batched_target_boxes),   # [B, S, 4]
        'loss_mask': torch.stack(batched_loss_mask),         # [B, S]
        'padding_mask': torch.stack(batched_padding_mask),   # [B, S]
        'kg_spatial_matrix': kg_spatial_matrices,
        'kg_vector': kg_vectors,
        'num_boxes': num_boxes
    }

if __name__ == "__main__":
    pass