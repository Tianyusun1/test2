# File: tianyusun1/test2/test2-4.0/data/dataset.py (V4.6: ENHANCED AUGMENTATION)

import os
import torch
import pandas as pd
import yaml 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np 
import random # [NEW]

# --- 导入知识图谱模型 ---
from models.kg import PoetryKnowledgeGraph
# --- 导入位置引导信号生成器 ---
from models.location import LocationSignalGenerator
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
        
        # [MODIFIED] 初始化位置信号生成器 (8x8)
        self.location_gen = LocationSignalGenerator(grid_size=8)
        print("✅ Location Signal Generator (8x8) initialized.")
        
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
        
        # 将 vector (0/1/0.5) 转为具体的类别 ID 列表 (2-10)
        existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
        
        # [A] ID 列表
        kg_class_ids = (existing_indices + 2).tolist() 
        
        # [B] 权重列表 (1.0 or 0.5) [NEW]
        kg_class_weights = kg_vector[existing_indices].tolist()

        if not kg_class_ids:
            kg_class_ids = [0]
            kg_class_weights = [0.0]

        # === [MODIFIED] 生成位置引导信号 (Location Grids - Stateful) ===
        # 初始化画布状态 (8x8)
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32) # [MODIFIED] 8x8
        location_grids_list = [] # 临时存储
        
        # 遍历生成的 Queries (kg_class_ids)
        for i, cls_id in enumerate(kg_class_ids):
            cls_id = int(cls_id)
            if cls_id == 0: # PAD
                # [MODIFIED] Padding 也要用 8x8
                location_grids_list.append(torch.zeros((8, 8), dtype=torch.float32))
                continue
                
            # Class ID (2-10) -> Matrix Index (0-8)
            matrix_idx = cls_id - 2
            
            # 获取该物体在 KG 中的空间关系
            # Row: 我对别人的关系; Col: 别人对我的关系
            spatial_row = kg_spatial_matrix[matrix_idx]  
            spatial_col = kg_spatial_matrix[:, matrix_idx] 
            
            # 调用 location_gen 进行有状态推理
            # 训练数据生成时通常使用默认的 'greedy' 模式以保持稳定，
            # 但也可以根据需要改为 'sample' 来做数据增强。这里保持默认。
            signal, current_occupancy = self.location_gen.infer_stateful_signal(
                i, spatial_row, spatial_col, current_occupancy
            )
            
            location_grids_list.append(signal)
            
        # =======================================================

        # 3. GT 对齐与清洗 (Alignment & Cleaning Logic)
        target_boxes = []
        loss_mask = [] # 1.0: 有效GT, 0.0: 无GT或脏数据

        # 将 GT 按类别分组
        gt_dict = {}
        for item in gt_boxes:
            cid, cx, cy, w, h = item
            cid = int(cid)
            if cid not in gt_dict: gt_dict[cid] = []
            gt_dict[cid].append([cx, cy, w, h])

        # [NEW] 全局数据增强决策 (Flip Augmentation)
        # 50% 概率水平翻转
        do_flip = random.random() < 0.5
        
        # 遍历 KG 要求的每个物体
        for k_cls in kg_class_ids:
            k_cls = int(k_cls)
            if k_cls == 0: # PAD
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                loss_mask.append(0.0)
                continue

            if k_cls in gt_dict and len(gt_dict[k_cls]) > 0:
                # 找到了 -> 取出
                box = gt_dict[k_cls].pop(0) # [cx, cy, w, h]
                
                # === [NEW] 脏数据过滤逻辑 ===
                # 1. 过滤掉面积过大的框 (例如超过 90% 画幅)
                if box[2] * box[3] > 0.90:
                    target_boxes.append([0.0, 0.0, 0.0, 0.0])
                    loss_mask.append(0.0) 
                    continue
                
                # 2. 过滤掉长宽比极端的框
                aspect_ratio = box[2] / (box[3] + 1e-6)
                if aspect_ratio > 10.0 or aspect_ratio < 0.1:
                    target_boxes.append([0.0, 0.0, 0.0, 0.0])
                    loss_mask.append(0.0)
                    continue
                # ===========================
                
                # [几何增强]
                # 1. Flip (如果触发)
                # cx -> 1.0 - cx
                if do_flip:
                    box[0] = 1.0 - box[0]
                
                # 2. Jitter (微小抖动)
                # 稍微加大一点抖动范围 (-0.02 ~ 0.02)
                noise = np.random.uniform(-0.02, 0.02, size=4)
                box_aug = [
                    np.clip(box[0] + noise[0], 0.0, 1.0),
                    np.clip(box[1] + noise[1], 0.0, 1.0),
                    np.clip(box[2] + noise[2], 0.01, 1.0),
                    np.clip(box[3] + noise[3], 0.01, 1.0)
                ]
                target_boxes.append(box_aug)
                loss_mask.append(1.0)
            else:
                # KG 说有，但 GT 没标
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                loss_mask.append(0.0)

        # 限制最大长度
        if len(kg_class_ids) > self.max_layout_length:
            kg_class_ids = kg_class_ids[:self.max_layout_length]
            kg_class_weights = kg_class_weights[:self.max_layout_length] 
            target_boxes = target_boxes[:self.max_layout_length]
            loss_mask = loss_mask[:self.max_layout_length]
            # [NEW] 同时裁剪 location_grids
            location_grids_list = location_grids_list[:self.max_layout_length]

        # 转为 Tensor
        location_grids = torch.stack(location_grids_list) # [T, 8, 8]
        
        # [注意] location_grids 也要 Flip!
        # Grid 是 8x8 的 Heatmap，水平翻转意味着在 dim=2 (Width) 上翻转
        if do_flip:
            # torch.flip(input, dims)
            location_grids = torch.flip(location_grids, dims=[2])

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'kg_class_ids': torch.tensor(kg_class_ids, dtype=torch.long),
            'kg_class_weights': torch.tensor(kg_class_weights, dtype=torch.float32), 
            'target_boxes': torch.tensor(target_boxes, dtype=torch.float32),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
            'kg_spatial_matrix': kg_spatial_matrix,
            'kg_vector': kg_vector,
            'num_boxes': torch.tensor(len(gt_boxes), dtype=torch.long),
            'location_grids': location_grids # [NEW] Added to return
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function to handle variable length kg_class_ids."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kg_spatial_matrices = torch.stack([item['kg_spatial_matrix'] for item in batch])
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])

    lengths = [len(item['kg_class_ids']) for item in batch]
    max_len = max(lengths)
    if max_len == 0: max_len = 1

    batched_class_ids = []
    batched_class_weights = [] 
    batched_target_boxes = []
    batched_loss_mask = []
    batched_padding_mask = [] 
    batched_location_grids = [] # [NEW]

    for item in batch:
        cur_len = len(item['kg_class_ids'])
        pad_len = max_len - cur_len
        
        # 1. IDs
        padded_ids = torch.cat([
            item['kg_class_ids'], 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batched_class_ids.append(padded_ids)
        
        # 2. Weights 
        padded_weights = torch.cat([
            item['kg_class_weights'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_class_weights.append(padded_weights)
        
        # 3. Boxes
        padded_boxes = torch.cat([
            item['target_boxes'], 
            torch.zeros((pad_len, 4), dtype=torch.float32)
        ])
        batched_target_boxes.append(padded_boxes)
        
        # 4. Mask
        padded_loss_mask = torch.cat([
            item['loss_mask'], 
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_loss_mask.append(padded_loss_mask)

        # 5. [NEW] Location Grids
        # item['location_grids'] shape: [cur_len, 8, 8]
        # padding shape: [pad_len, 8, 8]
        padded_grids = torch.cat([
            item['location_grids'],
            # [MODIFIED] Padding 维度改为 8x8
            torch.zeros((pad_len, 8, 8), dtype=torch.float32)
        ])
        batched_location_grids.append(padded_grids)
        
        # 6. Pad Mask
        pad_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[cur_len:] = True
        batched_padding_mask.append(pad_mask)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'kg_class_ids': torch.stack(batched_class_ids),      
        'kg_class_weights': torch.stack(batched_class_weights), 
        'target_boxes': torch.stack(batched_target_boxes),   
        'loss_mask': torch.stack(batched_loss_mask),         
        'padding_mask': torch.stack(batched_padding_mask),   
        'kg_spatial_matrix': kg_spatial_matrices,
        'kg_vector': kg_vectors,
        'num_boxes': num_boxes,
        'location_grids': torch.stack(batched_location_grids) # [NEW] [B, T, 8, 8]
    }

if __name__ == "__main__":
    pass