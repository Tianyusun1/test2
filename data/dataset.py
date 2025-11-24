import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional

# 类别定义（保持一致）
# 模型预期: 0: BOS, 1: EOS, 2-10: 布局元素
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

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
                            
                            # === 核心修复：移除错误的类别 ID 映射 ===
                            # 保持原始 ID (2~10)，使其与模型中保留的 BOS/EOS (0/1) 空间分开。
                            # 确保它们都是浮点数，因为 layout_seq 最终是 float32
                            cls_id_float = float(cls_id)
                            boxes.append((cls_id_float, cx, cy, w, h))
                            # ====================================
                            
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    'boxes': boxes[:max_layout_length] # 截断过长布局
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")

        # 初始化 BERT tokenizer（模型加载由 Trainer 负责）
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        # boxes: List[(cls_id, cx, cy, w, h)] where cls_id is 2.0~10.0
        boxes = sample['boxes'] 

        # 1. 文本编码（使用 BERT tokenizer）
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # 2. 布局序列：展平为一维 list
        layout_seq = []
        for cls_id_float, cx, cy, w, h in boxes:
            layout_seq.extend([cls_id_float, cx, cy, w, h]) # 展平为一维 list

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            # layout_seq 现在包含 ID 2.0~10.0
            'layout_seq': layout_seq, 
            'num_boxes': len(boxes)
        }

# ========================
# Collate Function for DataLoader - 保持不变
# ========================
def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将变长 layout_seq padding 到 batch 内最大长度
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    num_boxes_list = [item['num_boxes'] for item in batch]

    # 找到最大布局长度（以 5 为单位）
    max_seq_len = max(len(item['layout_seq']) for item in batch)
    if max_seq_len == 0:
        max_seq_len = 5 # 至少一个占位

    # padding layout_seq
    layout_seqs_padded = []
    layout_masks = []
    # 使用 0.0 进行 padding，这对坐标是合理的，但请注意，cls_id 0 已经被 BOS 占用。
    # 由于 mask 会将 padded 部分忽略，所以这里使用 0.0 padding 是可接受的。
    for item in batch:
        seq = item['layout_seq']
        pad_len = max_seq_len - len(seq)
        padded_seq = seq + [0.0] * pad_len
        layout_seqs_padded.append(padded_seq)

        # mask: 1 for real, 0 for pad
        mask = [1.0] * len(seq) + [0.0] * pad_len
        layout_masks.append(mask)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'layout_seq': torch.tensor(layout_seqs_padded, dtype=torch.float32), 
        'layout_mask': torch.tensor(layout_masks, dtype=torch.float32), 
        'num_boxes': torch.tensor(num_boxes_list, dtype=torch.long)
    }

# ========================
# 简单验证脚本 - 保持不变
# ========================
if __name__ == "__main__":
    # ... (验证脚本)
    pass