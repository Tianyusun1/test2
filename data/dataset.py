import os
import torch
import pandas as pd
import yaml # NEW: 导入 yaml 用于读取配置
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional

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

        # --- NEW: 提取 KG 视觉特征向量 ---
        # 这个向量长度为 9，对应 9 个视觉类别，表示它们在诗中是否被提及或暗示
        kg_vector = self.pkg.extract_visual_feature_vector(poem)
        # -------------------------------

        # 2. 布局序列：展平为一维 list (核心修改)
        # layout_seq 现在是一个包含整数 ID 的列表：
        # [cls_id, cx_id, cy_id, w_id, h_id, cls_id, cx_id, ...]
        layout_seq_ids = []
        for cls_id_float, cx, cy, w, h in boxes:
            
            # --- NEW: BBox 离散化 ---
            # 类别 ID 仍使用 float (2.0-10.0)，在 collate_fn 中转为 LongTensor
            
            # 将 4 个连续坐标转换为整数 ID (0 - num_bins-1)
            cx_id = self.bbox_tokenizer.tokenize(cx)
            cy_id = self.bbox_tokenizer.tokenize(cy)
            w_id = self.bbox_tokenizer.tokenize(w)
            h_id = self.bbox_tokenizer.tokenize(h)
            
            layout_seq_ids.extend([cls_id_float, cx_id, cy_id, w_id, h_id])
            # ---------------------------

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            # 传递包含 ID 的 float 列表 (在 collate_fn 中转为 LongTensor)
            'layout_seq': layout_seq_ids, 
            'num_boxes': len(boxes),
            'kg_vector': kg_vector # <--- NEW: 返回 KG 向量
        }

# ========================
# Collate Function for DataLoader - 必须修改以处理整数序列和 KG 向量
# ========================
def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将变长 layout_seq padding 到 batch 内最大长度
    并堆叠 kg_vector。
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    num_boxes_list = [item['num_boxes'] for item in batch]
    
    # --- NEW: 堆叠 KG 向量 ---
    # 结果形状: [Batch_Size, 9]
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    # ------------------------

    # 找到最大布局长度（以 5 为单位）
    max_seq_len = max(len(item['layout_seq']) for item in batch)
    if max_seq_len == 0:
        max_seq_len = 5 # 至少一个占位

    # padding layout_seq
    layout_seqs_padded = []
    layout_masks = []
    
    # NEW: padding 必须使用 0，因为 0 代表 PAD/无效/背景。
    for item in batch:
        seq = item['layout_seq']
        pad_len = max_seq_len - len(seq)
        padded_seq = seq + [0.0] * pad_len # 仍然使用 0.0 padding
        layout_seqs_padded.append(padded_seq)

        # mask: 1 for real, 0 for pad
        mask = [1.0] * len(seq) + [0.0] * pad_len
        layout_masks.append(mask)
    
    # 核心更改：将 layout_seqs 转换为 LongTensor，因为它们现在代表离散 ID
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        # 将 float32 转换为 LongTensor，其中 0 代表 PAD
        'layout_seq': torch.tensor(layout_seqs_padded, dtype=torch.long), 
        'layout_mask': torch.tensor(layout_masks, dtype=torch.float32), 
        'num_boxes': torch.tensor(num_boxes_list, dtype=torch.long),
        'kg_vector': kg_vectors # <--- NEW: 返回堆叠后的 KG 向量
    }

# ========================
# 简单验证脚本 - 保持不变
# ========================
if __name__ == "__main__":
    # ... (验证脚本)
    pass