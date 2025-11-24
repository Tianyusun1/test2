import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

def layout_seq_to_yolo_txt(layout_seq: List[Tuple], output_path: str):
    """
    将布局序列 [(cls_id, cx, cy, w, h), ...] 保存为 YOLO 格式的 .txt 文件。
    Args:
        layout_seq: 布局序列，每个元素为 (cls_id, cx, cy, w, h)，cls_id 必须是 2-10 范围的整数。
        output_path: 输出 .txt 文件路径
    """
    # 确保写入文件中的 cls_id 是整数
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in layout_seq:
            if len(item) != 5:
                continue
            cls_id, cx, cy, w, h = item
            
            # Note: 这里的 cls_id 必须是 2-10 范围的浮点数/整数，直接写入
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def yolo_txt_to_layout_seq(txt_path: str) -> List[Tuple]:
    """
    从 YOLO 格式的 .txt 文件加载布局序列 [(cls_id, cx, cy, w, h), ...]。
    
    WARNING: 此函数返回的 cls_id 范围是 2-10 (即原始 ID)，这与模型训练时使用的 
             BOS/EOS (0/1) ID 兼容。如果需要 0-8 范围的内部 ID，请在调用后手动映射。

    Args:
        txt_path: 输入 .txt 文件路径
    Returns:
        layout_seq: 布局序列 [(cls_id, cx, cy, w, h)]
    """
    layout_seq = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # 读取时，确保 cls_id 是整数，坐标是浮点数
                cls_id = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # 修复: 移除 map_to_internal 逻辑，确保始终返回原始 ID (2-10)
                # 这样可以避免与 BOS/EOS (0/1) 冲突
                layout_seq.append((cls_id, cx, cy, w, h))
    except FileNotFoundError:
        print(f"Error: YOLO file not found at {txt_path}")
    except Exception as e:
        print(f"Error reading YOLO file {txt_path}: {e}")
        
    return layout_seq

def get_device():
    """Returns the torch device (cuda if available, else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WARNING: 移除冗余的 collate_fn_for_layout，请使用 dataset.py 中的 layout_collate_fn