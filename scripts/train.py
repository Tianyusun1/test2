# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (train.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (train.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
import trainers # FIXED: 导入整个 trainers 包，依赖 trainers/__init__.py 
from collections import Counter
import numpy as np 

# --- 辅助函数：计算类别权重 (解决类别偏差问题) ---
def compute_class_weights(dataset, num_classes: int):
    """
    计算数据集内所有布局元素的类别频率，并返回反向频率权重。
    权重公式: w_i = 1.0 / log(1.02 + p_i)
    """
    class_counts = Counter()
    
    # 遍历整个数据集计算所有元素实例的类别计数
    for sample in dataset.data:
        # boxes 是 List[(cls_id, cx, cy, w, h)]，cls_id 是 2.0 - 10.0
        for cls_id_float, _, _, _, _ in sample['boxes']:
            # 映射到内部 ID 0-8
            internal_cls_id = int(cls_id_float) - 2
            if 0 <= internal_cls_id < num_classes:
                class_counts[internal_cls_id] += 1
                
    total_count = sum(class_counts.values())
    
    if total_count == 0:
        print("[Warning] No valid elements found in dataset for weight calculation. Using uniform weights.")
        return torch.ones(num_classes)

    weights = torch.zeros(num_classes)
    
    # 计算频率 p_i
    for i in range(num_classes):
        frequency = class_counts.get(i, 0) / total_count
        if frequency > 0:
            # 采用 log(1.0 + x) 或 log(1.02 + x) 来平滑和反转频率
            weights[i] = 1.0 / np.log(1.02 + frequency) 
        else:
            # 对于稀有/不存在的类别，赋予最高的权重
            weights[i] = 1.0 / np.log(1.02 + 1e-6) # 赋予最大权重
            
    # 标准化权重 (可选，但通常有助于稳定训练)
    weights = weights / weights.sum() * num_classes
    
    return weights.float()
# ------------------------------------------


def main():
    # 1. Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Init tokenizer and load FULL dataset for weight calculation
    model_config = config['model']
    
    # 重新初始化数据集以确保数据完整性
    dataset = PoegraphLayoutDataset(
        xlsx_path=model_config['xlsx_path'],
        labels_dir=model_config['labels_dir'],
        bert_model_path=model_config['bert_path'],
        max_layout_length=model_config['max_layout_length'],
        max_text_length=model_config['max_text_length']
    )
    
    # --- NEW: 计算类别权重 ---
    num_element_classes = model_config['num_classes'] # 9
    class_weights_tensor = compute_class_weights(dataset, num_element_classes)
    print(f"Calculated Class Weights (Internal 0-8): {class_weights_tensor.tolist()}")
    # ---------------------------

    # 3. Init model (传入 IoU 权重和类别权重)
    model = Poem2LayoutGenerator(
        bert_path=model_config['bert_path'],
        num_classes=num_element_classes, # 实际元素类别数 (9)
        hidden_size=model_config['hidden_size'],
        bb_size=model_config['bb_size'],
        decoder_layers=model_config['decoder_layers'],
        decoder_heads=model_config['decoder_heads'],
        dropout=model_config['dropout'],
        coord_loss_weight=model_config['coord_loss_weight'],
        # --- 传入解决堆叠和类别偏差问题的关键参数 ---
        iou_loss_weight=model_config.get('iou_loss_weight', 0.1), 
        class_weights=class_weights_tensor 
        # -----------------------------------------
    )

    # 4. Split dataset and init data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=layout_collate_fn 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=layout_collate_fn 
    )

    # 5. Init trainer and start training
    trainer = trainers.LayoutTrainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()