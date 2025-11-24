# scripts/train.py

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (train.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (train.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)
# --- 以上三行代码确保能正确导入 data, models, trainers 等模块 ---

# --- 现在可以安全地导入项目内部模块 ---
import torch
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset
from data.utils import collate_fn_for_layout
from models.poem2layout import Poem2LayoutGenerator
from trainers.trainer import LayoutTrainer

def main():
    # 1. Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Init tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        coord_loss_weight=config['model']['coord_loss_weight']
    )

    # 3. Load dataset
    dataset = PoegraphLayoutDataset(
        xlsx_path=config['model']['xlsx_path'],
        labels_dir=config['model']['labels_dir'],
        bert_model_path=config['model']['bert_path'],
        max_layout_length=config['model']['max_layout_length'],
        max_text_length=config['model']['max_text_length']
    )
    # Simple train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_for_layout
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_for_layout
    )

    # 4. Init trainer and start training
    trainer = LayoutTrainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()
