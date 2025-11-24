# scripts/infer.py

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (infer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (infer.py 的父目录的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)
# --- 以上三行代码确保能正确导入 models, inference, data 等模块 ---

# --- 现在可以安全地导入项目内部模块 ---
import torch
from transformers import BertTokenizer
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from data.utils import layout_seq_to_yolo_txt
import yaml

def main():
    # 1. Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Load model and tokenizer
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

    # Load checkpoint
    # You need to specify the path to your trained model checkpoint
    checkpoint_path = "/home/610-sty/Layout2Paint/outputs/model_epoch_99_val_loss_0.5050.pth" # TODO: Set correct path
    # checkpoint_path = "./outputs/model_epoch_9_val_loss_1.2345.pth" # Example
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set to evaluation mode

    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 4. Run inference
    poem = "山色空蒙雨亦奇，小桥流水绕柴扉。" # Example poem, replace with your input
    print(f"Input poem: {poem}")

    layout = greedy_decode_poem_layout(
        model, tokenizer, poem,
        max_elements=config['model']['max_elements'],
        device=device
    )

    print("Generated Layout:")
    for i, (cls_id, cx, cy, w, h) in enumerate(layout):
        print(f"  {i+1}: Class {cls_id}, (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})")

    # 5. Save layout to YOLO format
    output_txt_path = "./outputs/predicted_layout.txt"
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True) # Ensure output dir exists
    layout_seq_to_yolo_txt(layout, output_txt_path)
    print(f"Layout saved to {output_txt_path}")

if __name__ == "__main__":
    main()
