# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (infer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)

# 修复路径：infer.py 位于 scripts/ 下，所以 project_root 是它的父目录的父目录
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
    # 修复：确保配置文件路径相对于脚本的执行位置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Load model and tokenizer
    # Note: num_classes 应该已经被修正为 11 (9个元素 + BOS/EOS)
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

    # 4. Load checkpoint
    # 移除硬编码：使用配置中的 output_dir，并要求用户设置 CHECKPOINT_NAME
    output_dir = config['training']['output_dir']
    CHECKPOINT_NAME = "model_epoch_99_val_loss_0.5050.pth" # <<< TODO: 请用户在此设置正确的权重文件名
    
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}\n"
                                f"Please update CHECKPOINT_NAME in infer.py.")

    print(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set to evaluation mode

    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 5. Run inference
    # 优化：可以从命令行参数读取 poem，这里使用硬编码示例
    poem = "山色空蒙雨亦奇，小桥流水绕柴扉。"
    print(f"Input poem: {poem}")

    # 获取 max_elements
    max_elements = config['model'].get('max_elements', 30)
    
    layout = greedy_decode_poem_layout(
        model, tokenizer, poem,
        max_elements=max_elements,
        device=device
    )

    print("Generated Layout:")
    for i, (cls_id, cx, cy, w, h) in enumerate(layout):
        print(f"  {i+1}: Class {cls_id}, (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})")

    # 6. Save layout to YOLO format
    output_txt_path = os.path.join(output_dir, "predicted_layout.txt")
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True) # Ensure output dir exists
    layout_seq_to_yolo_txt(layout, output_txt_path)
    print(f"Layout saved to {output_txt_path}")

if __name__ == "__main__":
    main()