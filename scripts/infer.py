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

import torch
from transformers import BertTokenizer
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from data.utils import layout_seq_to_yolo_txt
# [新增] 导入可视化函数
from data.visualize import draw_layout
import yaml

def main():
    # 1. Load config
    config_path = os.path.join(project_root, "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Load model and tokenizer
    print("Initializing model...")
    # 注意：确保这些参数与训练时一致
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        # 推理时 Loss 权重不重要，但为了初始化不出错可以传入
        reg_loss_weight=config['model'].get('reg_loss_weight', 1.0)
    )

    # 4. Load checkpoint
    output_dir = config['training']['output_dir']
    # [请修改] 这里填写你训练好的权重文件名
    CHECKPOINT_NAME = "/home/610-sty/Layout2Paint/outputs/model_best_val_loss_0.8189.pth" 
    
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
    
    if not os.path.exists(checkpoint_path):
        # 如果指定文件不存在，尝试找最新的
        print(f"Checkpoint {CHECKPOINT_NAME} not found. Searching for latest in {output_dir}...")
        files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
        if files:
            # 简单的按文件名排序找最新的（假设 epoch 数字在文件名里）
            CHECKPOINT_NAME = sorted(files)[-1] 
            checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
            print(f"Found latest checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 

    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 5. Run inference
    # 这里你可以修改为你想测试的诗句
    poem = "山色空蒙雨亦奇，小桥流水绕柴扉。"
    print(f"------------------------------------------------")
    print(f"Input poem: {poem}")

    max_elements = config['model'].get('max_elements', 30)
    
    # 调用 greedy_decode.py 中的函数
    layout = greedy_decode_poem_layout(
        model, tokenizer, poem,
        max_elements=max_elements,
        device=device.type
    )

    print("\nGenerated Layout (Normalized Coords):")
    for i, (cls_id, cx, cy, w, h) in enumerate(layout):
        print(f"  {i+1}: Class {int(cls_id)} (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})")

    # 6. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # [A] 保存为 TXT
    output_txt_path = os.path.join(output_dir, "predicted_layout.txt")
    layout_seq_to_yolo_txt(layout, output_txt_path)
    print(f"\n-> Text layout saved to {output_txt_path}")

    # [B] [新增] 保存为图片可视化
    output_png_path = os.path.join(output_dir, "predicted_layout.png")
    draw_layout(layout, f"PRED: {poem}", output_png_path)
    print(f"-> Visualization saved to {output_png_path}")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()