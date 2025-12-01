# File: scripts/infer.py

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (infer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path)) 
sys.path.insert(0, project_root)
# ---------------------------------------------

import torch
import argparse # [NEW] 使用 argparse 处理命令行参数
from transformers import BertTokenizer
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from data.utils import layout_seq_to_yolo_txt
from data.visualize import draw_layout
import yaml

def main():
    # [NEW] 添加命令行参数解析，方便测试多样性
    parser = argparse.ArgumentParser(description="Inference for Poem2Layout")
    parser.add_argument("--poem", type=str, default="山色空蒙雨亦奇，小桥流水绕柴扉。", help="Input poem")
    parser.add_argument("--mode", type=str, default="sample", choices=["greedy", "sample"], help="Decoding mode: 'greedy' for deterministic, 'sample' for diversity")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K for sampling (only used in sample mode)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint. If None, uses latest/best in output_dir")
    args = parser.parse_args()

    # 1. Load config
    config_path = os.path.join(project_root, "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Init model (确保参数与训练时一致)
    print("Initializing model...")
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        reg_loss_weight=config['model'].get('reg_loss_weight', 1.0)
    )

    # 4. Load checkpoint
    output_dir = config['training']['output_dir']
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # 自动查找最佳或最新模型
        # 优先找 best，其次找最新的 epoch
        best_model_path = None
        latest_model_path = None
        max_epoch = -1
        
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.pth'):
                    full_path = os.path.join(output_dir, f)
                    if "best" in f:
                        best_model_path = full_path
                    elif "epoch_" in f:
                        # 尝试解析 epoch 数字
                        try:
                            parts = f.split('_')
                            epoch_idx = parts.index('epoch')
                            epoch_num = int(parts[epoch_idx+1])
                            if epoch_num > max_epoch:
                                max_epoch = epoch_num
                                latest_model_path = full_path
                        except:
                            pass
        
        checkpoint_path = best_model_path if best_model_path else latest_model_path

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No valid checkpoint found in {output_dir}. Please train the model first.")

    print(f"Loading checkpoint: {checkpoint_path}")
    # [CRITICAL] 可能会因为模型结构改变(加入PriorNet/GridEncoder维度变化)导致加载失败
    # 如果报错，请务必重新训练！
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("\n[ERROR] Checkpoint loading failed. This is likely because the model architecture has changed (e.g., 5x5 -> 8x8 Grid).")
        print("Please DELETE the old checkpoints in 'outputs/' and RETRAIN the model.")
        raise e

    model.to(device)
    model.eval() 

    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 5. Run inference
    poem = args.poem
    print(f"------------------------------------------------")
    print(f"Input poem: {poem}")
    print(f"Mode: {args.mode} (Top-K: {args.top_k})")

    max_elements = config['model'].get('max_elements', 30)
    
    # [MODIFIED] 传入 diversity 参数
    # 注意：这里我们需要修改 greedy_decode.py 的函数签名以接受这些参数，
    # 或者直接在这里修改 greedy_decode.py 内部逻辑（之前已在 greedy_decode.py 中硬编码了默认值）。
    # 为了灵活，建议去修改 greedy_decode.py 的定义，让它接受 mode 和 top_k。
    # 这里假设您已经按照下面的【补充修改】更新了 greedy_decode.py。
    
    layout = greedy_decode_poem_layout(
        model, tokenizer, poem,
        max_elements=max_elements,
        device=device.type,
        mode=args.mode,   # [NEW]
        top_k=args.top_k  # [NEW]
    )

    print("\nGenerated Layout:")
    for i, (cls_id, cx, cy, w, h) in enumerate(layout):
        print(f"  {i+1}: Class {int(cls_id)} (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})")

    # 6. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存时加上 mode 后缀，防止覆盖
    suffix = f"_{args.mode}"
    output_txt_path = os.path.join(output_dir, f"predicted_layout{suffix}.txt")
    layout_seq_to_yolo_txt(layout, output_txt_path)
    print(f"\n-> Text layout saved to {output_txt_path}")

    output_png_path = os.path.join(output_dir, f"predicted_layout{suffix}.png")
    draw_layout(layout, f"PRED ({args.mode}): {poem}", output_png_path)
    print(f"-> Visualization saved to {output_png_path}")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()