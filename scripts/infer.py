# File: tianyusun1/test2/test2-4.0/scripts/infer.py

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (infer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path)) 
sys.path.insert(0, project_root)
# ---------------------------------------------

import torch
import argparse 
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
    # [NEW V4.2] 新增采样次数参数，用于展示 CVAE 的多样性
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per poem (CVAE diversity check)")
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
    print(f"Initializing model with latent_dim={config['model'].get('latent_dim', 32)}...")
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        # === [NEW] 传入 CVAE 参数 ===
        latent_dim=config['model'].get('latent_dim', 32), # 必须传入，否则加载权重会报错
        # === Loss Weights (Inference时不使用，但需占位) ===
        reg_loss_weight=config['model'].get('reg_loss_weight', 1.0)
    )

    # 4. Load checkpoint
    output_dir = config['training']['output_dir']
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # 自动查找最佳或最新模型
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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("\n[ERROR] Checkpoint loading failed. This is likely because the model architecture has changed (e.g. latent_dim added).")
        print("Please DELETE the old checkpoints in 'outputs/' and RETRAIN the model.")
        raise e

    model.to(device)
    model.eval() 

    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 5. Run inference loop
    poem = args.poem
    print(f"------------------------------------------------")
    print(f"Input poem: {poem}")
    print(f"Mode: {args.mode} (Top-K: {args.top_k})")
    print(f"Generating {args.num_samples} sample(s)...")

    max_elements = config['model'].get('max_elements', 30)
    os.makedirs(output_dir, exist_ok=True)
    
    # [NEW] 循环生成多个样本
    for i in range(args.num_samples):
        # 每次调用 greedy_decode，CVAE 内部都会重新随机采样 z ~ N(0, I)
        layout = greedy_decode_poem_layout(
            model, tokenizer, poem,
            max_elements=max_elements,
            device=device.type,
            mode=args.mode,
            top_k=args.top_k
        )

        print(f"\n--- Sample {i+1} Generated Layout ---")
        for j, (cls_id, cx, cy, w, h) in enumerate(layout):
            print(f"  {j+1}: Class {int(cls_id)} (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})")

        # 6. Save outputs
        # 文件名增加 sample 编号
        suffix = f"_{args.mode}"
        if args.num_samples > 1:
            suffix += f"_sample{i+1}"
            
        output_txt_path = os.path.join(output_dir, f"predicted_layout{suffix}.txt")
        layout_seq_to_yolo_txt(layout, output_txt_path)
        print(f"-> Text layout saved to {output_txt_path}")

        output_png_path = os.path.join(output_dir, f"predicted_layout{suffix}.png")
        draw_layout(layout, f"PRED ({args.mode} #{i+1}): {poem}", output_png_path)
        print(f"-> Visualization saved to {output_png_path}")

    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()