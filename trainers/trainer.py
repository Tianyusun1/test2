# File: tianyusun1/test2/test2-2.0/trainers/trainer.py (FINAL: WEIGHTS + KG PRINT + LOCATION GRID)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (trainer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (trainer.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
from collections import Counter
import numpy as np 
import time
import contextlib 
import torch.nn.utils 
from torch.optim.lr_scheduler import LambdaLR 

# --- NEW IMPORTS for Visualization/Inference/Plotting ---
from inference.greedy_decode import greedy_decode_poem_layout 
from data.visualize import draw_layout
import matplotlib.pyplot as plt 
# ---------------------------------------------


# =========================================================================
# LayoutTrainer 类定义 (实现完整的训练、验证和测试逻辑)
# =========================================================================

class LayoutTrainer:
    """负责训练循环、优化器管理、日志记录和模型保存。"""
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        print(f"Trainer initialized on device: {self.device}")
        
        # 新增推理所需参数
        self.tokenizer = tokenizer
        self.example_poem = example_poem
        
        # 训练和保存频率设置
        self.lr = config['training']['learning_rate'] # Store base LR
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.lr
        )
        self.epochs = config['training']['epochs']
        self.output_dir = config['training']['output_dir']
        self.log_steps = config['training'].get('log_steps', 10)
        self.save_every = config['training'].get('save_every', 10)
        self.visualize_every = 1 
        self.plot_path = os.path.join(self.output_dir, "loss_trajectory.png")
        os.makedirs(self.output_dir, exist_ok=True)

        # 学习率调度器初始化
        self.warmup_steps = config['training'].get('warmup_steps', 0)
        self.total_steps = len(train_loader) * self.epochs
        self.global_step = 0
        self.scheduler = self._get_lr_scheduler()

        # 最佳模型路径追踪
        self.current_best_model_path = None
        
        # 损失历史追踪 (扩充以监控所有新 Loss)
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_reg_history = [] 
        self.val_iou_history = []
        self.val_area_history = [] 
        self.val_relation_history = [] # [NEW]
        self.val_overlap_history = []  # [NEW]
        self.val_size_history = []     # [NEW]

    def _get_lr_scheduler(self):
        """定义带线性 Warmup、5 Epoch Hold 和后续衰减的学习率调度器。"""
        N_HOLD_EPOCHS = 5
        steps_per_epoch = len(self.train_loader)
        hold_steps = steps_per_epoch * N_HOLD_EPOCHS
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            elif current_step < hold_steps:
                return 1.0 
            else:
                decay_start_step = hold_steps
                decay_steps = self.total_steps - decay_start_step
                if decay_steps > 0:
                    relative_step = current_step - decay_start_step
                    return max(0.0, 1.0 - (relative_step / decay_steps))
                return 0.0
            
        return LambdaLR(self.optimizer, lr_lambda)

    def _run_epoch(self, data_loader, is_training: bool):
        """处理一个训练/验证/测试轮次"""
        self.model.train() if is_training else self.model.eval()
        
        # 初始化运行损失变量
        total_loss = 0.0
        total_reg_loss = 0.0 
        total_iou_loss = 0.0
        total_area_loss = 0.0
        total_relation_loss = 0.0 # [NEW]
        total_overlap_loss = 0.0  # [NEW]
        total_size_loss = 0.0     # [NEW]
        
        context_manager = contextlib.nullcontext() if is_training else torch.no_grad()
        
        with context_manager:
            for step, batch in enumerate(data_loader):
                # 1. 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                kg_class_ids = batch['kg_class_ids'].to(self.device) 
                padding_mask = batch['padding_mask'].to(self.device) 
                target_boxes = batch['target_boxes'].to(self.device) 
                loss_mask = batch['loss_mask'].to(self.device)       
                
                # [NEW] 提取 KG Weights (用于 Size Loss)
                kg_class_weights = None
                if 'kg_class_weights' in batch:
                    kg_class_weights = batch['kg_class_weights'].to(self.device)

                # KG Spatial Matrix
                kg_spatial_matrix = None
                if 'kg_spatial_matrix' in batch:
                    kg_spatial_matrix = batch['kg_spatial_matrix'].to(self.device)
                
                # [NEW] 提取 Location Grids (用于 Position Guide)
                location_grids = None
                if 'location_grids' in batch:
                    location_grids = batch['location_grids'].to(self.device)
                
                # 2. 前向传播
                _, _, pred_boxes, _ = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    kg_class_ids=kg_class_ids, 
                    padding_mask=padding_mask, 
                    kg_spatial_matrix=kg_spatial_matrix,
                    location_grids=location_grids # [NEW] 传入位置网格
                )
                
                # 3. 计算损失
                # [MODIFIED] 传入 kg_spatial_matrix 和 kg_class_weights
                loss_tuple = self.model.get_loss(
                    pred_cls=None, 
                    pred_bbox_ids=None, 
                    pred_boxes=pred_boxes, 
                    pred_count=None, 
                    layout_seq=None, 
                    layout_mask=loss_mask, 
                    num_boxes=batch['num_boxes'].to(self.device), 
                    target_coords_gt=target_boxes,
                    kg_spatial_matrix=kg_spatial_matrix, # [NEW] Relation/Overlap Loss
                    kg_class_weights=kg_class_weights    # [NEW] Size Loss (Implicit object shrinking)
                )
                
                # 解包损失: total, relation, overlap, reg, iou, size, area
                loss_total_item, relation_loss, overlap_loss, reg_loss, iou_loss, size_loss, area_loss = loss_tuple
                
                if is_training:
                    # 4. 训练步骤
                    self.optimizer.zero_grad()
                    loss_total_item.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    self.scheduler.step()
                    self.global_step += 1
                
                # 累加所有损失
                total_loss += loss_total_item.item()
                total_reg_loss += reg_loss.item()
                total_iou_loss += iou_loss.item()
                total_area_loss += area_loss.item()
                total_relation_loss += relation_loss.item()
                total_overlap_loss += overlap_loss.item()
                total_size_loss += size_loss.item()
                
                # 5. 打印日志 (Training log)
                if is_training and (step + 1) % self.log_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [TRAIN] Step {step+1}/{len(data_loader)} | "
                          f"LR: {current_lr:.6e} | "
                          f"Total: {loss_total_item.item():.4f} | "
                          f"Rel: {relation_loss.item():.4f} | "  # [NEW]
                          f"Over: {overlap_loss.item():.4f} | " # [NEW]
                          f"Reg: {reg_loss.item():.4f} | " 
                          f"IoU: {iou_loss.item():.4f}")
        
        # 计算平均损失
        num_batches = len(data_loader)
        return (
            total_loss / num_batches,
            total_relation_loss / num_batches,
            total_overlap_loss / num_batches,
            total_reg_loss / num_batches,
            total_iou_loss / num_batches,
            total_size_loss / num_batches,
            total_area_loss / num_batches
        )

    def validate(self):
        """运行验证集上的评估"""
        start_time = time.time()
        print("\n--- Starting Validation ---")
        
        avg_losses = self._run_epoch(self.val_loader, is_training=False)
        # 解包 7 个值
        avg_val_loss, avg_rel, avg_over, avg_reg, avg_iou, avg_size, avg_area = avg_losses
        
        end_time = time.time()
        print(f"--- Validation Finished in {end_time - start_time:.2f}s ---")
        
        self.val_loss_history.append(avg_val_loss)
        self.val_reg_history.append(avg_reg) 
        self.val_iou_history.append(avg_iou)
        self.val_area_history.append(avg_area) 
        self.val_relation_history.append(avg_rel) # [NEW]
        self.val_overlap_history.append(avg_over) # [NEW]
        self.val_size_history.append(avg_size)    # [NEW]
        
        print(f"Val Avg Loss: Total: {avg_val_loss:.4f} | "
              f"Rel: {avg_rel:.4f} | "
              f"Over: {avg_over:.4f} | "
              f"Reg: {avg_reg:.4f} | " 
              f"IoU: {avg_iou:.4f} | "
              f"Area: {avg_area:.4f}") 
              
        return avg_val_loss

    def test(self):
        """运行测试集上的评估"""
        start_time = time.time()
        print("\n--- Starting Test Set Evaluation ---")
        
        avg_losses = self._run_epoch(self.test_loader, is_training=False)
        avg_test_loss, avg_rel, avg_over, avg_reg, avg_iou, avg_size, avg_area = avg_losses
        
        end_time = time.time()
        print(f"--- Test Finished in {end_time - start_time:.2f}s ---")
        
        print(f"Test Avg Loss: Total: {avg_test_loss:.4f} | "
              f"Rel: {avg_rel:.4f} | "
              f"Over: {avg_over:.4f} | "
              f"Reg: {avg_reg:.4f} | " 
              f"IoU: {avg_iou:.4f}") 
              
        return avg_test_loss

    def _run_inference_example(self, epoch):
        """运行固定样例的推理并保存可视化图片"""
        print(f"\n--- Running Inference Example for Epoch {epoch+1} ---")
        poem_text = self.example_poem['poem']
        
        # === [MODIFIED] 打印测试样本的 KG 信息 ===
        print(f"Poem: {poem_text}")
        try:
            # 处理 Subset 包装的情况
            ds = self.train_loader.dataset
            if hasattr(ds, 'dataset'):
                ds = ds.dataset
                
            if hasattr(ds, 'pkg'):
                pkg = ds.pkg
                # 1. 打印物体
                kg_vector = pkg.extract_visual_feature_vector(poem_text)
                # 转为 ID 列表
                indices = torch.nonzero(kg_vector).squeeze(1).tolist()
                # 内部 index 0-8 对应 ID 2-10
                ids = [i + 2 for i in indices]
                print(f"KG Objects (IDs): {ids}")
                
                # 2. 打印空间矩阵
                print("KG Spatial Matrix:")
                matrix = pkg.extract_spatial_matrix(poem_text)
                print(matrix)
        except Exception as e:
            print(f"[Warning] Could not print KG info: {e}")
        # =========================================

        max_elements = self.config['model'].get('max_elements', 30)
        
        layout = greedy_decode_poem_layout(
            self.model, 
            self.tokenizer, 
            poem_text, 
            max_elements=max_elements,
            device=self.device.type
        )
        
        output_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_layout_pred.png")
        draw_layout(layout, f"PRED: {poem_text}", output_path)
        print(f"-> Generated layout saved to {output_path}")

        true_boxes = self.example_poem['boxes']
        true_layout_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_layout_true.png")
        draw_layout(true_boxes, f"TRUE: {poem_text}", true_layout_path)
        print(f"-> True layout saved to {true_layout_path}")
        print("---------------------------------------------------")

    def _plot_loss_history(self):
        """绘制并保存损失变化轨迹图"""
        if not self.train_loss_history:
            return

        epochs = range(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, self.train_loss_history, label='Train Total', marker='.', linestyle='-')
        plt.plot(epochs, self.val_loss_history, label='Val Total', marker='.', linestyle='-')
        
        if len(self.val_reg_history) > 1:
            plt.plot(epochs, self.val_relation_history, label='Relation(KG)', linestyle=':') # [NEW]
            plt.plot(epochs, self.val_overlap_history, label='Overlap', linestyle=':')     # [NEW]
            plt.plot(epochs, self.val_reg_history, label='Reg', linestyle='--', alpha=0.5) 
            plt.plot(epochs, self.val_iou_history, label='IoU', linestyle='--', alpha=0.5)

        plt.title('Loss Trajectory (KG & Logic Enhanced)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        try:
            plt.savefig(self.plot_path)
            plt.close()
            print(f"-> Loss history plot updated and saved to {self.plot_path}")
        except Exception as e:
            print(f"[Warning] Could not save loss plot: {e}")

    def train(self):
        """主训练循环"""
        print("--- Starting Full Training ---")
        best_val_loss = float('inf')
        
        print(f"Total training steps: {self.total_steps}, Warmup steps: {self.warmup_steps}, Base LR: {self.lr:.6e}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n==================== Epoch {epoch+1}/{self.epochs} | Training ====================")
            
            avg_train_loss = self._run_epoch(self.train_loader, is_training=True)[0]
            self.train_loss_history.append(avg_train_loss)
            
            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1} finished. Avg Training Loss: {avg_train_loss:.4f} ({epoch_end_time - epoch_start_time:.2f}s)")
            
            avg_val_loss = self.validate() 

            self._plot_loss_history()
            
            if (epoch + 1) % self.visualize_every == 0:
                self._run_inference_example(epoch)

            if (epoch + 1) % self.save_every == 0:
                 self.test()
                 model_name = f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
                 checkpoint_path = os.path.join(self.output_dir, model_name)
                 torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, checkpoint_path)
                 print(f"-> Checkpoint saved to {checkpoint_path}")

            if avg_val_loss < best_val_loss:
                print("-> New best validation loss achieved. Replacing previous best model.")
                prev_best_path = self.current_best_model_path
                best_val_loss = avg_val_loss
                model_name = f"model_best_val_loss_{avg_val_loss:.4f}.pth" 
                new_best_path = os.path.join(self.output_dir, model_name)
                
                torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, new_best_path)
                print(f"-> New best model saved to {new_best_path}")

                if prev_best_path and os.path.exists(prev_best_path) and prev_best_path != new_best_path:
                    try:
                        os.remove(prev_best_path)
                        print(f"-> Deleted previous best model: {prev_best_path}")
                    except OSError as e:
                        print(f"[Warning] Could not delete previous best model: {e}")
                
                self.current_best_model_path = new_best_path
                 
        print("--- Training Completed ---")