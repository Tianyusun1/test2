# File: trainers/trainer.py
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
    # 构造函数新增 test_loader 参数
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

        # <<< NEW: 学习率调度器初始化 >>>
        self.warmup_steps = config['training'].get('warmup_steps', 0)
        self.total_steps = len(train_loader) * self.epochs
        self.global_step = 0
        self.scheduler = self._get_lr_scheduler()
        # <<< END NEW >>>

        # <<< NEW: 最佳模型路径追踪 >>>
        self.current_best_model_path = None
        
        # 损失历史追踪
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_cls_history = []
        self.val_coord_history = []
        self.val_reg_history = [] 
        self.val_iou_history = []
        self.val_count_history = [] 
        self.val_area_history = [] 

    # <<< MODIFIED METHOD: 学习率调度函数 (Warmup + Hold + Decay) >>>
    def _get_lr_scheduler(self):
        """定义带线性 Warmup、5 Epoch Hold 和后续衰减的学习率调度器。"""
        
        # 设定保持最大学习率的 Epoch 数
        N_HOLD_EPOCHS = 5
        steps_per_epoch = len(self.train_loader)
        hold_steps = steps_per_epoch * N_HOLD_EPOCHS
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # 阶段 1: 线性 Warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            
            elif current_step < hold_steps:
                 # 阶段 2: 保持最大学习率 (N_HOLD_EPOCHS 个 Epoch)
                return 1.0 

            else:
                # 阶段 3: 线性衰减
                decay_start_step = hold_steps
                decay_steps = self.total_steps - decay_start_step
                
                if decay_steps > 0:
                    # 衰减因子：从 1.0 线性衰减到 0.0
                    relative_step = current_step - decay_start_step
                    return max(0.0, 1.0 - (relative_step / decay_steps))
                
                return 0.0
            
        return LambdaLR(self.optimizer, lr_lambda)
    # <<< END MODIFIED METHOD >>>


    def _run_epoch(self, data_loader, is_training: bool):
        """处理一个训练/验证/测试轮次"""
        self.model.train() if is_training else self.model.eval()
        
        # 初始化运行损失变量
        total_loss = 0.0
        total_cls_loss = 0.0
        total_coord_loss = 0.0
        total_reg_loss = 0.0 
        total_iou_loss = 0.0
        total_count_loss = 0.0 
        total_area_loss = 0.0
        
        context_manager = contextlib.nullcontext() if is_training else torch.no_grad()
        
        with context_manager:
            for step, batch in enumerate(data_loader):
                # 1. 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layout_seq = batch['layout_seq'].to(self.device)
                layout_mask = batch['layout_mask'].to(self.device)
                
                # 获取 KG 向量并移动到设备
                if 'kg_vector' in batch:
                    kg_vectors = batch['kg_vector'].to(self.device)
                else:
                    batch_size = input_ids.size(0)
                    kg_vectors = torch.zeros((batch_size, 9), device=self.device)
                
                # [NEW] 获取 KG 空间矩阵并移动到设备
                kg_spatial_matrix = None
                if 'kg_spatial_matrix' in batch:
                    kg_spatial_matrix = batch['kg_spatial_matrix'].to(self.device)
                
                # 获取 num_boxes (真实数量)
                num_boxes = batch['num_boxes'].to(self.device)

                # 2. 前向传播
                pred_cls, pred_bbox_ids, pred_coord_float, pred_count = self.model(
                    input_ids, attention_mask, layout_seq, kg_vectors, kg_spatial_matrix=kg_spatial_matrix
                )
                
                # 3. 计算损失 (接收 7 个返回值)
                total_loss_item, cls_loss, coord_loss, reg_loss, iou_loss, count_loss, area_loss = self.model.get_loss( 
                    pred_cls, pred_bbox_ids, pred_coord_float, pred_count, 
                    layout_seq, layout_mask, num_boxes
                )
                
                if is_training:
                    # 4. 训练步骤
                    self.optimizer.zero_grad()
                    total_loss_item.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # <<< NEW: 更新学习率 >>>
                    self.scheduler.step()
                    self.global_step += 1
                    # <<< END NEW >>>
                
                # 累加所有损失
                total_loss += total_loss_item.item()
                total_cls_loss += cls_loss.item()
                total_coord_loss += coord_loss.item()
                total_reg_loss += reg_loss.item()
                total_iou_loss += iou_loss.item()
                total_count_loss += count_loss.item()
                total_area_loss += area_loss.item()
                
                # 5. 打印日志 (Training log)
                if is_training and (step + 1) % self.log_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr'] # 获取当前学习率
                    print(f"Epoch [TRAIN] Step {step+1}/{len(data_loader)} | "
                          f"LR: {current_lr:.6e} | " # 打印当前学习率
                          f"Total: {total_loss_item.item():.4f} | "
                          f"Cls: {cls_loss.item():.4f} | "
                          f"Coord: {coord_loss.item():.4f} | "
                          f"Reg: {reg_loss.item():.4f} | " 
                          f"IoU: {iou_loss.item():.4f} | "
                          f"Count: {count_loss.item():.4f} | "
                          f"Area: {area_loss.item():.4f}") 
        
        # 计算平均损失
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches 
        avg_iou_loss = total_iou_loss / num_batches
        avg_count_loss = total_count_loss / num_batches 
        avg_area_loss = total_area_loss / num_batches 
        
        # 返回 7 个平均损失
        return avg_loss, avg_cls_loss, avg_coord_loss, avg_reg_loss, avg_iou_loss, avg_count_loss, avg_area_loss


    def validate(self):
        """运行验证集上的评估 (并记录详细损失历史)"""
        start_time = time.time()
        print("\n--- Starting Validation ---")
        
        # 接收 7 个平均损失
        avg_val_loss, avg_val_cls, avg_val_coord, avg_val_reg, avg_val_iou, avg_val_count, avg_val_area = self._run_epoch(self.val_loader, is_training=False) 
        
        end_time = time.time()
        print(f"--- Validation Finished in {end_time - start_time:.2f}s ---")
        
        # 记录详细损失历史 (NEW)
        self.val_loss_history.append(avg_val_loss)
        self.val_cls_history.append(avg_val_cls)
        self.val_coord_history.append(avg_val_coord)
        self.val_reg_history.append(avg_val_reg) 
        self.val_iou_history.append(avg_val_iou)
        self.val_count_history.append(avg_val_count)
        self.val_area_history.append(avg_val_area) 
        
        # 输出详细损失信息
        print(f"Validation Avg Loss: Total: {avg_val_loss:.4f} | "
              f"Cls: {avg_val_cls:.4f} | "
              f"Coord: {avg_val_coord:.4f} | "
              f"Reg: {avg_val_reg:.4f} | " 
              f"IoU: {avg_val_iou:.4f} | "
              f"Count: {avg_val_count:.4f} | "
              f"Area: {avg_val_area:.4f}") 
              
        return avg_val_loss

    def test(self):
        """运行测试集上的评估并输出详细损失"""
        start_time = time.time()
        print("\n--- Starting Test Set Evaluation ---")
        
        # 接收 7 个平均损失
        avg_test_loss, avg_test_cls, avg_test_coord, avg_test_reg, avg_test_iou, avg_test_count, avg_test_area = self._run_epoch(self.test_loader, is_training=False) 
        
        end_time = time.time()
        print(f"--- Test Finished in {end_time - start_time:.2f}s ---")
        
        # 输出详细损失信息
        print(f"Test Avg Loss: Total: {avg_test_loss:.4f} | "
              f"Cls: {avg_test_cls:.4f} | "
              f"Coord: {avg_test_coord:.4f} | "
              f"Reg: {avg_test_reg:.4f} | " 
              f"IoU: {avg_test_iou:.4f} | "
              f"Count: {avg_test_count:.4f} | "
              f"Area: {avg_test_area:.4f}") 
              
        return avg_test_loss

    def _run_inference_example(self, epoch):
        """运行固定样例的推理并保存可视化图片"""
        print(f"\n--- Running Inference Example for Epoch {epoch+1} ---")
        poem_text = self.example_poem['poem']
        max_elements = self.config['model'].get('max_elements', 30)
        
        # 1. 调用贪婪解码生成布局
        layout = greedy_decode_poem_layout(
            self.model, 
            self.tokenizer, 
            poem_text, 
            max_elements=max_elements,
            device=self.device.type
        )
        
        # 2. 可视化预测布局
        output_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_layout_pred.png")
        draw_layout(layout, f"PRED: {poem_text}", output_path)
        print(f"-> Generated layout saved to {output_path}")

        # 3. 可视化真实布局（使用原始 boxes 数据）
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
        
        # 1. 绘制总损失 (Train & Val)
        plt.plot(epochs, self.train_loss_history, label='Train Total Loss', marker='.', linestyle='-')
        plt.plot(epochs, self.val_loss_history, label='Validation Total Loss', marker='.', linestyle='-')
        
        # 2. 绘制详细验证损失
        if len(self.val_cls_history) > 1:
            plt.plot(epochs, self.val_cls_history, label='Val Cls Loss', linestyle='--')
            plt.plot(epochs, self.val_coord_history, label='Val Coord Loss', linestyle='--')
            plt.plot(epochs, self.val_reg_history, label='Val Reg Loss', linestyle='--') 
            plt.plot(epochs, self.val_iou_history, label='Val IoU Loss', linestyle='--')
            plt.plot(epochs, self.val_count_history, label='Val Count Loss', linestyle='--')
            plt.plot(epochs, self.val_area_history, label='Val Area Loss', linestyle='--') 

        plt.title('Loss Trajectory Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        try:
            plt.savefig(self.plot_path)
            plt.close() # 关闭 figure 释放内存
            print(f"-> Loss history plot updated and saved to {self.plot_path}")
        except Exception as e:
            print(f"[Warning] Could not save loss plot (Is matplotlib/PIL installed?): {e}")


    def train(self):
        """主训练循环"""
        print("--- Starting Full Training ---")
        best_val_loss = float('inf')
        
        # 打印总步数和初始学习率，方便追踪
        print(f"Total training steps: {self.total_steps}, Warmup steps: {self.warmup_steps}, Base LR: {self.lr:.6e}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n==================== Epoch {epoch+1}/{self.epochs} | Training ====================")
            
            # 运行训练轮次
            avg_train_loss, _, _, _, _, _, _ = self._run_epoch(self.train_loader, is_training=True) 
            self.train_loss_history.append(avg_train_loss)
            
            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1} finished. Avg Training Loss: {avg_train_loss:.4f} "
                  f"({epoch_end_time - epoch_start_time:.2f}s)")
            
            # 运行验证轮次
            avg_val_loss = self.validate() 

            # Step 1: 损失轨迹绘图 (每 Epoch 执行)
            self._plot_loss_history()
            
            # Step 2: 样例推理与可视化
            if (epoch + 1) % self.visualize_every == 0:
                self._run_inference_example(epoch)

            # Step 3: 定期检查点保存
            if (epoch + 1) % self.save_every == 0:
                 # 运行测试集评估
                 self.test()
                 
                 # 保存带 Epoch 编号的检查点 (这部分仍然有意义，用于历史和恢复)
                 model_name = f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
                 checkpoint_path = os.path.join(self.output_dir, model_name)
                 torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, checkpoint_path)
                 print(f"-> Checkpoint saved to {checkpoint_path}")

            # Step 4: 最佳模型保存 (仅保留当前最佳，删除旧文件) <<< MODIFIED LOGIC
            if avg_val_loss < best_val_loss:
                print("-> New best validation loss achieved. Replacing previous best model.")
                
                # 1. 记录旧的最佳路径
                prev_best_path = self.current_best_model_path
                
                # 2. 更新最佳损失和新路径
                best_val_loss = avg_val_loss
                # 使用包含损失的新名称，但我们只会保留最新的这个
                model_name = f"model_best_val_loss_{avg_val_loss:.4f}.pth" 
                new_best_path = os.path.join(self.output_dir, model_name)
                
                # 3. 保存新模型
                torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, new_best_path)
                print(f"-> New best model saved to {new_best_path}")

                # 4. 删除旧的最佳模型（如果存在且不是新保存的模型）
                if prev_best_path and os.path.exists(prev_best_path) and prev_best_path != new_best_path:
                    try:
                        os.remove(prev_best_path)
                        print(f"-> Deleted previous best model: {prev_best_path}")
                    except OSError as e:
                        print(f"[Warning] Could not delete previous best model {prev_best_path}: {e}")
                
                # 5. 更新当前最佳路径
                self.current_best_model_path = new_best_path
                 
        print("--- Training Completed ---")