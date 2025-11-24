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

# --- NEW IMPORTS for Visualization/Inference/Plotting ---
from inference.greedy_decode import greedy_decode_poem_layout 
from data.visualize import draw_layout
import matplotlib.pyplot as plt # <--- NEW IMPORT for plotting
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
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate']
        )
        self.epochs = config['training']['epochs']
        self.output_dir = config['training']['output_dir']
        self.log_steps = config['training'].get('log_steps', 10)
        self.save_every = config['training'].get('save_every', 10)
        self.visualize_every = 1 # <--- NEW: Hardcoded for every epoch
        self.plot_path = os.path.join(self.output_dir, "loss_trajectory.png") # Loss plot path
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 损失历史追踪 (NEW)
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_cls_history = []
        self.val_coord_history = []
        self.val_reg_history = [] # **回归损失历史**
        self.val_iou_history = []
        self.val_count_history = [] # **Count 损失历史**
        self.val_area_history = [] # <<<< 新增: Area 损失历史


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
        total_area_loss = 0.0 # <<<< 新增: Area 损失
        
        context_manager = contextlib.nullcontext() if is_training else torch.no_grad()
        
        with context_manager:
            for step, batch in enumerate(data_loader):
                # 1. 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layout_seq = batch['layout_seq'].to(self.device)
                layout_mask = batch['layout_mask'].to(self.device)
                
                # --- FIX: 获取 KG 向量并移动到设备 ---
                if 'kg_vector' in batch:
                    kg_vectors = batch['kg_vector'].to(self.device)
                else:
                    # Fallback
                    batch_size = input_ids.size(0)
                    kg_vectors = torch.zeros((batch_size, 9), device=self.device)
                # -----------------------------------
                
                # 获取 num_boxes (真实数量)
                num_boxes = batch['num_boxes'].to(self.device)

                # 2. 前向传播
                pred_cls, pred_bbox_ids, pred_coord_float, pred_count = self.model(
                    input_ids, attention_mask, layout_seq, kg_vectors
                )
                
                # 3. 计算损失 (FIXED: 接收 7 个返回值)
                total_loss_item, cls_loss, coord_loss, reg_loss, iou_loss, count_loss, area_loss = self.model.get_loss( # <<<< 接收 area_loss
                    pred_cls, pred_bbox_ids, pred_coord_float, pred_count, 
                    layout_seq, layout_mask, num_boxes
                )
                
                if is_training:
                    # 4. 训练步骤
                    self.optimizer.zero_grad()
                    total_loss_item.backward()
                    self.optimizer.step()
                
                # 累加所有损失
                total_loss += total_loss_item.item()
                total_cls_loss += cls_loss.item()
                total_coord_loss += coord_loss.item()
                total_reg_loss += reg_loss.item()
                total_iou_loss += iou_loss.item()
                total_count_loss += count_loss.item()
                total_area_loss += area_loss.item() # <<<< 累加 Area Loss
                
                # 5. 打印日志 (Training log)
                if is_training and (step + 1) % self.log_steps == 0:
                    print(f"Epoch [TRAIN] Step {step+1}/{len(data_loader)} | "
                          f"Total: {total_loss_item.item():.4f} | "
                          f"Cls: {cls_loss.item():.4f} | "
                          f"Coord: {coord_loss.item():.4f} | "
                          f"Reg: {reg_loss.item():.4f} | " 
                          f"IoU: {iou_loss.item():.4f} | "
                          f"Count: {count_loss.item():.4f} | "
                          f"Area: {area_loss.item():.4f}") # <<<< 打印 Area Loss
        
        # 计算平均损失
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches 
        avg_iou_loss = total_iou_loss / num_batches
        avg_count_loss = total_count_loss / num_batches 
        avg_area_loss = total_area_loss / num_batches # <<<< 计算 Avg Area Loss
        
        # 返回 7 个平均损失
        return avg_loss, avg_cls_loss, avg_coord_loss, avg_reg_loss, avg_iou_loss, avg_count_loss, avg_area_loss # <<<< 返回


    def validate(self):
        """运行验证集上的评估 (并记录详细损失历史)"""
        start_time = time.time()
        print("\n--- Starting Validation ---")
        
        # 接收 7 个平均损失
        avg_val_loss, avg_val_cls, avg_val_coord, avg_val_reg, avg_val_iou, avg_val_count, avg_val_area = self._run_epoch(self.val_loader, is_training=False) # <<<< 接收 avg_val_area
        
        end_time = time.time()
        print(f"--- Validation Finished in {end_time - start_time:.2f}s ---")
        
        # 记录详细损失历史 (NEW)
        self.val_loss_history.append(avg_val_loss)
        self.val_cls_history.append(avg_val_cls)
        self.val_coord_history.append(avg_val_coord)
        self.val_reg_history.append(avg_val_reg) 
        self.val_iou_history.append(avg_val_iou)
        self.val_count_history.append(avg_val_count)
        self.val_area_history.append(avg_val_area) # <<<< 记录 Area Loss
        
        # 输出详细损失信息
        print(f"Validation Avg Loss: Total: {avg_val_loss:.4f} | "
              f"Cls: {avg_val_cls:.4f} | "
              f"Coord: {avg_val_coord:.4f} | "
              f"Reg: {avg_val_reg:.4f} | " 
              f"IoU: {avg_val_iou:.4f} | "
              f"Count: {avg_val_count:.4f} | "
              f"Area: {avg_val_area:.4f}") # <<<< 打印
              
        return avg_val_loss

    def test(self):
        """运行测试集上的评估并输出详细损失"""
        start_time = time.time()
        print("\n--- Starting Test Set Evaluation ---")
        
        # 接收 7 个平均损失
        avg_test_loss, avg_test_cls, avg_test_coord, avg_test_reg, avg_test_iou, avg_test_count, avg_test_area = self._run_epoch(self.test_loader, is_training=False) # <<<< 接收
        
        end_time = time.time()
        print(f"--- Test Finished in {end_time - start_time:.2f}s ---")
        
        # 输出详细损失信息
        print(f"Test Avg Loss: Total: {avg_test_loss:.4f} | "
              f"Cls: {avg_test_cls:.4f} | "
              f"Coord: {avg_test_coord:.4f} | "
              f"Reg: {avg_test_reg:.4f} | " 
              f"IoU: {avg_test_iou:.4f} | "
              f"Count: {avg_test_count:.4f} | "
              f"Area: {avg_test_area:.4f}") # <<<< 打印
              
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
            plt.plot(epochs, self.val_area_history, label='Val Area Loss', linestyle='--') # <<<< 绘制 Area Loss

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
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n==================== Epoch {epoch+1}/{self.epochs} | Training ====================")
            
            # 运行训练轮次
            # **接收 7 个损失项，只取第一个 (总损失) 来记录**
            avg_train_loss, _, _, _, _, _, _ = self._run_epoch(self.train_loader, is_training=True) # <<<< 忽略其他损失
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

            # Step 3: 检查点保存和测试集评估
            if (epoch + 1) % self.save_every == 0:
                 # 运行测试集评估
                 self.test()
                 
                 # 保存带 Epoch 编号的检查点
                 model_name = f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
                 checkpoint_path = os.path.join(self.output_dir, model_name)
                 torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, checkpoint_path)
                 print(f"-> Checkpoint saved to {checkpoint_path}")

            # Step 4: 最佳模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_name = f"model_best_val_loss_{avg_val_loss:.4f}.pth"
                checkpoint_path = os.path.join(self.output_dir, model_name)
                torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch+1, 'val_loss': avg_val_loss}, checkpoint_path)
                print(f"-> New best model saved to {checkpoint_path}")
                 
        print("--- Training Completed ---")