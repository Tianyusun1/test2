# File: tianyusun1/test2/test2-5.2/trainers/rl_trainer.py (V5.22: Robust & Complete)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt
import numpy as np

class RLTrainer(LayoutTrainer):
    """
    强化学习训练器 (RL Fine-tuning Trainer) - V5.22 完备版
    
    设计理念:
    1. Anti-Collapse: 通过 Entropy Regularization 和 Free Bits 防止模式坍塌。
    2. Balanced Exploration: 利用 SCST 算法，在 Greedy Baseline 基础上进行探索。
    3. Hybrid Loss: 结合 RL Loss (灵活性) + Supervised Loss (准确性) + KL Loss (多样性)。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # === 1. RL 优化器配置 ===
        # 建议使用稍大的学习率 (2e-5) 以便在 RL 阶段能跳出局部最优
        self.rl_lr = float(config['training'].get('rl_learning_rate', 2e-5))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # === 2. 奖励权重体系 (Reward Engineering) ===
        reward_cfg = config['training'].get('reward_weights', {})
        
        # [核心配置]
        self.w_iou = float(reward_cfg.get('iou', 3.0))              # 高权重: 保证物体大小和位置的语义准确性
        self.w_rel = float(reward_cfg.get('relation', 1.0))         # 低权重: 防止过拟合特定空间关系
        self.w_dispersion = float(reward_cfg.get('dispersion', 1.0)) # 中高权重: 鼓励物体散开
        self.w_overlap = float(reward_cfg.get('overlap', -1.0))     # 惩罚项: 禁止重叠
        self.w_composition = float(reward_cfg.get('composition', 0.5)) # 美学项: 三分法构图

        # === 3. 训练策略参数 ===
        self.save_every = 50          # [用户指定] 每 50 轮保存一次
        self.entropy_weight = 0.05    # [关键] 熵奖励权重，越大越"散"
        self.kl_threshold = 2.0       # [关键] Free Bits 阈值 (nats)
        self.supervised_weight = 1.0  # [关键] 监督信号锚点权重

        # === 4. 日志与监控 ===
        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer] Initialized V5.22 (Robust).")
        print(f"  > LR: {self.rl_lr:.2e}")
        print(f"  > Weights: IoU={self.w_iou}, Rel={self.w_rel}, Disp={self.w_dispersion}, Over={self.w_overlap}")
        print(f"  > Strategy: Ent_W={self.entropy_weight}, KL_Thres={self.kl_threshold}, Sup_W={self.supervised_weight}")
        print(f"  > Save Interval: Every {self.save_every} Epochs")

    def compute_reward(self, pred_boxes, batch):
        """
        计算每个样本的综合奖励值。
        Input: pred_boxes [B, T, 4] (cx, cy, w, h)
        Output: total_reward [B]
        """
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # 提取 Batch 信息
        loss_mask = batch['loss_mask']          # 1.0 = 有效物体
        target_boxes = batch['target_boxes']
        kg_spatial_matrix = batch['kg_spatial_matrix']
        kg_class_ids = batch['kg_class_ids']
        
        # 初始化奖励累加器
        obj_rewards = torch.zeros(B, T, device=device)
        
        # --- 1. IoU Reward (理解准确度) ---
        iou = self._calculate_iou(pred_boxes, target_boxes)
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # --- 2. Relation Reward (空间逻辑) ---
        # 即使没有 GT，只要符合 KG 逻辑也给分
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids)
        r_rel = rel_scores * self.w_rel
        obj_rewards += r_rel

        # --- 3. Dispersion Reward (分散度 - 最近邻距离) ---
        centers = pred_boxes[..., :2]
        dists = torch.cdist(centers, centers)
        # 把对角线设为大值，避免自己和自己比较
        eye = torch.eye(T, device=device).unsqueeze(0)
        dists = dists + eye * 10.0
        # 找最近的邻居
        min_dist, _ = dists.min(dim=2)
        # 截断奖励，避免为了分散而分散到无限远
        disp_score = torch.clamp(min_dist, max=0.3) 
        r_disp = disp_score * self.w_dispersion
        obj_rewards += r_disp

        # --- 4. Composition Reward (构图美学 - 三分法) ---
        cx, cy = centers[..., 0], centers[..., 1]
        # 距离 1/3 或 2/3 线越近越好
        dist_x = torch.min(torch.abs(cx - 0.333), torch.abs(cx - 0.667))
        dist_y = torch.min(torch.abs(cy - 0.333), torch.abs(cy - 0.667))
        # 距离越小，奖励越高 (max 0.2)
        r_comp = (0.2 - (dist_x + dist_y)).clamp(min=0) * self.w_composition
        obj_rewards += r_comp 

        # --- 5. Border Penalty (边缘惩罚 - 防止死守角落) ---
        dist_to_edge = torch.min(centers, 1.0 - centers)
        # 如果离边缘小于 5%，扣分
        is_too_close = (dist_to_edge < 0.05).float().sum(dim=-1)
        r_border = is_too_close * -0.5
        obj_rewards += r_border

        # --- 6. Overlap Penalty (物理重叠惩罚) ---
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over

        # --- 7. 统计与汇总 ---
        self.last_reward_stats = {
            'IoU': (r_iou.sum() / (loss_mask.sum() + 1e-6)).item(),
            'Rel': (r_rel.sum() / (B * T)).item(), 
            'Disp': disp_score.mean().item(),
            'Comp': r_comp.mean().item(),
            'Over': overlap_penalty.mean().item()
        }

        # 对每个样本内的物体取平均，得到 Batch 级的奖励
        total_sample_reward = obj_rewards.sum(dim=1) / (T + 1e-6)
        return total_sample_reward

    # === 辅助计算函数 (保持高效) ===
    def _calculate_iou(self, pred, target):
        """计算 Batch IoU"""
        p_x1, p_y1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        p_x2, p_y2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        t_x1, t_y1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        t_x2, t_y2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        
        i_x1 = torch.max(p_x1, t_x1); i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2); i_y2 = torch.min(p_y2, t_y2)
        
        i_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
        p_area = pred[..., 2] * pred[..., 3]
        t_area = target[..., 2] * target[..., 3]
        u_area = p_area + t_area - i_area
        return i_area / (u_area + 1e-6)

    def _calculate_relation_reward(self, boxes, matrix, class_ids):
        """计算 KG 关系一致性"""
        B, T, _ = boxes.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        if matrix is None: return rewards

        for b in range(B):
            for i in range(T):
                cid_i = class_ids[b, i].item(); idx_i = int(cid_i) - 2
                if not (0 <= idx_i < 9): continue
                for j in range(T):
                    if i == j: continue
                    cid_j = class_ids[b, j].item(); idx_j = int(cid_j) - 2
                    if not (0 <= idx_j < 9): continue
                    
                    rel = matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue
                    
                    box_i = boxes[b, i]; box_j = boxes[b, j]
                    reward_val = 0.0
                    
                    # 密集奖励逻辑: 只要方向对就给满分，方向不对按距离惩罚
                    if rel == 1: # ABOVE
                        diff = box_j[1] - box_i[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff
                    elif rel == 2: # BELOW
                        diff = box_i[1] - box_j[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff
                    elif rel == 3: # INSIDE
                        dx = abs(box_i[0] - box_j[0]); dy = abs(box_i[1] - box_j[1])
                        if dx < box_j[2]/2 and dy < box_j[3]/2: reward_val = 1.0
                    elif rel == 4: # SURROUNDS
                        dx = abs(box_i[0] - box_j[0]); dy = abs(box_i[1] - box_j[1])
                        if dx < box_i[2]/2 and dy < box_i[3]/2: reward_val = 1.0
                    elif rel == 5: # ON TOP
                        diff = box_j[1] - box_i[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff

                    rewards[b, i] += reward_val
        return torch.clamp(rewards, min=-1.0, max=3.0)

    def _calculate_overlap_penalty(self, boxes):
        """计算重叠惩罚"""
        B, T, _ = boxes.shape
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers)
        
        # 判定距离阈值 (例如 0.05)
        too_close = (dist < 0.05).float()
        mask = torch.eye(T, device=boxes.device).unsqueeze(0).expand(B, -1, -1)
        too_close = too_close * (1 - mask)
        penalty = too_close.sum(dim=2) 
        return penalty

    def _plot_reward_history(self):
        """绘制奖励曲线"""
        if not self.reward_history: return
        epochs = range(1, len(self.reward_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.reward_history, label='Avg Epoch Reward', color='purple', marker='o', linestyle='-')
        plt.title('RL Training Reward Trajectory', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        try:
            plt.savefig(self.plot_path_reward)
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not save reward plot: {e}")

    def train_rl_epoch(self, epoch):
        """
        执行一个 Epoch 的 SCST 训练 + 熵正则 + 监督修正
        """
        self.model.train()
        total_reward = 0
        steps = 0
        
        print(f"\n--- [RL] Starting RL Epoch {epoch+1} (Mixed Training) ---")
        
        for step, batch in enumerate(self.train_loader):
            # 1. 数据搬运
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # ==========================================
            # Part A: Reinforcement Learning (SCST)
            # ==========================================
            
            # 1. Sampling (探索): 获取带有梯度的 log_probs
            sample_boxes, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                sample=True
            )
            
            # 2. Baseline (贪婪): 作为 Reward 的基准线
            self.model.eval()
            with torch.no_grad():
                baseline_boxes, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                    sample=False
                )
            self.model.train()
            
            # 3. 计算 Reward 和 Advantage
            reward_sample = self.compute_reward(sample_boxes, batch)
            reward_baseline = self.compute_reward(baseline_boxes, batch)
            
            advantage = reward_sample - reward_baseline # [B]
            
            # 4. 计算 Policy Gradient Loss
            # Loss = - (log_p * advantage)
            # log_probs 是整个序列的概率和 (针对每个样本 Sum over T)
            log_prob_sum = log_probs.sum(dim=1) 
            rl_loss = -(log_prob_sum * advantage).mean()
            
            # ==========================================
            # Part B: Entropy Regularization (解决散不开的关键)
            # ==========================================
            # 熵 H(p) ≈ -log_p
            # 我们希望 Maximize Entropy => Minimize (log_p)
            # 实际上 log_p 是负数，我们希望它越小（绝对值越大），表示分布越平坦
            # 添加 Entropy Bonus: Loss - w * Entropy
            # 近似计算: entropy_bonus = -log_probs.mean()
            entropy_val = -log_probs.mean()
            
            # ==========================================
            # Part C: Supervised Anchor & Free Bits (防崩坏)
            # ==========================================
            # 使用标准的 Forward 获取 CVAE 的 mu, logvar 和监督预测
            mu, logvar, pred_boxes_sup, _ = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                target_boxes=batch['target_boxes']
            )
            
            # 计算监督 Loss
            loss_tuple = self.model.get_loss(
                pred_cls=None, pred_bbox_ids=None, 
                pred_boxes=pred_boxes_sup,
                pred_count=None, layout_seq=None, 
                layout_mask=batch['loss_mask'], 
                num_boxes=batch['num_boxes'].to(self.device), 
                target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch['kg_spatial_matrix'],
                kg_class_weights=batch['kg_class_weights'],
                kg_class_ids=batch['kg_class_ids']
            )
            supervised_loss = loss_tuple[0]
            
            # 计算 KL Divergence (Free Bits)
            if mu is not None and logvar is not None:
                raw_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                # 只有当 KL 大于阈值时才优化，保留 2.0 的信息量空间
                min_kl = torch.tensor(self.kl_threshold, device=self.device)
                kl_loss = torch.max(raw_kl, min_kl)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            # ==========================================
            # Part D: 总损失融合与反向传播
            # ==========================================
            # Total Loss = RL + Anchor + KL - Entropy_Reward
            total_combined_loss = 1.0 * rl_loss + \
                                  self.supervised_weight * supervised_loss + \
                                  0.1 * kl_loss - \
                                  self.entropy_weight * entropy_val
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL] Step {step+1} | R_Avg:{reward_sample.mean():.3f} | Adv:{advantage.mean():.3f} | "
                      f"Ent:{entropy_val.item():.2f} | " # 观察这个值，应该保持正数且不趋于0
                      f"Loss:{total_combined_loss.item():.2f} || "
                      f"IoU:{stats.get('IoU', 0):.2f} Rel:{stats.get('Rel', 0):.2f} Disp:{stats.get('Disp', 0):.2f}")

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"--- [RL] Epoch {epoch+1} Finished. Avg Epoch Reward: {avg_reward:.4f} ---")
        
        # 绘图
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        
        # [修改] 每 50 轮保存一次 Checkpoint
        if (epoch + 1) % self.save_every == 0:
            checkpoint_path = os.path.join(self.output_dir, f"rl_model_epoch_{epoch+1}.pth")
            torch.save(
                {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'epoch': epoch + 1,
                 'avg_reward': avg_reward,
                 'config': self.config}, # 保存 Config 方便复现
                checkpoint_path
            )
            print(f"-> [RL] Checkpoint saved: {checkpoint_path}")
        
        return avg_reward