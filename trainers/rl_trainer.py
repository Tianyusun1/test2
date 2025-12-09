# File: tianyusun1/test2/test2-5.2/trainers/rl_trainer.py (V5.13: Fixed Broadcasting Bug)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt

class RLTrainer(LayoutTrainer):
    """
    强化学习训练器 (RL Fine-tuning Trainer)。
    继承自 LayoutTrainer，复用了验证、保存模型等逻辑，
    但重写了训练循环以支持 SCST (Self-Critical Sequence Training)。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # === 1. RL 超参数设置 ===
        # 尝试从 config 中读取，如果没有则使用默认值
        self.rl_lr = float(config['training'].get('rl_learning_rate', 5e-6))
        
        # 重新定义优化器 (针对 RL 微调)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # === 2. 奖励权重 (Reward Weights) ===
        reward_cfg = config['training'].get('reward_weights', {})
        
        self.w_iou = float(reward_cfg.get('iou', 2.0))              
        self.w_rel = float(reward_cfg.get('relation', 1.0))         
        self.w_dispersion = float(reward_cfg.get('dispersion', 2.5)) 
        self.w_overlap = float(reward_cfg.get('overlap', -0.5))     

        # [NEW] 用于记录最近一次 batch 的奖励明细
        self.last_reward_stats = {}

        # [新增] 奖励历史记录，用于画图
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer] Initialized. LR={self.rl_lr:.2e}")
        print(f"[RLTrainer] Reward Weights: IoU={self.w_iou}, Rel={self.w_rel}, Disp={self.w_dispersion}, Overlap={self.w_overlap}")
        print(f"[RLTrainer] Reward plot will be saved to: {self.plot_path_reward}")

    def compute_reward(self, pred_boxes, batch):
        """
        计算每个样本的奖励值 (Batch-wise Reward Calculation)。
        """
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # 提取 Batch 信息
        loss_mask = batch['loss_mask']          # 1.0 = 有效GT, 0.0 = 无GT/脏数据
        target_boxes = batch['target_boxes']
        kg_spatial_matrix = batch['kg_spatial_matrix'] # [B, 9, 9]
        kg_class_ids = batch['kg_class_ids']    # [B, T]
        
        # 初始化奖励矩阵 [B, T] -> 每个物体单独计算，最后求和
        obj_rewards = torch.zeros(B, T, device=device)
        
        # ===========================================================
        # A. 监督奖励 (Supervised Reward) - 针对有 GT 的部分
        # ===========================================================
        iou = self._calculate_iou(pred_boxes, target_boxes) # [B, T]
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # ===========================================================
        # B. 关系奖励 (Relation Reward) - 解决“标注缺失”
        # ===========================================================
        no_gt_mask = 1.0 - loss_mask
        
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) # [B, T]
        r_rel = rel_scores * no_gt_mask * self.w_rel
        obj_rewards += r_rel

        # ===========================================================
        # C. 分散度奖励 (Dispersion Reward) - 解决“中心堆积”
        # ===========================================================
        centers = pred_boxes[..., :2] # [B, T, 2]
        
        # std_x: [B]
        std_x = centers[..., 0].std(dim=1)
        std_y = centers[..., 1].std(dim=1)
        
        # [策略] 设定上限 (0.8)，防止模型为了刷分把物体无限推向边缘
        # 这里需要 unsqueeze(1) 是因为 std 降维了，变成了 [B]，要变回 [B, 1] 才能和 [B, T] 相乘
        disp_score = torch.clamp(std_x + std_y, max=0.8).unsqueeze(1) # [B, 1]
        
        r_disp = disp_score * no_gt_mask * self.w_dispersion
        obj_rewards += r_disp

        # ===========================================================
        # D. 构图美学奖励：三分法 (Rule of Thirds)
        # ===========================================================
        # 鼓励物体中心接近 0.33 或 0.66
        cx = centers[..., 0]
        cy = centers[..., 1]
        
        # 计算离最近的 "三分线" 的距离 [B, T]
        dist_x_third = torch.min(torch.abs(cx - 0.333), torch.abs(cx - 0.667))
        dist_y_third = torch.min(torch.abs(cy - 0.333), torch.abs(cy - 0.667))
        
        # 距离越小，奖励越高。距离大于 0.2 时奖励归零。
        # r_composition 本身就是 [B, T]
        r_composition = (0.2 - (dist_x_third + dist_y_third)).clamp(min=0) * 2.5
        
        # [BUG FIX] 移除 .unsqueeze(1)，因为 r_composition 已经是 [B, T]
        obj_rewards += r_composition * no_gt_mask

        # ===========================================================
        # E. 边缘惩罚 (Border Penalty)
        # ===========================================================
        # 严厉惩罚贴边行为 (margin = 0.05)
        dist_to_edge = torch.min(centers, 1.0 - centers) # [B, T, 2]
        is_too_close_to_edge = (dist_to_edge < 0.05).float().sum(dim=-1) # [B, T]
        
        # r_border_penalty 本身就是 [B, T]
        r_border_penalty = is_too_close_to_edge * -1.0 # 每次违规扣 1 分
        
        # [BUG FIX] 移除 .unsqueeze(1)
        obj_rewards += r_border_penalty

        # ===========================================================
        # F. 重叠惩罚 (Overlap Penalty)
        # ===========================================================
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes) # [B, T]
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over

        # [NEW] 记录明细
        self.last_reward_stats = {
            'IoU': (r_iou.sum() / (loss_mask.sum() + 1e-6)).item(),
            'Rel': (r_rel.sum() / (no_gt_mask.sum() + 1e-6)).item(),
            'Disp': disp_score.mean().item(),
            'Comp': r_composition.mean().item(), # Composition Score
            'Over': overlap_penalty.mean().item()
        }

        # === 汇总 ===
        total_sample_reward = obj_rewards.sum(dim=1) / (T + 1e-6)
        
        return total_sample_reward

    def _calculate_iou(self, pred, target):
        """辅助函数：计算 IoU [B, T]"""
        p_x1, p_y1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        p_x2, p_y2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        t_x1, t_y1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        t_x2, t_y2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        
        i_x1 = torch.max(p_x1, t_x1)
        i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2)
        i_y2 = torch.min(p_y2, t_y2)
        
        i_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
        p_area = pred[..., 2] * pred[..., 3]
        t_area = target[..., 2] * target[..., 3]
        u_area = p_area + t_area - i_area
        return i_area / (u_area + 1e-6)

    def _calculate_relation_reward(self, boxes, matrix, class_ids):
        """辅助函数：检查 KG 关系一致性 [B, T]"""
        B, T, _ = boxes.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        
        if matrix is None: return rewards

        for b in range(B):
            for i in range(T):
                cid_i = class_ids[b, i].item()
                idx_i = int(cid_i) - 2
                if not (0 <= idx_i < 9): continue
                
                for j in range(T):
                    if i == j: continue
                    cid_j = class_ids[b, j].item()
                    idx_j = int(cid_j) - 2
                    if not (0 <= idx_j < 9): continue
                    
                    rel = matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue
                    
                    box_i = boxes[b, i]
                    box_j = boxes[b, j]
                    
                    satisfied = False
                    if rel == 1 and box_i[1] < box_j[1]: satisfied = True # above
                    elif rel == 2 and box_i[1] > box_j[1]: satisfied = True # below
                    elif rel == 3: # inside
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_j[2]/2 and dy < box_j[3]/2: satisfied = True
                    elif rel == 4: # surrounds
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_i[2]/2 and dy < box_i[3]/2: satisfied = True
                    
                    if satisfied:
                        rewards[b, i] += 1.0 
                        
        return torch.clamp(rewards, max=3.0)

    def _calculate_overlap_penalty(self, boxes):
        """计算两两重叠惩罚"""
        B, T, _ = boxes.shape
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers)
        
        too_close = (dist < 0.1).float()
        mask = torch.eye(T, device=boxes.device).unsqueeze(0).expand(B, -1, -1)
        too_close = too_close * (1 - mask)
        
        penalty = too_close.sum(dim=2) 
        return penalty

    def _plot_reward_history(self):
        """[新增] 绘制并保存奖励曲线"""
        if not self.reward_history:
            return
            
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
        """执行一个 Epoch 的 SCST 训练"""
        self.model.train()
        total_reward = 0
        total_loss = 0
        steps = 0
        
        print(f"\n--- [RL] Starting RL Epoch {epoch+1} (SCST) ---")
        
        for step, batch in enumerate(self.train_loader):
            # 1. 数据移至 GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # ==========================================
            # 步骤 1: Baseline (Greedy Search)
            # ==========================================
            self.model.eval()
            with torch.no_grad():
                baseline_boxes, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                    sample=False
                )
                reward_baseline = self.compute_reward(baseline_boxes, batch)
            
            # ==========================================
            # 步骤 2: Sampling (Exploration)
            # ==========================================
            self.model.train()
            sample_boxes, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                sample=True
            )
            reward_sample = self.compute_reward(sample_boxes, batch)
            
            # ==========================================
            # 步骤 3: 优势函数与梯度更新
            # ==========================================
            advantage = reward_sample - reward_baseline
            
            log_prob_sum = log_probs.sum(dim=1)
            loss = -(log_prob_sum * advantage).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            total_loss += loss.item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL] Epoch {epoch+1} Step {step+1} | "
                      f"R_Avg: {reward_sample.mean().item():.3f} | "
                      f"Adv: {advantage.mean().item():.3f} || "
                      f"IoU:{stats.get('IoU', 0):.2f} "
                      f"Rel:{stats.get('Rel', 0):.2f} "
                      f"Disp:{stats.get('Disp', 0):.2f} "
                      f"Comp:{stats.get('Comp', 0):.2f} "
                      f"Over:{stats.get('Over', 0):.2f}")

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"--- [RL] Epoch {epoch+1} Finished. Avg Epoch Reward: {avg_reward:.4f} ---")
        
        # 记录历史并绘图，然后返回平均奖励
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        
        return avg_reward