# File: tianyusun1/test2/test2-5.2/trainers/rl_trainer.py (V5.8: SCST & Detailed Logging)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time

class RLTrainer(LayoutTrainer):
    """
    强化学习训练器 (RL Fine-tuning Trainer)。
    继承自 LayoutTrainer，复用了验证、保存模型等逻辑，
    但重写了训练循环以支持 SCST (Self-Critical Sequence Training)。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # === 1. RL 超参数设置 ===
        # RL 通常需要极小的学习率，防止破坏预训练的权重
        self.rl_lr = 5e-6 
        
        # 重新定义优化器 (针对 RL 微调)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # === 2. 奖励权重 (Reward Weights) ===
        # 根据痛点调整这些权重
        self.w_iou = 2.0        # [监督] 有 GT 的物体：IoU 越高越好
        self.w_rel = 1.0        # [逻辑] 无 GT 的物体：满足 KG 关系即奖励
        self.w_dispersion = 0.5 # [构图] 无 GT 的物体：鼓励分散，防止堆积
        self.w_overlap = -0.5   # [惩罚] 任何物体：重叠惩罚 (负奖励)

        # [NEW] 用于记录最近一次 batch 的奖励明细
        self.last_reward_stats = {}

        print(f"[RLTrainer] Initialized. LR={self.rl_lr}")
        print(f"[RLTrainer] Reward Weights: IoU={self.w_iou}, Rel={self.w_rel}, Disp={self.w_dispersion}, Overlap={self.w_overlap}")

    def compute_reward(self, pred_boxes, batch):
        """
        计算每个样本的奖励值 (Batch-wise Reward Calculation)。
        Args:
            pred_boxes: [B, T, 4] (cx, cy, w, h)
            batch: 包含 target_boxes, loss_mask, kg_spatial_matrix 等
        Returns:
            rewards: [B] 每个样本的总奖励
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
        
        # === A. 监督奖励 (Supervised Reward) - 针对有 GT 的部分 ===
        # 计算 IoU
        iou = self._calculate_iou(pred_boxes, target_boxes) # [B, T]
        # 只有 loss_mask=1 的地方才给 IoU 奖励
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # === B. 关系奖励 (Relation Reward) - 解决“标注缺失” ===
        # 针对 loss_mask=0 (无 GT) 的部分，我们用 KG 关系来指导它
        # 只要位置关系对，就给分
        no_gt_mask = 1.0 - loss_mask
        
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) # [B, T]
        r_rel = rel_scores * no_gt_mask * self.w_rel
        obj_rewards += r_rel

        # === C. 分散度奖励 (Dispersion Reward) - 解决“中心堆积” ===
        # 计算每个样本中，所有无 GT 物体的中心点标准差
        # 鼓励它们铺开，而不是缩在中间
        centers = pred_boxes[..., :2] # [B, T, 2]
        
        # 为了计算方便，我们计算整个样本的 std，然后加给每个无 GT 的物体
        # std_x: [B]
        std_x = centers[..., 0].std(dim=1)
        std_y = centers[..., 1].std(dim=1)
        # 限制最大奖励，防止为了分散而分散
        disp_score = torch.clamp(std_x + std_y, max=0.6).unsqueeze(1) # [B, 1]
        
        r_disp = disp_score * no_gt_mask * self.w_dispersion
        obj_rewards += r_disp

        # === D. 重叠惩罚 (Overlap Penalty) - 全局约束 ===
        # 简单的两两 IoU 惩罚
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes) # [B, T]
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over

        # [NEW] 记录明细 (取 batch 平均值以便打印)
        # 注意：分母加 1e-6 防止除以 0
        self.last_reward_stats = {
            'IoU': (r_iou.sum() / (loss_mask.sum() + 1e-6)).item(), # 仅统计有 GT 的部分
            'Rel': (r_rel.sum() / (no_gt_mask.sum() + 1e-6)).item(), # 仅统计无 GT 的部分
            'Disp': disp_score.mean().item(),
            'Over': overlap_penalty.mean().item()
        }

        # === 汇总 ===
        # 对每个样本内的所有物体奖励取均值，作为该样本最终的 Reward
        # [B]
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
        """
        辅助函数：检查 KG 关系一致性 [B, T]
        逻辑：遍历所有物体对，如果满足 matrix 中的关系，给双方加分。
        """
        B, T, _ = boxes.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        
        if matrix is None: return rewards

        # 简化版实现，利用 Python 循环 (虽然慢但逻辑清晰，且 batch size 不大)
        # 也可以用矩阵操作加速
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
                    
                    # 查找 i 和 j 之间的关系
                    rel = matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue
                    
                    # 检查是否满足关系
                    box_i = boxes[b, i] # [cx, cy, w, h]
                    box_j = boxes[b, j]
                    
                    satisfied = False
                    # 1: above (i 在 j 上方 -> cy_i < cy_j)
                    if rel == 1 and box_i[1] < box_j[1]: satisfied = True
                    # 2: below (i 在 j 下方 -> cy_i > cy_j)
                    elif rel == 2 and box_i[1] > box_j[1]: satisfied = True
                    # 3: inside (i 在 j 内部)
                    elif rel == 3:
                        # 简单判定：中心点距离小于 j 的宽高的一半
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_j[2]/2 and dy < box_j[3]/2: satisfied = True
                    # 4: surrounds (i 包围 j -> 即 j inside i)
                    elif rel == 4:
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_i[2]/2 and dy < box_i[3]/2: satisfied = True
                    
                    if satisfied:
                        rewards[b, i] += 1.0 # 满足一个关系加 1 分
                        
        # 归一化：避免某个物体关系太多分数过高
        return torch.clamp(rewards, max=3.0)

    def _calculate_overlap_penalty(self, boxes):
        """计算两两重叠惩罚"""
        B, T, _ = boxes.shape
        penalty = torch.zeros(B, T, device=boxes.device)
        
        # 计算所有物体对的 IoU
        # 这里为了速度简化，只惩罚中心点距离过近的
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers) # [B, T, T]
        
        # 阈值：如果两个物体中心点距离小于 0.1，视为重叠/堆积
        too_close = (dist < 0.1).float()
        
        # 去掉对角线 (自己和自己距离为0)
        mask = torch.eye(T, device=boxes.device).unsqueeze(0).expand(B, -1, -1)
        too_close = too_close * (1 - mask)
        
        # 每个物体受到的惩罚 = 它离多少个其他物体太近
        penalty = too_close.sum(dim=2) 
        
        # 归一化
        return penalty

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
                # 注意：这里调用的是新的 forward_rl 接口
                # sample=False -> Greedy
                baseline_boxes, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                    sample=False
                )
                # 计算 Baseline 奖励 (b)
                reward_baseline = self.compute_reward(baseline_boxes, batch)
            
            # ==========================================
            # 步骤 2: Sampling (Exploration)
            # ==========================================
            self.model.train()
            # sample=True -> 从分布中采样，并返回 log_prob
            sample_boxes, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                sample=True
            )
            # 计算 Sample 奖励 (r)
            reward_sample = self.compute_reward(sample_boxes, batch)
            
            # ==========================================
            # 步骤 3: 优势函数与梯度更新
            # ==========================================
            # Advantage = r - b
            # 如果 Sample 结果比 Baseline 好，Advantage > 0，增加该动作概率
            # 如果 Sample 结果比 Baseline 差，Advantage < 0，降低该动作概率
            advantage = reward_sample - reward_baseline
            
            # SCST Loss = - (r - b) * log p(sample)
            # log_probs 是 [B, T]，我们需要对序列求和得到整个轨迹的概率
            # 或者取平均，这里取平均更稳定
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
                      f"Details -> IoU: {stats.get('IoU', 0):.2f}, "
                      f"Rel: {stats.get('Rel', 0):.2f}, "
                      f"Disp: {stats.get('Disp', 0):.2f}, "
                      f"Over: {stats.get('Over', 0):.2f}")

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"--- [RL] Epoch {epoch+1} Finished. Avg Epoch Reward: {avg_reward:.4f} ---")