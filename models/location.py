import torch
import torch.nn.functional as F
import random

class LocationSignalGenerator:
    """
    [V3.1] 状态感知型位置生成器 (State-Aware) - 支持 8x8 Grid 与多样性采样。
    核心能力：
    1. 维护全局占用图 (Occupancy Map)。
    2. 基于重叠代价 (Overlap Cost) 动态选择最佳空位。
    3. 支持策略回退 (Fallback)，如果首选位置满了，自动寻找次优解。
    4. [NEW] 支持多样性采样 (Diversity Sampling)，打破确定性僵局。
    """
    def __init__(self, grid_size=8):  # [MODIFIED] 默认 grid_size 改为 8
        self.H = grid_size
        self.W = grid_size
        
        # 1. 初始化精细化模式库 (适配 8x8 的精细度)
        self.patterns = self._init_fine_grained_patterns()
        
        # 2. 定义关系的“候选策略池”
        # 当检测到某种关系时，我们可以尝试以下位置（按优先级排序）
        self.STRATEGY_POOL = {
            'above': ['sky_left', 'sky_right', 'sky_center'],  # 上方：优先左右角，最后正中
            'below': ['ground_left', 'ground_right', 'ground_full'],
            'on_top': ['ground_full', 'ground_left', 'ground_right'], # on_top 倾向于铺开
            'inside': ['core', 'center_focus'],
            'surrounds': ['frame'],
            'near': ['wing_left', 'wing_right']
        }
        
        # 关系 ID 映射
        self.REL_IDS = {
            'none': 0, 'above': 1, 'below': 2, 'inside': 3,
            'surrounds': 4, 'on_top': 5, 'near': 6
        }

    def _zeros(self):
        return torch.zeros((self.H, self.W), dtype=torch.float32)

    def _init_fine_grained_patterns(self):
        """初始化模式库 (适配 8x8 Grid 的设计)"""
        pats = {}
        zeros = self._zeros
        
        # --- [MODIFIED] 适配 8x8 的切片逻辑 ---
        
        # Sky (上方区域)
        # sky_left: 左上角 3x3
        p = zeros(); p[0:3, 0:3] = 1.0; pats['sky_left'] = p
        # sky_right: 右上角 3x3
        p = zeros(); p[0:3, -3:] = 1.0; pats['sky_right'] = p
        # sky_center: 上方中间条带 (前2行, 中间留空两边)
        p = zeros(); p[0:2, 2:-2] = 1.0; pats['sky_center'] = p
        
        # Ground (下方区域)
        # ground_left: 左下角 3x3
        p = zeros(); p[-3:, 0:3] = 1.0; pats['ground_left'] = p
        # ground_right: 右下角 3x3
        p = zeros(); p[-3:, -3:] = 1.0; pats['ground_right'] = p
        # ground_full: 底部 2 行 + 倒数第 3 行的部分缓冲
        p = zeros(); p[-2:, :] = 1.0; p[-3, 1:-1] = 0.8; pats['ground_full'] = p
        
        # Core & Frame
        # core: 中间 4x4 区域 (2:6)
        p = zeros(); p[2:6, 2:6] = 1.0; p[3:5, 3:5] = 1.5; pats['core'] = p
        # frame: 四周 1 像素边框
        p = torch.ones((self.H, self.W)) * 0.2
        p[0,:]=1; p[-1,:]=1; p[:,0]=1; p[:,-1]=1; pats['frame'] = p
        
        # Wings & Center
        # wing_left: 左侧 2 列 (避开极顶和极底)
        p = zeros(); p[1:-1, 0:2] = 1.0; pats['wing_left'] = p
        # wing_right: 右侧 2 列
        p = zeros(); p[1:-1, -2:] = 1.0; pats['wing_right'] = p
        # center_focus: 核心聚焦 (比 core 更集中)
        p = zeros(); p[2:6, 2:6] = 0.5; p[3:5, 3:5] = 1.0; pats['center_focus'] = p
        
        return pats

    def infer_stateful_signal(self, my_idx, spatial_row, spatial_col, current_occupancy, mode='greedy', top_k=3):
        """
        [AI 核心逻辑 V3.1 - Diversity]
        Args:
            my_idx: 当前物体索引 (未使用，保留接口)
            spatial_row: 该物体对他人的关系 (Active)
            spatial_col: 他人对该物体的关系 (Passive)
            current_occupancy: 当前画布的占用热力图 [8, 8]
            mode: 'greedy' (默认, 确定性) 或 'sample' (随机性, 用于生成多样化布局)
            top_k: 采样时考虑前 K 个最佳位置
        Returns:
            best_signal: 选定的最佳位置信号 [8, 8]
            new_occupancy: 更新后的占用图 [8, 8] (用于传给下一个物体)
        """
        # 1. 确定意图：根据 KG 关系决定我想去哪 (Candidates)
        candidates = [] # List of pattern_names
        
        # (A) 主动关系检查 (Row)
        for rel_id in spatial_row:
            rel = int(rel_id)
            if rel == 0: continue
            
            # 查找策略池
            rel_name = next((k for k, v in self.REL_IDS.items() if v == rel), None)
            if rel_name in self.STRATEGY_POOL:
                candidates.extend(self.STRATEGY_POOL[rel_name])

        # (B) 被动关系检查 (Col)
        for rel_id in spatial_col:
            rel = int(rel_id)
            if rel == 0: continue
            
            # 逻辑反转
            if rel == self.REL_IDS['above']: candidates.extend(self.STRATEGY_POOL['below'])
            elif rel == self.REL_IDS['below']: candidates.extend(self.STRATEGY_POOL['above'])
            elif rel == self.REL_IDS['inside']: candidates.extend(self.STRATEGY_POOL['surrounds']) # 别人在我里面->我是Frame

        # (C) 兜底策略：如果没有特殊关系，尝试去任何空闲的通用区域
        if not candidates:
            candidates = ['center_focus', 'wing_left', 'wing_right']

        # 2. 计算代价 (Cost Calculation)
        # 去重并保持顺序 (优先策略在前)
        unique_candidates = []
        [unique_candidates.append(x) for x in candidates if x not in unique_candidates]
        
        candidate_scores = [] # List of (cost, pattern)
        
        for pat_name in unique_candidates:
            pattern = self.patterns[pat_name]
            
            # 计算代价: Overlap + Penalty
            # Overlap: 当前 pattern 与 历史占用 的重叠程度
            overlap = (pattern * current_occupancy).sum()
            cost = overlap.item() # Convert Tensor to float
            
            candidate_scores.append((cost, pattern))
            
        # 3. 选择策略 (Selection Strategy)
        best_signal = None
        
        # 按代价从小到大排序
        candidate_scores.sort(key=lambda x: x[0])
        
        if not candidate_scores:
            # 极端情况回退
            best_signal = self.patterns['center_focus']
        else:
            if mode == 'greedy':
                # 原始逻辑：只选 Cost 最小的
                # 如果有多个 cost 相同的最小值，greedy 默认选第一个（即策略池中优先级最高的）
                best_signal = candidate_scores[0][1]
                
            elif mode == 'sample':
                # [NEW] 多样性逻辑：从 Top-K 中随机选一个
                # 如果前几个 Cost 都很小（比如都是 0），那么随机选一个能带来布局变化
                k = min(len(candidate_scores), top_k)
                opts = candidate_scores[:k]
                
                # 权重计算：Cost 越小权重越大 (Inverse Cost)
                # 为防止除零，加一个 epsilon。也防止 cost 极大时权重过小。
                weights = [1.0 / (c + 0.1) for c, p in opts]
                
                # 随机选择
                try:
                    # random.choices 返回列表，取第一个
                    chosen_pair = random.choices(opts, weights=weights, k=1)[0]
                    best_signal = chosen_pair[1]
                except Exception:
                    #由于浮点数精度等问题兜底
                    best_signal = opts[0][1]

        # 4. 更新状态
        # 使用 max 而不是 add，防止热力值无限叠加爆炸，保持在 0-1 范围更有意义
        # 或者用 add 但限制上限。这里用 add 模拟“堆积感”也是可以的，
        # 但为了让后来的物体知道这里很挤，add 更好。
        new_occupancy = current_occupancy + best_signal
        
        # 归一化输出信号 (仅归一化当前的 best_signal，不归一化 occupancy)
        if best_signal.max() > 0:
            best_signal = best_signal / best_signal.max()
            
        return best_signal, new_occupancy

# 单元测试
if __name__ == "__main__":
    # [MODIFIED] 测试 8x8 Grid
    gen = LocationSignalGenerator(grid_size=8)
    
    # 初始化一张白纸 (8x8)
    occupancy = torch.zeros((8, 8))
    
    print("--- 场景模拟 (8x8): 3只鸟都在天上 (Above) ---")
    
    # 模拟 KG: Row 都有 1 (above)
    
    # 1. 第一只鸟 (优先占领左上 3x3)
    print("\nBird 1 (Idx 0) - Greedy:")
    sig1, occupancy = gen.infer_stateful_signal(0, [0, 1], [0, 0], occupancy, mode='greedy')
    print(sig1) 
    
    # 2. 第二只鸟 (发现左上满了，自动去右上 3x3)
    print("\nBird 2 (Idx 1) - Greedy:")
    sig2, occupancy = gen.infer_stateful_signal(1, [0, 1], [0, 0], occupancy, mode='greedy')
    print(sig2) 
    
    # 3. 第三只鸟 (测试 Sample 模式，可能去正中，也可能强行挤在其他空隙)
    print("\nBird 3 (Idx 2) - Sampling (Diversity Check):")
    # 这里我们不更新 occupancy，模拟多次采样看结果
    for i in range(3):
        sig3, _ = gen.infer_stateful_signal(2, [0, 1], [0, 0], occupancy, mode='sample', top_k=3)
        print(f"Sample {i+1} Max Loc: {torch.nonzero(sig3 == sig3.max())[0].tolist()}")
    
    print("\nFinal Occupancy Map (8x8):")
    print(occupancy)