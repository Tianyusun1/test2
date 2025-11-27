import torch
import torch.nn.functional as F

class LocationSignalGenerator:
    """
    [V3.0] 状态感知型位置生成器 (State-Aware)。
    核心能力：
    1. 维护全局占用图 (Occupancy Map)。
    2. 基于重叠代价 (Overlap Cost) 动态选择最佳空位。
    3. 支持策略回退 (Fallback)，如果首选位置满了，自动寻找次优解。
    """
    def __init__(self, grid_size=5):
        self.H = grid_size
        self.W = grid_size
        
        # 1. 初始化精细化模式库 (保持 V2.0 的精细度)
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
        """初始化模式库 (复用 V2.0 的设计)"""
        pats = {}
        zeros = self._zeros
        
        # Sky
        p = zeros(); p[0:2, 0:2] = 1.0; pats['sky_left'] = p
        p = zeros(); p[0:2, -2:] = 1.0; pats['sky_right'] = p
        p = zeros(); p[0, 1:-1] = 1.0; pats['sky_center'] = p
        
        # Ground
        p = zeros(); p[-2:, 0:2] = 1.0; pats['ground_left'] = p
        p = zeros(); p[-2:, -2:] = 1.0; pats['ground_right'] = p
        p = zeros(); p[-1, :] = 1.0; p[-2, 1:-1] = 0.8; pats['ground_full'] = p
        
        # Core & Frame
        p = zeros(); p[1:4, 1:4] = 1.0; p[2, 2] = 1.5; pats['core'] = p
        p = torch.ones((self.H, self.W)) * 0.2
        p[0,:]=1; p[-1,:]=1; p[:,0]=1; p[:,-1]=1; pats['frame'] = p
        
        # Wings & Center
        p = zeros(); p[1:-1, 0] = 1.0; pats['wing_left'] = p
        p = zeros(); p[1:-1, -1] = 1.0; pats['wing_right'] = p
        p = zeros(); p[1:4, 1:4] = 0.5; p[2, 2] = 1.0; pats['center_focus'] = p
        
        return pats

    def infer_stateful_signal(self, my_idx, spatial_row, spatial_col, current_occupancy):
        """
        [AI 核心逻辑 V3.0]
        Args:
            current_occupancy: 当前画布的占用热力图 [5, 5]
        Returns:
            best_signal: 选定的最佳位置信号 [5, 5]
            new_occupancy: 更新后的占用图 [5, 5] (用于传给下一个物体)
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

        # 2. 贪心选择：计算重叠代价 (Cost Function)
        best_signal = None
        min_cost = float('inf')
        
        # 去重并保持顺序 (优先策略在前)
        unique_candidates = []
        [unique_candidates.append(x) for x in candidates if x not in unique_candidates]
        
        for pat_name in unique_candidates:
            pattern = self.patterns[pat_name]
            
            # 计算代价: Overlap + Penalty
            # Overlap: 当前 pattern 与 历史占用 的重叠程度
            overlap = (pattern * current_occupancy).sum()
            
            # Penalty: 稍微惩罚靠后的候选者，倾向于首选策略
            # 但如果首选策略重叠太严重，代价会很高，从而选择次优
            cost = overlap
            
            if cost < min_cost:
                min_cost = cost
                best_signal = pattern
                
                # 如果找到了完美的空位 (Cost=0)，直接根据贪心原则提前结束，节省计算
                if cost < 0.1: 
                    break
        
        # 3. 如果所有地方都满了 (极端情况)，回退到默认
        if best_signal is None:
            best_signal = self.patterns['center_focus']

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
    gen = LocationSignalGenerator()
    
    # 初始化一张白纸
    occupancy = torch.zeros((5, 5))
    
    print("--- 场景模拟: 3只鸟都在天上 (Above) ---")
    
    # 模拟 KG: Row 都有 1 (above)
    
    # 1. 第一只鸟 (优先占领左上)
    print("\nBird 1 (Idx 0):")
    sig1, occupancy = gen.infer_stateful_signal(0, [0, 1], [0, 0], occupancy)
    print(sig1) # 应该高亮左上
    
    # 2. 第二只鸟 (发现左上满了，自动去右上)
    print("\nBird 2 (Idx 1):")
    sig2, occupancy = gen.infer_stateful_signal(1, [0, 1], [0, 0], occupancy)
    print(sig2) # 应该高亮右上
    
    # 3. 第三只鸟 (左右都满了，被迫去正中)
    print("\nBird 3 (Idx 2):")
    sig3, occupancy = gen.infer_stateful_signal(2, [0, 1], [0, 0], occupancy)
    print(sig3) # 应该高亮正上
    
    print("\nFinal Occupancy Map:")
    print(occupancy)