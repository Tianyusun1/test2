# File: tianyusun1/test2/test2-5.1/scripts/train.py (V5.4: CLUSTERING LOSS INTEGRATION)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (train.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (train.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
import trainers 
from collections import Counter
import numpy as np 

# --- 辅助函数：计算类别权重 (解决类别偏差问题) ---
def compute_class_weights(dataset, num_classes: int, max_weight_ratio: float = 3.0):
    """
    计算数据集内所有布局元素的类别频率，并返回反向频率权重。
    返回 (num_classes + 1) 个权重，其中索引 0 对应 EOS。
    """
    element_class_counts = Counter()
    
    # 遍历整个数据集计算所有元素实例的类别计数
    for sample in dataset.data:
        # boxes 是 List[(cls_id, cx, cy, w, h)]，cls_id 是 2.0 - 10.0
        for cls_id_float, _, _, _, _ in sample['boxes']:
            # 映射到内部 ID 0-8 (对应元素 2-10)
            internal_element_id = int(cls_id_float) - 2
            if 0 <= internal_element_id < num_classes:
                element_class_counts[internal_element_id] += 1
                
    total_count = sum(element_class_counts.values())
    
    # 最终权重数组大小必须是 9 (元素) + 1 (EOS) = 10
    final_num_classes = num_classes + 1 
    
    if total_count == 0:
        print("[Warning] No valid elements found in dataset for weight calculation. Using uniform weights.")
        return torch.ones(final_num_classes)

    # 初始化权重: size 10 (索引 0 for EOS, 索引 1-9 for elements)
    weights = torch.zeros(final_num_classes) 
    
    # 计算 9 个元素 (内部 ID 0-8) 的权重，并将它们存储在索引 1-9
    for i in range(num_classes): # i goes from 0 to 8 (internal element ID)
        frequency = element_class_counts.get(i, 0) / total_count
        
        # 将元素的内部 ID i (0-8) 映射到新的索引 i+1 (1-9)
        new_index = i + 1 
        
        if frequency > 0:
            # 采用 log(1.0 + x) 或 log(1.02 + x) 来平滑和反转频率
            weights[new_index] = 1.0 / np.log(1.02 + frequency) 
        else:
            # 对于稀有/不存在的类别，赋予最高的权重
            weights[new_index] = 1.0 / np.log(1.02 + 1e-6) # 赋予最大权重
            
    # 为 EOS 类 (索引 0) 分配权重
    # 我们使用计算出的元素权重的平均值作为基线
    avg_element_weight = weights[1:].sum() / num_classes if num_classes > 0 else 1.0
    weights[0] = avg_element_weight # 将平均权重分配给 EOS (索引 0)
            
    # 权重钳位和标准化
    # 对所有 10 个类别计算平均权重
    avg_weight = weights.mean()
    max_allowed_weight = avg_weight * max_weight_ratio
    weights = torch.clamp(weights, max=max_allowed_weight)
    
    # 重新归一化，确保总和是 final_num_classes (10)
    weights = weights / weights.sum() * final_num_classes
    
    return weights.float()
# ------------------------------------------


def main():
    # 1. Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Init tokenizer and load FULL dataset for weight calculation
    model_config = config['model']
    
    # 重新初始化数据集以确保数据完整性
    dataset = PoegraphLayoutDataset(
        xlsx_path=model_config['xlsx_path'],
        labels_dir=model_config['labels_dir'],
        bert_model_path=model_config['bert_path'],
        max_layout_length=model_config['max_layout_length'],
        max_text_length=model_config['max_text_length']
    )
    
    # --- 计算类别权重 ---
    num_element_classes = model_config['num_classes'] # 9
    class_weights_tensor = compute_class_weights(dataset, num_element_classes)
    
    print(f"Calculated Class Weights (Internal 0:EOS, 1-9:Elements 2-10): {class_weights_tensor.tolist()}")
    # ---------------------------

    # 3. Init model (传入所有损失权重，包括新增的 Clustering Loss)
    print(f"Initializing model with latent_dim={model_config.get('latent_dim', 32)}...")
    model = Poem2LayoutGenerator(
        bert_path=model_config['bert_path'],
        num_classes=num_element_classes, # 实际元素类别数 (9)
        # --- BBox Discrete Parameters (Legacy) ---
        num_bbox_bins=model_config.get('num_bbox_bins', 1000),
        bbox_embed_dim=model_config.get('bbox_embed_dim', 24),
        # --------------------------------------
        hidden_size=model_config['hidden_size'],
        bb_size=model_config['bb_size'],
        decoder_layers=model_config['decoder_layers'],
        decoder_heads=model_config['decoder_heads'],
        dropout=model_config['dropout'],
        
        # === CVAE 参数 ===
        latent_dim=model_config.get('latent_dim', 32),
        
        # --- 传入所有损失权重 (Updated for V5.4) ---
        coord_loss_weight=model_config.get('coord_loss_weight', 0.0),
        iou_loss_weight=model_config.get('iou_loss_weight', 1.0), 
        reg_loss_weight=model_config.get('reg_loss_weight', 1.0),    
        cls_loss_weight=model_config.get('cls_loss_weight', 0.0),    
        count_loss_weight=model_config.get('count_loss_weight', 0.0),
        area_loss_weight=model_config.get('area_loss_weight', 1.0),
        
        # 核心逻辑权重
        relation_loss_weight=model_config.get('relation_loss_weight', 5.0),
        overlap_loss_weight=model_config.get('overlap_loss_weight', 3.0),
        size_loss_weight=model_config.get('size_loss_weight', 2.0),
        
        # 审美权重
        alignment_loss_weight=model_config.get('alignment_loss_weight', 0.0),
        balance_loss_weight=model_config.get('balance_loss_weight', 0.0),
        
        # [NEW V5.4] 聚类损失权重
        clustering_loss_weight=model_config.get('clustering_loss_weight', 1.0),
        
        class_weights=class_weights_tensor 
        # -----------------------------------------
    )

    # 4. Split dataset and init data loaders
    total_size = len(dataset)
    train_size = int(0.8 * total_size) # 80%
    val_size = int(0.1 * total_size)   # 10%
    test_size = total_size - train_size - val_size # 剩余为 10%

    # 执行 80/10/10 随机划分
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=layout_collate_fn 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=layout_collate_fn 
    )
    # 增加测试集加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=layout_collate_fn 
    )
    
    # --- 获取 tokenizer 和一个固定样例 ---
    tokenizer = dataset.tokenizer 
    # 从验证集中选择第一个样例
    example_idx_in_full_dataset = val_dataset.indices[0]
    example_poem = dataset.data[example_idx_in_full_dataset]
    
    # **打印固定推理样例的 KG 向量和空间矩阵**
    print("\n---------------------------------------------------")
    print(f"Inference Example Poem: '{example_poem['poem']}'")
    print(f"Inference Example GT Boxes: {example_poem['boxes']}")
    print("-------------------- KG DEBUG ---------------------")
    
    # 1. 视觉向量
    kg_vector_example = dataset.pkg.extract_visual_feature_vector(example_poem['poem'])
    print(f"KG Vector (0:mountain(2), 1:water(3), ..., 8:animal(10)): {kg_vector_example.tolist()}")
    
    # 2. [NEW] 空间矩阵
    kg_spatial_matrix_example = dataset.pkg.extract_spatial_matrix(example_poem['poem'])
    print("Spatial Matrix (9x9):")
    print(kg_spatial_matrix_example)
    print("---------------------------------------------------\n")

    # 5. Init trainer and start training
    # 将 test_loader 传递给 Trainer
    trainer = trainers.LayoutTrainer(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
    trainer.train()

if __name__ == "__main__":
    main()