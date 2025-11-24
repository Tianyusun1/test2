import torch

class PoetryKnowledgeGraph:
    def __init__(self):
        # 实体类型字典
        self.entities = {
            'poem_entity': set(),      # 诗词意象词
            'visual_entity': set(),    # 视觉元素
            'scene': set(),            # 场景/主题
            'spatial_concept': set(),  # 空间概念
            'attribute': set()         # 属性
        }
        
        # 关系图谱 (关系名 -> (头实体类型, 尾实体类型))
        self.relations = {
            # 诗词->视觉映射
            'links_to': ('poem_entity', 'visual_entity'),
            # 视觉-视觉空间关系
            'crosses': ('visual_entity', 'visual_entity'),
            'on': ('visual_entity', 'visual_entity'),
            'grows_near': ('visual_entity', 'visual_entity'),
            'in_background_of': ('visual_entity', 'visual_entity'),
            'beside': ('visual_entity', 'visual_entity'),
            'above': ('visual_entity', 'visual_entity'),
            'flows_through': ('visual_entity', 'visual_entity'),
            'surrounds': ('visual_entity', 'visual_entity'),
            'under': ('visual_entity', 'visual_entity'),
            'flies_over': ('visual_entity', 'visual_entity'),
            'grows_under': ('visual_entity', 'visual_entity'),
            'reflects_in': ('visual_entity', 'visual_entity'),
            'descends_to': ('visual_entity', 'visual_entity'),
            # 诗词->场景
            'belongs_to': ('poem_entity', 'scene'),
            # 场景->视觉
            'implies': ('scene', 'visual_entity'),
            'typically_contains': ('scene', 'visual_entity'),
            # 属性
            'has_attribute': ('visual_entity', 'attribute'),
            'conveys_mood': ('scene', 'attribute')
        }
        
        # 三元组集合 (head, relation, tail)
        self.triplets = set()
        
        # 为每个视觉类别分配ID映射 (2-10)
        self.visual_class_mapping = {
            'mountain': 2,
            'water': 3,
            'people': 4,
            'tree': 5,
            'building': 6,
            'bridge': 7,
            'flower': 8,
            'bird': 9,
            'animal': 10
        }
        self.num_classes = 9 # 类别总数
        
        # 每个视觉类别的所有同义词/变体
        self.visual_synonyms = self._init_visual_synonyms()
        
        # 构建知识图谱
        self._build_knowledge_graph()
        
        # --- NEW: 构建快速查找表，用于特征提取 ---
        self._build_lookup_tables()
    
    def _init_visual_synonyms(self):
        """初始化每个视觉类别的所有同义词和变体"""
        synonyms = {
            'mountain': ['山', '峰', '峦', '丘', '岭', '岗', '峰峦', '山岭', '山峰', '山丘', '山岗', '崇山', '峻岭', '高山', '远山', '近山'],
            'water': ['水', '河', '溪', '江', '湖', '海', '泉', '瀑', '涧', '潭', '池', '波', '浪', '流', '潮', '水波', '流水', '河水', '溪水', '湖水', '江水', '瀑布', '清泉', '深潭', '碧波', '涟漪'],
            'people': ['人', '士', '客', '渔', '樵', '僧', '道', '隐', '夫', '女', '童', '子', '行', '旅', '舟子', '渔夫', '樵夫', '隐士', '僧人', '道士', '行人', '旅人', '女子', '童子', '渔人', '老翁', '仙人'],
            'tree': ['树', '木', '松', '柏', '柳', '竹', '梅', '枫', '桂', '梧桐', '杨', '槐', '榆', '樟', '杉', '藤', '花树', '果树', '老树', '枯树', '青松', '翠柏', '垂柳', '翠竹', '梅花', '枫树', '桂树'],
            'building': ['屋', '楼', '阁', '亭', '台', '寺', '庙', '庵', '观', '宫', '殿', '宅', '院', '房', '舍', '轩', '榭', '廊', '桥头屋', '茅屋', '草堂', '楼阁', '亭台', '寺庙', '道观', '宫殿', '宅院', '庭院', '廊桥', '水榭', '山亭', '水亭', '画舫'],
            'bridge': ['桥', '石桥', '木桥', '拱桥', '曲桥', '栈桥', '浮桥', '廊桥', '竹桥', '独木桥', '石拱桥', '九曲桥', '风雨桥', '虹桥', '小桥', '长桥', '断桥', '古桥', '木拱桥', '石板桥', '竹索桥', '铁索桥'],
            'flower': ['花', '草', '兰', '菊', '荷', '莲', '蒲', '苇', '菖', '芦', '梅', '桃', '杏', '李', '梨', '樱', '杜鹃', '牡丹', '荷花', '莲花', '兰花', '菊花', '梅花', '桃花', '杏花', '樱花', '杜鹃花', '水仙', '荷花', '芦苇', '蒲草', '菖蒲'],
            'bird': ['鸟', '雀', '燕', '雁', '鹤', '鸥', '鹭', '莺', '鹧', '鸪', '鹊', '鸦', '鹰', '鹏', '凤', '鸾', '孔雀', '白鹭', '黄莺', '杜鹃', '喜鹊', '乌鸦', '大鹏', '仙鹤', '海鸥', '鹧鸪', '沙鸥', '翠鸟', '燕子', '归鸿', '鸿雁', '惊鸿'],
            'animal': ['兽', '鹿', '马', '牛', '虎', '豹', '猿', '猴', '兔', '狐', '狼', '蛇', '鱼', '龙', '凤', '龟', '鹤', '牛', '马', '鹿', '虎', '豹', '猿猴', '白兔', '狐狸', '野狼', '游鱼', '蛟龙', '凤凰', '龟', '仙鹤']
        }
        return synonyms
    
    def _build_knowledge_graph(self):
        """构建完整的知识图谱"""
        self._add_poem_entity_mappings()
        self._add_spatial_relations()
        self._add_scene_themes()
        self._add_attributes()
    
    def _add_poem_entity_mappings(self):
        """添加诗词实体到视觉实体的映射"""
        # 山水类
        mountain_keywords = ['山', '峰', '峦', '丘', '岭', '崖', '岩', '峰峦', '山岭', '崇山', '峻岭', 
                             '青山', '远山', '近山', '高山', '云山', '空山', '寒山', '千山', '万壑']
        water_keywords = ['水', '河', '溪', '江', '湖', '海', '泉', '瀑', '涧', '潭', '池', '波', '浪', '流', '潮',
                         '清流', '碧水', '绿水', '秋水', '春水', '流水', '溪流', '江流', '湖水', '河水', '泉水', '瀑布']
        
        # 人物类
        people_keywords = ['人', '士', '客', '渔', '樵', '僧', '道', '隐', '夫', '女', '童', '子', '行', '旅', '舟',
                          '渔人', '樵夫', '隐士', '高僧', '道士', '游人', '旅人', '故人', '仙人', '老翁', '童子', '美人', '佳人']
        
        # 植物类
        tree_keywords = ['树', '木', '松', '柏', '柳', '竹', '梅', '枫', '桂', '梧桐', '杨', '槐', '榆', '樟', '藤',
                        '青松', '翠柏', '垂柳', '翠竹', '梅花', '枫树', '桂树', '梧桐树', '老树', '枯树', '绿树', '花树']
        flower_keywords = ['花', '草', '兰', '菊', '荷', '莲', '梅', '桃', '杏', '李', '梨', '樱', '杜鹃',
                          '牡丹', '荷花', '莲花', '兰花', '菊花', '梅花', '桃花', '杏花', '樱花', '杜鹃花', '水仙', '芦苇', '蒲草']
        
        # 建筑类
        building_keywords = ['屋', '楼', '阁', '亭', '台', '寺', '庙', '庵', '观', '宫', '殿', '宅', '院', '房', '舍',
                            '轩', '榭', '廊', '桥头屋', '茅屋', '草堂', '楼阁', '亭台', '寺庙', '道观', '宫殿', '宅院', '庭院', '水榭']
        bridge_keywords = ['桥', '石桥', '木桥', '拱桥', '曲桥', '栈桥', '浮桥', '廊桥', '竹桥', '独木桥',
                          '石拱桥', '九曲桥', '风雨桥', '虹桥', '小桥', '长桥', '断桥', '古桥', '木拱桥', '石板桥']
        
        # 飞禽走兽
        bird_keywords = ['鸟', '雀', '燕', '雁', '鹤', '鸥', '鹭', '莺', '鹧', '鸪', '鹊', '鸦', '鹰', '鹏', '凤', '鸾',
                        '孔雀', '白鹭', '黄莺', '杜鹃', '喜鹊', '乌鸦', '大鹏', '仙鹤', '海鸥', '鹧鸪', '沙鸥', '翠鸟', '归鸿', '鸿雁']
        animal_keywords = ['兽', '鹿', '马', '牛', '虎', '豹', '猿', '猴', '兔', '狐', '狼', '蛇', '鱼', '龙', '凤', '龟',
                          '鹿', '马', '虎', '豹', '猿猴', '白兔', '狐狸', '野狼', '游鱼', '蛟龙', '凤凰', '龟', '仙鹤']
        
        # 为每个类别添加映射
        categories = [
            ('mountain', mountain_keywords),
            ('water', water_keywords),
            ('people', people_keywords),
            ('tree', tree_keywords),
            ('flower', flower_keywords),  # 注意：flower是单独类别
            ('building', building_keywords),
            ('bridge', bridge_keywords),
            ('bird', bird_keywords),
            ('animal', animal_keywords)
        ]
        
        for class_name, keywords in categories:
            for keyword in keywords:
                self._add_triplet(keyword, 'links_to', class_name, 'poem_entity', 'visual_entity')
    
    def _add_spatial_relations(self):
        """添加视觉元素之间的空间关系"""
        # 桥与水
        self._add_triplet('bridge', 'crosses', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('stone_bridge', 'crosses', 'river', 'visual_entity', 'visual_entity')
        self._add_triplet('wooden_bridge', 'crosses', 'stream', 'visual_entity', 'visual_entity')
        self._add_triplet('arched_bridge', 'crosses', 'lake', 'visual_entity', 'visual_entity')
        
        # 人与桥
        self._add_triplet('people', 'on', 'bridge', 'visual_entity', 'visual_entity')
        self._add_triplet('fisherman', 'beside', 'bridge', 'visual_entity', 'visual_entity')
        
        # 树与水
        self._add_triplet('tree', 'grows_near', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('willow', 'hangs_over', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('pine', 'stands_by', 'lake', 'visual_entity', 'visual_entity')
        
        # 山与水
        self._add_triplet('mountain', 'in_background_of', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('peak', 'rises_above', 'river', 'visual_entity', 'visual_entity')
        self._add_triplet('cliff', 'descends_to', 'water', 'visual_entity', 'visual_entity')
        
        # 建筑与水
        self._add_triplet('building', 'beside', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('pavilion', 'by', 'lake', 'visual_entity', 'visual_entity')
        self._add_triplet('temple', 'on_mountain', 'mountain', 'visual_entity', 'visual_entity')
        
        # 云与山
        self._add_triplet('cloud', 'above', 'mountain', 'visual_entity', 'visual_entity')
        self._add_triplet('mist', 'surrounds', 'peak', 'visual_entity', 'visual_entity')
        
        # 鸟与水/山
        self._add_triplet('bird', 'flies_over', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('wild_goose', 'flies_above', 'mountain', 'visual_entity', 'visual_entity')
        
        # 花与树
        self._add_triplet('flower', 'grows_under', 'tree', 'visual_entity', 'visual_entity')
        self._add_triplet('plum_blossom', 'blooms_on', 'branch', 'visual_entity', 'visual_entity')
        
        # 人与自然
        self._add_triplet('people', 'walk_on', 'path', 'visual_entity', 'visual_entity')
        self._add_triplet('fisherman', 'fishes_in', 'river', 'visual_entity', 'visual_entity')
        self._add_triplet('woodcutter', 'works_in', 'forest', 'visual_entity', 'visual_entity')
        
        # 水体关系
        self._add_triplet('river', 'flows_into', 'lake', 'visual_entity', 'visual_entity')
        self._add_triplet('waterfall', 'flows_into', 'pool', 'visual_entity', 'visual_entity')
        self._add_triplet('mountain', 'reflects_in', 'lake', 'visual_entity', 'visual_entity')
        
        # 人物活动
        self._add_triplet('people', 'row', 'boat', 'visual_entity', 'visual_entity')
        self._add_triplet('people', 'sit_in', 'pavilion', 'visual_entity', 'visual_entity')
        self._add_triplet('monk', 'meditates_in', 'temple', 'visual_entity', 'visual_entity')
        
        # 建筑关系
        self._add_triplet('pagoda', 'stands_on', 'hill', 'visual_entity', 'visual_entity')
        self._add_triplet('bridge', 'connects', 'shore', 'visual_entity', 'visual_entity')
        
        # 山体关系
        self._add_triplet('path', 'winds_up', 'mountain', 'visual_entity', 'visual_entity')
        self._add_triplet('valley', 'lies_between', 'peaks', 'visual_entity', 'visual_entity')
    
    def _add_scene_themes(self):
        """添加场景/主题及其相关元素"""
        # 场景定义
        scenes = [
            ('misty_mountains', ['空山', '云山', '雾山', '烟山', '山色空蒙', '云雾缭绕']),
            ('riverside_scene', ['小桥流水', '江畔', '水边', '湖畔', '溪畔', '河岸']),
            ('rainy_day', ['雨', '雨亦奇', '细雨', '烟雨', '雨蒙蒙', '雨霏霏']),
            ('spring_scene', ['春', '春色', '春风', '春水', '春山', '春意']),
            ('autumn_scene', ['秋', '秋色', '秋水', '秋山', '秋风', '落叶']),
            ('winter_scene', ['冬', '雪', '冰', '寒山', '雪峰', '雪溪']),
            ('buddhist_retreat', ['寺', '庙', '庵', '僧', '禅', '佛']),
            ('taoist_refuge', ['观', '道', '仙', '隐', '道士', '仙境']),
            ('fishing_scene', ['渔', '舟', '钓', '渔夫', '垂钓', '渔舟']),
            ('woodcutting_scene', ['樵', '伐', '砍', '樵夫', '山林', '伐木']),
            ('scholar_retreat', ['读书', '吟诗', '隐士', '草堂', '书斋', '文人'])
        ]
        
        # 为每个场景添加映射
        for scene_name, keywords in scenes:
            for keyword in keywords:
                self._add_triplet(keyword, 'belongs_to', scene_name, 'poem_entity', 'scene')
        
        # 场景蕴含的视觉元素
        scene_implications = {
            'misty_mountains': ['mountain', 'cloud', 'mist', 'path', 'temple'],
            'riverside_scene': ['water', 'bridge', 'tree', 'building', 'people', 'boat', 'flower'],
            'rainy_day': ['cloud', 'rain', 'water', 'umbrella', 'people'],
            'spring_scene': ['tree', 'flower', 'bird', 'butterfly', 'people', 'water'],
            'autumn_scene': ['tree', 'red_leaf', 'mountain', 'water', 'bird', 'people'],
            'winter_scene': ['snow', 'mountain', 'water', 'people', 'building'],
            'buddhist_retreat': ['temple', 'monk', 'mountain', 'path', 'tree', 'bell'],
            'taoist_refuge': ['pavilion', ' Daoist_monk', 'mountain', 'cloud', 'crane', 'immortal'],
            'fishing_scene': ['water', 'boat', 'fisherman', 'rod', 'reeds', 'bird'],
            'woodcutting_scene': ['mountain', 'tree', 'woodcutter', 'axe', 'path', 'forest'],
            'scholar_retreat': ['building', 'scholar', 'book', 'pavilion', 'garden', 'ink', 'brush']
        }
        
        for scene, entities in scene_implications.items():
            for entity in entities:
                self._add_triplet(scene, 'implies', entity, 'scene', 'visual_entity')
        
        # 场景典型包含
        scene_contains = {
            'misty_mountains': ['distant_mountains', 'clouds', 'winding_path', 'hidden_temple'],
            'riverside_scene': ['curved_bridge', 'rippling_water', 'willow_trees', 'pavilion_by_water', 'fishing_boat'],
            'rainy_day': ['rain_streaks', 'distant_mountains', 'misty_water', 'people_with_umbrellas', 'dripping_eaves'],
            'spring_scene': ['cherry_blossoms', 'swallows', 'frogs', 'new_green_leaves', 'butterflies', 'gentle_breeze'],
            'autumn_scene': ['maple_trees_with_red_leaves', 'geese_flying_south', 'harvest_fields', 'moon_reflection_on_water'],
            'winter_scene': ['snow_covered_mountains', 'frozen_lake', 'bare_trees', 'people_in_thick_clothes', 'warm_cottage'],
            'buddhist_retreat': ['ancient_temple', 'stone_stairs', 'bell_tower', 'praying_monks', 'incense_smoke'],
            'taoist_refuge': ['mountain_pavilion', 'crane', 'pine_trees', 'flowing_robes', 'mysterious_clouds', 'immortal_figures'],
            'fishing_scene': ['bamboo_pole', 'reeds_along_shore', 'ripples_on_water', 'distant_mountains', 'heron_watching'],
            'woodcutting_scene': ['axe', 'cut_logs', 'dense_forest', 'mountain_path', 'carrying_firewood', 'waterfall_nearby'],
            'scholar_retreat': ['scrolls', 'ink_stone', 'brush', 'lute', 'crane_standing_by_window', 'moon_viewing_pavilion']
        }
        
        for scene, elements in scene_contains.items():
            for element in elements:
                self._add_triplet(scene, 'typically_contains', element, 'scene', 'visual_entity')
    
    def _add_attributes(self):
        """添加属性和情感基调"""
        # 视觉元素属性
        visual_attributes = {
            'mountain': ['tall', 'majestic', 'distant', 'misty', 'steep', 'green', 'snowy', 'ancient'],
            'water': ['clear', 'rippled', 'flowing', 'still', 'deep', 'shallow', 'broad', 'narrow'],
            'people': ['small', 'distant', 'walking', 'sitting', 'fishing', 'rowing', 'meditating', 'contemplating'],
            'tree': ['tall', 'ancient', 'gnarled', 'green', 'red_autumn_leaves', 'bare_winter', 'blooming', 'dense_foliage'],
            'building': ['small', 'ancient', 'wooden', 'stone', 'painted_roof', 'open_window', 'hanging_lantern', 'enclosed_courtyard'],
            'bridge': ['curved', 'arched', 'stone', 'wood', 'simple', 'ornate', 'weathered', 'covered'],
            'flower': ['blossoming', 'dewy', 'colorful', 'delicate', 'wild_growing', 'planted_in_garden', 'fragrant', 'scattered_petals'],
            'bird': ['flying', 'perched', 'calling', 'flock', 'solitary', 'white_crane', 'golden_oriole', 'wild_goose_in_flight'],
            'animal': ['grazing', 'drinking', 'hiding', 'running', 'peaceful', 'majestic_deer', 'white_rabbit', 'monkeys_in_trees']
        }
        
        for entity, attrs in visual_attributes.items():
            for attr in attrs:
                self._add_triplet(entity, 'has_attribute', attr, 'visual_entity', 'attribute')
        
        # 场景情感基调
        scene_moods = {
            'misty_mountains': ['tranquil', 'mysterious', 'majestic', 'contemplative', 'ethereal'],
            'riverside_scene': ['peaceful', 'harmonious', 'leisurely', 'refreshing', 'melancholy'],
            'rainy_day': ['melancholic', 'tranquil', 'refreshing', 'cool', 'introspective'],
            'spring_scene': ['joyful', 'hopeful', 'vibrant', 'renewal', 'gentle'],
            'autumn_scene': ['nostalgic', 'mellow', 'reflective', 'serene', 'bittersweet'],
            'winter_scene': ['serene', 'pure', 'isolated', 'contemplative', 'resilient'],
            'buddhist_retreat': ['tranquil', 'serene', 'detached', 'spiritual', 'meditative'],
            'taoist_refuge': ['mysterious', 'eternal', 'natural', 'free', 'transcendent'],
            'fishing_scene': ['leisurely', 'peaceful', 'self_sufficient', 'harmonious', 'simple'],
            'woodcutting_scene': ['industrious', 'rugged', 'natural', 'practical', 'connected_to_earth'],
            'scholar_retreat': ['refined', 'thoughtful', 'cultured', 'contemplative', 'aesthetic']
        }
        
        for scene, moods in scene_moods.items():
            for mood in moods:
                self._add_triplet(scene, 'conveys_mood', mood, 'scene', 'attribute')
    
    def _add_triplet(self, head, relation, tail, head_type=None, tail_type=None):
        """添加一个三元组到图谱"""
        self.triplets.add((head, relation, tail))
        if head_type: self.entities[head_type].add(head)
        if tail_type: self.entities[tail_type].add(tail)
    
    # --- NEW: 辅助方法 - 构建查找表 ---
    def _build_lookup_tables(self):
        """
        构建反向查找表，用于快速特征提取:
        1. keyword_to_class_id: 诗词关键词 -> 视觉类别 ID (2-10)
        2. scene_to_implied_ids: 场景关键词 -> [隐含的视觉类别 ID 列表]
        """
        self.keyword_to_class_id = {}
        self.scene_to_implied_ids = {}
        
        # 遍历三元组构建索引
        for head, rel, tail in self.triplets:
            # 1. 直接映射: poem_entity -> links_to -> visual_entity
            if rel == 'links_to':
                cls_id = self.get_visual_class_id(tail)
                if cls_id:
                    self.keyword_to_class_id[head] = cls_id
            
            # 2. 场景映射: poem_entity -> belongs_to -> scene
            elif rel == 'belongs_to':
                # 这里的 head 是诗词词汇 (如 '垂钓'), tail 是场景 (如 'fishing_scene')
                # 我们需要先存下 词->场景 的映射，之后在 extract 时再查 场景->视觉元素
                if head not in self.scene_to_implied_ids:
                    self.scene_to_implied_ids[head] = set()
                
                # 查找该场景 imply 的所有视觉元素
                implied_elements = self.get_scene_elements(tail)
                for elem in implied_elements:
                    cls_id = self.get_visual_class_id(elem)
                    if cls_id:
                        self.scene_to_implied_ids[head].add(cls_id)

    def get_visual_class_id(self, visual_entity):
        """获取视觉实体对应的类别ID"""
        if visual_entity in self.visual_class_mapping:
            return self.visual_class_mapping[visual_entity]
        return None
    
    def get_scene_elements(self, scene_name):
        """获取场景隐含的所有视觉元素"""
        elements = set()
        for head, rel, tail in self.triplets:
            if head == scene_name and (rel == 'implies' or rel == 'typically_contains'):
                elements.add(tail)
        return elements

    # --- NEW: 核心特征提取方法 ---
    def extract_visual_feature_vector(self, poem_text: str) -> torch.Tensor:
        """
        输入诗歌文本，利用KG提取视觉元素特征向量。
        
        Returns:
            vector: 长度为 9 的 Tensor (对应内部 index 0-8, 即类别 2-10)
                    1.0 表示该元素在诗中被明确提及或强烈暗示。
        """
        # 初始化 9 维向量 (对应 2-10)
        # Index 0 -> Class 2 (mountain), Index 1 -> Class 3 (water), ...
        visual_vector = torch.zeros(self.num_classes)
        
        # 简单的字符串匹配 (实际项目中可以使用分词后匹配，这里为了简化直接查子串)
        # 1. 检查直接关键词 (Direct Keywords)
        for keyword, cls_id in self.keyword_to_class_id.items():
            if keyword in poem_text:
                # 映射 ID (2-10) 到 Vector Index (0-8)
                idx = cls_id - 2
                if 0 <= idx < self.num_classes:
                    visual_vector[idx] = 1.0
        
        # 2. 检查场景暗示 (Scene Implications)
        # 例如: 诗中有 "垂钓" -> 属于 fishing_scene -> 暗示 water, boat(other)
        for keyword, implied_ids in self.scene_to_implied_ids.items():
            if keyword in poem_text:
                for cls_id in implied_ids:
                    idx = cls_id - 2
                    if 0 <= idx < self.num_classes:
                        # 场景暗示的权重可以设为 1.0 (强约束) 或 0.5 (弱约束)
                        # 这里设为 1.0，表示只要场景出现，相关元素就"应该"出现
                        visual_vector[idx] = 1.0
                        
        return visual_vector

    def visualize(self):
        """(原有可视化代码保持不变)"""
        pass # (为了节省篇幅，此处省略，请保留您原有的 visualize 方法内容)

# 使用示例
if __name__ == "__main__":
    pkg = PoetryKnowledgeGraph()
    
    test_poem_1 = "空山新雨后"
    vec_1 = pkg.extract_visual_feature_vector(test_poem_1)
    print(f"Poem: '{test_poem_1}' -> Vector: {vec_1.tolist()}")
    # 预期: 空山->mountain(idx 0), 雨->rainy_day->implies water(idx 1)
    
    test_poem_2 = "孤舟蓑笠翁，独钓寒江雪"
    vec_2 = pkg.extract_visual_feature_vector(test_poem_2)
    print(f"Poem: '{test_poem_2}' -> Vector: {vec_2.tolist()}")
    # 预期: 舟->boat(other, ignored), 钓->fishing_scene->implies water(idx 1), 江->water(idx 1), 雪->winter->snow