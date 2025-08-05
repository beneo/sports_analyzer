"""使用深度学习和无监督聚类的团队分类模块。"""

from typing import Generator, Iterable, List, TypeVar, Optional, Tuple
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv
import torch
import umap
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    从具有指定批大小的序列生成批次。
    
    Args:
        sequence (Iterable[V]): 要分批的输入序列。
        batch_size (int): 每个批次的大小。
        
    Yields:
        Generator[List[V], None, None]: 一个生成器，产生输入序列的批次。
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    一个分类器，使用预训练的 SiglipVisionModel 进行特征提取，
    UMAP 进行降维，以及 KMeans 进行聚类。
    
    # 架构师视角:
    # 这个类是本项目的核心创新之一。它采用了一种无监督的方法来解决团队分类问题，
    # 避免了需要大量标注数据的传统监督学习方法。流程如下：
    # 1. **特征提取**: 使用强大的视觉模型（Siglip）将球员的图像（crops）转换为高维特征向量。
    # 2. **降维**: 利用 UMAP 将高维特征投影到低维空间（默认为3维），这有助于可视化和聚类。
    # 3. **聚类**: 在低维空间中使用 KMeans 将球员分为两个簇，代表两个不同的球队。
    # 这种设计使得模型能够自动发现球员之间的视觉相似性（如队服颜色），而无需任何先验知识。

    该类通过使用无监督学习分析球员的视觉外观（球衣颜色、图案等）来自动识别球员的团队归属。
    """
    
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
        使用设备和批大小初始化 TeamClassifier。
        
        Args:
            device (str): 运行模型的设备（'cpu' 或 'cuda'）。
            batch_size (int): 处理图像的批大小。
        """
        self.device = device
        self.batch_size = batch_size
        
        # 初始化视觉模型和处理器
        try:
            self.features_model = SiglipVisionModel.from_pretrained(
                SIGLIP_MODEL_PATH).to(device)
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"加载 SiglipVisionModel 失败: {e}")
        
        # 初始化降维和聚类
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        
        # 存储拟合状态
        self._is_fitted = False

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        使用预训练的 SiglipVisionModel 从图像裁剪列表中提取特征。
        
        Args:
            crops (List[np.ndarray]): 图像裁剪列表（球员图像）。
                每个裁剪应为 BGR 格式（OpenCV 标准）。
                
        Returns:
            np.ndarray: 提取的特征，作为形状为 (N, feature_dim) 的 numpy 数组。
        """
        if not crops:
            return np.array([]).reshape(0, -1)
        
        # 将 BGR 转换为 PIL 格式以用于 transformers
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops_pil, self.batch_size)
        
        features_list = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc='提取球员特征'):
                try:
                    # 通过视觉模型处理批次
                    inputs = self.processor(
                        images=batch, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.features_model(**inputs)
                    
                    # 在空间维度上使用平均池化
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                    features_list.append(embeddings.cpu().numpy())
                    
                except Exception as e:
                    warnings.warn(f"处理批次时出错: {e}")
                    # 为失败的批次创建零特征
                    batch_size = len(batch)
                    feature_dim = 768  # SiglipVisionModel 默认特征维度
                    features_list.append(np.zeros((batch_size, feature_dim)))

        if not features_list:
            return np.array([]).reshape(0, -1)
            
        return np.concatenate(features_list, axis=0)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        在图像裁剪列表上拟合分类器模型。
        
        该方法：
        1. 从球员裁剪中提取特征
        2. 使用 UMAP 进行降维
        3. 使用 KMeans 将球员聚类到团队中
        
        Args:
            crops (List[np.ndarray]): 用于训练的图像裁剪列表。
        """
        if not crops:
            raise ValueError("不能在空的裁剪列表上进行拟合")
        
        print(f"在 {len(crops)} 个球员裁剪上训练团队分类器...")
        
        # 提取特征
        features = self.extract_features(crops)
        
        if features.shape[0] == 0:
            raise ValueError("从裁剪中未提取到有效特征")
        
        # 应用降维
        print("应用 UMAP 降维...")
        projections = self.reducer.fit_transform(features)
        
        # 拟合聚类模型
        print("将球员聚类到团队中...")
        self.cluster_model.fit(projections)
        
        self._is_fitted = True
        print("团队分类器训练完成！")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        预测图像裁剪列表的簇标签。
        
        Args:
            crops (List[np.ndarray]): 要分类的图像裁剪列表。
            
        Returns:
            np.ndarray: 预测的簇标签（两个团队为 0 或 1）。
            
        Raises:
            RuntimeError: 如果分类器尚未拟合。
        """
        if not self._is_fitted:
            raise RuntimeError("在进行预测之前必须拟合 TeamClassifier。"
                             "请先调用 fit() 方法。")
        
        if len(crops) == 0:
            return np.array([])

        # 提取特征
        features = self.extract_features(crops)
        
        if features.shape[0] == 0:
            return np.array([])
        
        # 使用拟合的 UMAP 进行转换
        projections = self.reducer.transform(features)
        
        # 预测团队标签
        return self.cluster_model.predict(projections)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        获取降维空间中的簇中心。
        
        Returns:
            Optional[np.ndarray]: 如果已拟合，则返回簇中心，否则返回 None。
        """
        if not self._is_fitted:
            return None
        return self.cluster_model.cluster_centers_
    
    def predict_proba(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        预测样本的类别概率。
        
        注意：KMeans 本身不提供概率，因此该方法
        返回到簇中心的距离（越近=置信度越高）。
        
        Args:
            crops (List[np.ndarray]): 要分类的图像裁剪列表。
            
        Returns:
            Optional[np.ndarray]: 基于距离的置信度分数，如果未拟合则返回 None。
        """
        if not self._is_fitted:
            return None
            
        if len(crops) == 0:
            return np.array([])

        # 提取特征并转换
        features = self.extract_features(crops)
        if features.shape[0] == 0:
            return np.array([])
            
        projections = self.reducer.transform(features)
        
        # 计算到簇中心的距离
        distances = self.cluster_model.transform(projections)
        
        # 将距离转换为类似概率的分数（反距离）
        # 添加小的 epsilon 以避免除以零
        probabilities = 1.0 / (distances + 1e-8)
        
        # 归一化，使每行总和为 1
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        将拟合的模型组件保存到磁盘。
        
        Args:
            filepath (str): 保存模型的路径（不带扩展名）。
        """
        if not self._is_fitted:
            raise RuntimeError("无法保存未拟合的模型")
        
        import pickle
        
        model_data = {
            'reducer': self.reducer,
            'cluster_model': self.cluster_model,
            'device': self.device,
            'batch_size': self.batch_size,
            'is_fitted': self._is_fitted
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        从磁盘加载拟合的模型。
        
        Args:
            filepath (str): 保存模型的路径（不带扩展名）。
        """
        import pickle
        
        with open(f"{filepath}.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.reducer = model_data['reducer']
        self.cluster_model = model_data['cluster_model']
        self._is_fitted = model_data['is_fitted']
