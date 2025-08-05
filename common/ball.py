from collections import deque
from typing import List

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    一个用于在帧上标注具有不同半径和颜色的圆圈以进行球跟踪的类。

    该标注器通过绘制尺寸递减、颜色变化的圆圈来创建轨迹效果，
    以可视化球随时间变化的轨迹。

    # 架构师视角:
    # 这个类是专门为可视化设计的。它将检测到的球的位置
    # 存储在一个固定大小的队列（`deque`）中，从而可以创建出
    # “拖尾”效果。这种设计将标注逻辑与跟踪逻辑分离，
    # 使得代码更易于维护和扩展。例如，可以轻松更换
    # 不同的颜色方案或插值方法，而不会影响到核心的跟踪算法。

    Attributes:
        radius (int): 要绘制的圆的最大半径。
        buffer (deque): 用于存储最近坐标以进行标注的双端队列缓冲区。
        color_palette (sv.ColorPalette): 圆圈的调色板。
        thickness (int): 圆圈边框的粗细。
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):
        """
        初始化 BallAnnotator。
        
        Args:
            radius (int): 球标注圆圈的最大半径。
            buffer_size (int): 用于轨迹效果的最近位置保留数量。
            thickness (int): 圆圈边框的粗细。
        """
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        根据索引在 1 和最大半径之间插值半径。
        
        创建轨迹效果，其中较早的位置具有较小的圆圈。
        
        Args:
            i (int): 缓冲区中的当前索引。
            max_i (int): 缓冲区中的最大索引。
            
        Returns:
            int: 插值后的半径。
        """
        if max_i == 1:
            return self.radius
        # 从 1 到最大半径的线性插值
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        根据检测结果在帧上标注圆圈。
        
        Args:
            frame (np.ndarray): 要标注的帧。
            detections (sv.Detections): 包含球坐标的检测结果。
            
        Returns:
            np.ndarray: 标注后的帧。
        """
        # 获取锚点坐标（检测框的底部中心）
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        
        # 为缓冲区中的每个位置绘制圆圈
        for i, xy_positions in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            
            # 为每个检测到的球位置绘制一个圆圈
            for center in xy_positions:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    一个用于在视频帧之间跟踪足球位置的类。
    
    BallTracker 维护一个最近球位置的缓冲区，并使用此缓冲区
    通过选择最接近最近位置平均位置（质心）的检测来预测
    当前帧中的球位置。
    
    这种方法有助于过滤掉错误的检测并提供时间上的一致性。

    # 架构师视角:
    # 这个跟踪器实现了一种简单而有效的跟踪策略：基于质心的最近邻。
    # 它不依赖于复杂的状态估计模型（如卡尔曼滤波器），
    # 而是通过平滑历史位置来预测当前位置，这在处理快速移动且
    # 外观一致的物体（如足球）时非常有效。这种方法的优点是实现简单，
    # 计算开销小，并且对于短暂的遮挡具有一定的鲁棒性。
    
    Attributes:
        buffer (collections.deque): 用于存储最近球位置的双端队列缓冲区。
    """
    
    def __init__(self, buffer_size: int = 10):
        """
        初始化 BallTracker。
        
        Args:
            buffer_size (int): 用于平滑跟踪的最近位置保留数量。
        """
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        用新的检测更新缓冲区，并返回最接近最近位置质心的检测。
        
        该方法实现了一个简单但有效的跟踪策略：
        1. 将当前检测存储在缓冲区中
        2. 计算缓冲区中所有最近位置的质心
        3. 返回最接近该质心的检测
        
        Args:
            detections (sv.Detections): 当前帧的球检测结果。
            
        Returns:
            sv.Detections: 最接近最近位置质心的检测。
                如果没有检测，则返回未更改的输入检测。
        """
        # 获取所有检测的中心坐标
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        # 如果当前帧没有检测，则按原样返回
        if len(detections) == 0:
            return detections

        # 计算缓冲区中所有位置的质心
        all_positions = np.concatenate(list(self.buffer))
        if len(all_positions) == 0:
            return detections
            
        centroid = np.mean(all_positions, axis=0)
        
        # 查找最接近质心的检测
        distances = np.linalg.norm(xy - centroid, axis=1)
        closest_index = np.argmin(distances)
        
        # 仅返回最接近的检测
        return detections[[closest_index]]
    
    def get_trajectory(self) -> List[np.ndarray]:
        """
        获取最近球位置的轨迹。
        
        Returns:
            List[np.ndarray]: 来自缓冲区的位​​置数组列表。
        """
        return list(self.buffer)
    
    def reset(self) -> None:
        """
        通过清除位置缓冲区来重置跟踪器。
        """
        self.buffer.clear()
    
    def get_predicted_position(self) -> np.ndarray:
        """
        根据最近的轨迹获取预测的下一个位置。
        
        使用基于最后两个位置的简单线性外推。
        
        Returns:
            np.ndarray: 预测的位置，格式为 (x, y) 坐标。
                如果数据不足，则返回最后已知的位置。
        """
        if len(self.buffer) < 2:
            if len(self.buffer) == 1:
                positions = list(self.buffer)[-1]
                if len(positions) > 0:
                    return positions[0]  # 如果可用，返回第一个位置
            return np.array([0, 0])  # 如果没有数据，则返回默认值
        
        # 获取最后两个位置集
        recent_positions = list(self.buffer)
        if len(recent_positions[-1]) == 0 or len(recent_positions[-2]) == 0:
            return np.array([0, 0])
        
        # 使用最近两个位置集的质心
        current_centroid = np.mean(recent_positions[-1], axis=0)
        previous_centroid = np.mean(recent_positions[-2], axis=0)
        
        # 线性外推
        velocity = current_centroid - previous_centroid
        predicted_position = current_centroid + velocity
        
        return predicted_position
