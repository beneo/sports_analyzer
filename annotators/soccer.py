"""足球场可视化与标注工具集。"""

from typing import Optional, List, Tuple, Union
import warnings

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from loguru import logger

from ..configs.soccer import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),  # 绿色场地
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    根据指定的尺寸、颜色和比例绘制足球场。
    
    # 架构师视角:
    # 这是一个基础的绘图函数，用于生成足球场的静态图像。
    # 它依赖于 `SoccerPitchConfiguration` 来获取场地尺寸，
    # 并使用 OpenCV (cv2) 进行实际的绘图操作。
    # 通过调整 `scale` 参数，可以生成不同分辨率的场地图像，
    # 这对于在不同大小的显示器或视频上进行叠加非常有用。

    Args:
        config: 包含场地尺寸和布局的配置对象。
        background_color: 场地背景颜色 (默认为绿色)。
        line_color: 场地线条颜色 (默认为白色)。
        padding: 场地周围的填充像素。
        line_thickness: 场地线条的粗细（以像素为单位）。
        point_radius: 罚球点半径（以像素为单位）。
        scale: 场地尺寸的缩放因子。
        
    Returns:
        np.ndarray: BGR 格式的足球场图像 numpy 数组。
    """
    # 计算缩放后的尺寸
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    # 创建场地背景
    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # 使用边定义绘制场地线条
    vertices = config.vertices
    for start_idx, end_idx in config.edges:
        # 转换为 0-based 索引
        start_vertex = vertices[start_idx - 1]
        end_vertex = vertices[end_idx - 1]
        
        # 缩放并平移点
        point1 = (
            int(start_vertex[0] * scale) + padding,
            int(start_vertex[1] * scale) + padding
        )
        point2 = (
            int(end_vertex[0] * scale) + padding,
            int(end_vertex[1] * scale) + padding
        )
        
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # 绘制中心圆
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    # 绘制罚球点
    penalty_spots = [
        (  # 左侧罚球点
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (  # 右侧罚球点
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1  # 填充圆形
        )

    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    在足球场上绘制点。
    
    # 架构师视角:
    # 此函数用于在现有的场地图像上（或新创建的图像上）绘制点，
    # 通常代表球员或球的位置。它支持自定义颜色、大小和标签，
    # 使得可视化更具信息量。通过 `pitch` 参数，
    # 可以实现图层的叠加，例如，先绘制场地，再绘制球员位置。

    Args:
        config: 包含场地尺寸的配置对象。
        xy: 要绘制的点的数组，形状为 (N, 2)。
        face_color: 点的表面颜色。
        edge_color: 点的边缘颜色。
        radius: 点的半径（以像素为单位）。
        thickness: 点边缘的粗细（以像素为单位）。
        padding: 场地周围的填充像素。
        scale: 场地尺寸的缩放因子。
        pitch: 用于绘制点的现有场地图像。如果为 None，则创建新场地。
        labels: 每个点的可选标签。
        
    Returns:
        np.ndarray: 带有绘制点的足球场图像。
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 验证输入
    if xy.size == 0:
        logger.warning("未提供要绘制的点")
        return pitch
        
    if len(xy.shape) != 2 or xy.shape[1] != 2:
        raise ValueError("点的数组必须为 (N, 2) 形状")

    # 绘制每个点
    for i, point in enumerate(xy):
        # 缩放并平移点
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        
        # 绘制填充圆形
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        
        # 绘制边缘圆形
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )
        
        # 如果提供了标签，则添加标签
        if labels is not None and i < len(labels):
            label_position = (
                scaled_point[0] - radius//2,
                scaled_point[1] + radius + 15
            )
            cv2.putText(
                img=pitch,
                text=str(labels[i]),
                org=label_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=edge_color.as_bgr(),
                thickness=1
            )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    alpha: float = 0.7
) -> np.ndarray:
    """
    在足球场上绘制路径。
    
    # 架构师视角:
    # 该函数用于可视化球员或球的运动轨迹。
    # 它通过在场地上绘制一系列连接的线段来表示路径。
    # `alpha` 参数允许路径具有半透明效果，
    # 这在多条路径重叠时非常有用，可以避免视觉混乱。

    Args:
        config: 包含场地尺寸的配置对象。
        paths: 路径列表，其中每个路径是 (x, y) 坐标的数组。
        color: 路径的颜色。
        thickness: 路径的粗细（以像素为单位）。
        padding: 场地周围的填充像素。
        scale: 场地尺寸的缩放因子。
        pitch: 用于绘制路径的现有场地图像。如果为 None，则创建新场地。
        alpha: 路径的透明度 (0.0 = 完全透明, 1.0 = 完全不透明)。
        
    Returns:
        np.ndarray: 带有绘制路径的足球场图像。
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 创建一个覆盖层以实现透明效果
    overlay = pitch.copy()

    # 绘制每条路径
    for path_idx, path in enumerate(paths):
        if path.size == 0:
            continue
            
        # 缩放并平移路径点
        scaled_path = []
        for point in path:
            if point.size < 2:
                continue
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            scaled_path.append(scaled_point)

        # 绘制路径段
        if len(scaled_path) >= 2:
            for i in range(len(scaled_path) - 1):
                cv2.line(
                    img=overlay,
                    pt1=scaled_path[i],
                    pt2=scaled_path[i + 1],
                    color=color.as_bgr(),
                    thickness=thickness
                )

    # 将覆盖层与原始图像混合
    if alpha < 1.0:
        pitch = cv2.addWeighted(pitch, 1 - alpha, overlay, alpha, 0)
    else:
        pitch = overlay

    return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.BLUE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    在足球场上绘制表示两队控制区域的 Voronoi 图。
    
    # 架构师视角:
    # Voronoi 图是一种强大的可视化工具，用于展示空间划分。
    # 在足球分析中，它可以直观地显示每个球员的“控制区域”。
    # 此函数通过计算场地上每个像素点到最近球员的距离，
    # 来确定该点属于哪个球队的控制范围。
    # 这种方法虽然计算量较大，但能生成精确的像素级控制图。

    该可视化显示了基于球员位置，使用最近邻分配，哪个球队控制了场地的每个区域。
    
    Args:
        config: 包含场地尺寸的配置对象。
        team_1_xy: 球队 1 球员的 (x, y) 坐标数组。
        team_2_xy: 球队 2 球员的 (x, y) 坐标数组。
        team_1_color: 代表球队 1 控制区域的颜色。
        team_2_color: 代表球队 2 控制区域的颜色。
        opacity: Voronoi 图覆盖层的不透明度。
        padding: 场地周围的填充像素。
        scale: 场地尺寸的缩放因子。
        pitch: 现有的场地图像。如果为 None，则创建新场地。
        
    Returns:
        np.ndarray: 带有 Voronoi 图覆盖层的足球场图像。
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 验证输入
    if team_1_xy.size == 0 and team_2_xy.size == 0:
        logger.warning("未提供用于 Voronoi 图的球员位置")
        return pitch

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    # 创建坐标网格
    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    # 将坐标调整到场地坐标系
    y_coordinates = y_coordinates - padding
    x_coordinates = x_coordinates - padding

    # 初始化控制掩码
    control_mask = np.zeros((scaled_width + 2 * padding, scaled_length + 2 * padding), dtype=bool)

    def calculate_distances(xy: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """计算从所有场地像素点到球员位置的距离。"""
        if xy.size == 0:
            return np.full_like(x_coords, np.inf, dtype=float)
            
        distances = []
        for player_pos in xy:
            dist = np.sqrt(
                (player_pos[0] * scale - x_coords) ** 2 + 
                (player_pos[1] * scale - y_coords) ** 2
            )
            distances.append(dist)
        
        return np.stack(distances, axis=0)

    # 计算到所有球员的距离
    if team_1_xy.size > 0:
        distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
        min_distances_team_1 = np.min(distances_team_1, axis=0)
    else:
        min_distances_team_1 = np.full_like(x_coordinates, np.inf, dtype=float)

    if team_2_xy.size > 0:
        distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)
        min_distances_team_2 = np.min(distances_team_2, axis=0)
    else:
        min_distances_team_2 = np.full_like(x_coordinates, np.inf, dtype=float)

    # 创建 Voronoi 图
    voronoi = np.zeros_like(pitch, dtype=np.uint8)
    
    # 根据最近邻分配控制权
    team_1_control = min_distances_team_1 < min_distances_team_2
    team_2_control = min_distances_team_2 < min_distances_team_1
    
    voronoi[team_1_control] = team_1_color.as_bgr()
    voronoi[team_2_control] = team_2_color.as_bgr()

    # 与原始场地图像混合
    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


def create_match_statistics_overlay(
    config: SoccerPitchConfiguration,
    team_1_stats: dict,
    team_2_stats: dict,
    padding: int = 50,
    scale: float = 0.1
) -> np.ndarray:
    """
    为比赛分析创建统计信息覆盖层。
    
    # 架构师视角:
    # 这个函数用于生成一个独立的图层，显示比赛的关键统计数据，
    # 例如控球率和射门次数。这种模块化的设计允许将统计信息
    # 轻松地叠加到视频帧或最终的分析报告中，而无需修改核心的视频处理逻辑。

    Args:
        config: 足球场配置。
        team_1_stats: 球队 1 的统计数据 (例如, {'控球率': 0.6, '射门次数': 5})。
        team_2_stats: 球队 2 的统计数据。
        padding: 场地周围的填充。
        scale: 场地的缩放因子。
        
    Returns:
        np.ndarray: 统计信息覆盖层图像。
    """
    pitch_height = int(config.width * scale) + 2 * padding
    pitch_width = int(config.length * scale) + 2 * padding
    
    # 创建具有透明背景的覆盖层
    overlay = np.zeros((pitch_height, pitch_width + 300, 3), dtype=np.uint8)
    
    # 添加统计文本
    stats_x = pitch_width + 20
    y_offset = 50
    
    # 球队 1 统计数据
    cv2.putText(overlay, "球队 1", (stats_x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    for key, value in team_1_stats.items():
        text = f"{key}: {value}"
        cv2.putText(overlay, text, (stats_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y_offset += 25
    
    y_offset += 30
    
    # 球队 2 统计数据
    cv2.putText(overlay, "球队 2", (stats_x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    for key, value in team_2_stats.items():
        text = f"{key}: {value}"
        cv2.putText(overlay, text, (stats_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        y_offset += 25
    
    return overlay


def draw_heatmap_on_pitch(
    config: SoccerPitchConfiguration,
    positions: np.ndarray,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6
) -> np.ndarray:
    """
    在场地上绘制球员位置的热力图。
    
    # 架构师视角:
    # 热力图是分析球员活动区域和战术模式的关键工具。
    # 此函数通过在每个球员位置上叠加一个高斯核来生成热力图，
    # 从而平滑地展示活动密度的分布。
    # 使用 `alpha` 参数可以控制热力图与场地背景的融合程度，
    # 确保了视觉上的清晰度。

    Args:
        config: 足球场配置。
        positions: (x, y) 位置数组，形状为 (N, 2)。
        padding: 场地周围的填充。
        scale: 场地的缩放因子。
        pitch: 现有的场地图像。如果为 None，则创建新场地。
        colormap: 用于热力图可视化的 OpenCV 颜色映射。
        alpha: 热力图覆盖层的透明度。
        
    Returns:
        np.ndarray: 带有热力图覆盖层的场地图像。
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)
    
    if positions.size == 0:
        return pitch
    
    # 创建热力图
    scaled_width = int(config.width * scale) + 2 * padding
    scaled_length = int(config.length * scale) + 2 * padding
    
    heatmap = np.zeros((scaled_width, scaled_length), dtype=np.float32)
    
    # 为每个位置添加高斯模糊
    for pos in positions:
        x = int(pos[0] * scale) + padding
        y = int(pos[1] * scale) + padding
        
        if 0 <= x < scaled_length and 0 <= y < scaled_width:
            # 添加高斯核
            kernel_size = 50
            kernel = cv2.getGaussianKernel(kernel_size, 15)
            kernel_2d = kernel @ kernel.T
            
            # 计算边界
            y1 = max(0, y - kernel_size//2)
            y2 = min(scaled_width, y + kernel_size//2)
            x1 = max(0, x - kernel_size//2)
            x2 = min(scaled_length, x + kernel_size//2)
            
            # 添加到热力图
            ky1 = kernel_size//2 - (y - y1)
            ky2 = ky1 + (y2 - y1)
            kx1 = kernel_size//2 - (x - x1)
            kx2 = kx1 + (x2 - x1)
            
            if ky1 >= 0 and ky2 <= kernel_size and kx1 >= 0 and kx2 <= kernel_size:
                heatmap[y1:y2, x1:x2] += kernel_2d[ky1:ky2, kx1:kx2]
    
    # 归一化热力图
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # 与场地图像混合
    result = cv2.addWeighted(pitch, 1 - alpha, heatmap_colored, alpha, 0)
    
    return result
