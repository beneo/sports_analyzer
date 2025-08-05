from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt


class ViewTransformer:
    """
    一个使用单应性矩阵进行视图变换的类。

    # 架构师视角:
    # 这个类是计算机视觉中一个非常核心和基础的工具，
    # 尤其在体育分析领域。它封装了单应性变换的计算和应用，
    # 允许我们将图像或点从一个视角（例如，摄像机视角）
    # 转换到另一个视角（例如，鸟瞰图或标准化的场地坐标）。
    # 使用 RANSAC 算法来计算单应性矩阵，可以有效处理
    # 匹配点中的噪声和异常值，从而提高了变换的鲁棒性。
    # 将变换逻辑封装在这个类中，使得坐标映射和视角校正
    # 的代码可以被复用，并且与应用逻辑解耦。

    该类能够在不同坐标系之间进行转换，
    例如将摄像机视图坐标转换为场地坐标。
    常用于体育分析中的透视校正和坐标映射。
    """
    
    def __init__(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32]
    ) -> None:
        """
        使用源点和目标点初始化 ViewTransformer。
        
        Args:
            source (npt.NDArray[np.float32]): 用于单应性计算的源点。
                形状应为 (N, 2)，其中 N >= 4。
            target (npt.NDArray[np.float32]): 用于单应性计算的目标点。
                形状应为 (N, 2)，其中 N >= 4。
                
        Raises:
            ValueError: 如果源和目标形状不同，或者它们不是二维坐标，
                或者无法计算单应性矩阵。
        """
        if source.shape != target.shape:
            raise ValueError("源和目标必须具有相同的形状。")
        if source.shape[1] != 2:
            raise ValueError("源和目标点必须是二维坐标。")
        if source.shape[0] < 4:
            raise ValueError("单应性计算至少需要 4 对点。")

        # 确保为 float32 类型以兼容 OpenCV
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        # 使用 RANSAC 计算单应性矩阵以提高鲁棒性
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC)
        if self.m is None:
            raise ValueError("无法计算单应性矩阵。"
                           "请检查点是否共线并提供足够的对应关系。")

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        使用单应性矩阵变换给定的点。
        
        Args:
            points (npt.NDArray[np.float32]): 要变换的点。
                对于 N 个点，形状应为 (N, 2)。
                
        Returns:
            npt.NDArray[np.float32]: 与输入形状相同的变换后的点。
            
        Raises:
            ValueError: 如果点不是二维坐标。
        """
        if points.size == 0:
            return points
            
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("点必须是形状为 (N, 2) 的二维坐标。")

        # 为 OpenCV 透视变换重塑点
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # 应用透视变换
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        
        # 重塑回原始格式
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        使用单应性矩阵变换给定的图像。
        
        这对于从透视摄像机视图创建鸟瞰图或校正图像非常有用。
        
        Args:
            image (npt.NDArray[np.uint8]): 要变换的图像。
                可以是灰度图 (H, W) 或彩色图 (H, W, 3)。
            resolution_wh (Tuple[int, int]): 输出图像的宽度和高度。
                格式: (宽度, 高度)。
                
        Returns:
            npt.NDArray[np.uint8]: 具有指定分辨率的变换后的图像。
            
        Raises:
            ValueError: 如果图像既不是灰度图也不是彩色图。
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("图像必须是灰度图 (2D) 或彩色图 (3D)。")
            
        # 对图像应用透视变换
        return cv2.warpPerspective(image, self.m, resolution_wh)
    
    def get_homography_matrix(self) -> npt.NDArray[np.float32]:
        """
        获取单应性变换矩阵。
        
        Returns:
            npt.NDArray[np.float32]: 3x3 单应性矩阵。
        """
        return self.m.copy()
    
    def inverse_transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        使用逆单应性矩阵变换点。
        
        用于从目标坐标系转换回源坐标系。
        
        Args:
            points (npt.NDArray[np.float32]): 要进行逆变换的点。
                对于 N 个点，形状应为 (N, 2)。
                
        Returns:
            npt.NDArray[np.float32]: 逆变换后的点。
            
        Raises:
            ValueError: 如果点不是二维坐标。
        """
        if points.size == 0:
            return points
            
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("点必须是形状为 (N, 2) 的二维坐标。")

        # 计算逆单应性矩阵
        inverse_m = cv2.invert(self.m)[1]
        
        # 为 OpenCV 透视变换重塑点
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # 应用逆透视变换
        transformed_points = cv2.perspectiveTransform(reshaped_points, inverse_m)
        
        # 重塑回原始格式
        return transformed_points.reshape(-1, 2).astype(np.float32)
