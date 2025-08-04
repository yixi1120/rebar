import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import cv2

class PointCloudPreprocessor:
    def __init__(self):
        self.voxel_size = 0.01  # 1cm体素大小
        self.distance_threshold = 0.02  # 2cm距离阈值
        
    def load_point_cloud(self, file_path):
        """加载点云文件"""
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"加载点云: {len(pcd.points)} 个点")
        return pcd
    
    def remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """去除离群点"""
        cleaned_pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        print(f"去除离群点后: {len(cleaned_pcd.points)} 个点")
        return cleaned_pcd
    
    def downsample(self, pcd, voxel_size=None):
        """体素下采样"""
        if voxel_size is None:
            voxel_size = self.voxel_size
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"下采样后: {len(downsampled.points)} 个点")
        return downsampled
    
    def estimate_normals(self, pcd, radius=0.1, max_nn=30):
        """估计法向量"""
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamRadius(radius=radius, max_nn=max_nn)
        )
        return pcd
    
    def segment_ground(self, pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
        """地面分割"""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        ground_cloud = pcd.select_by_index(inliers)
        structure_cloud = pcd.select_by_index(inliers, invert=True)
        return ground_cloud, structure_cloud
    
    def extract_steel_structures(self, pcd):
        """提取钢结构特征"""
        # 基于几何特征提取钢结构
        # 这里可以添加更复杂的钢结构识别算法
        return pcd
    
    def preprocess_pipeline(self, file_path):
        """完整的预处理流程"""
        print("开始点云预处理...")
        
        # 1. 加载点云
        pcd = self.load_point_cloud(file_path)
        
        # 2. 去除离群点
        pcd = self.remove_outliers(pcd)
        
        # 3. 下采样
        pcd = self.downsample(pcd)
        
        # 4. 估计法向量
        pcd = self.estimate_normals(pcd)
        
        # 5. 地面分割
        ground, structure = self.segment_ground(pcd)
        
        # 6. 提取钢结构
        steel_structures = self.extract_steel_structures(structure)
        
        print("预处理完成!")
        return steel_structures, ground

# 使用示例
if __name__ == "__main__":
    preprocessor = PointCloudPreprocessor()
    steel_cloud, ground_cloud = preprocessor.preprocess_pipeline("example_point_cloud.ply")
    
    # 可视化结果
    o3d.visualization.draw_geometries([steel_cloud], window_name="钢结构点云") 