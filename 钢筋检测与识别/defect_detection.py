import open3d as o3d
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

class SteelDefectDetector:
    def __init__(self):
        # 加载YOLO模型
        self.yolo_model = YOLO('yolov8n.pt')
        
        # 加载SAM模型
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(self.sam)
        
        # 缺陷类型定义
        self.defect_types = {
            'crack': '裂缝',
            'corrosion': '腐蚀',
            'deformation': '变形',
            'weld_defect': '焊接缺陷',
            'surface_damage': '表面损伤'
        }
    
    def point_cloud_to_image(self, pcd, resolution=1024):
        """将点云投影为2D图像用于检测"""
        # 获取点云边界框
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        
        # 创建2D投影图像
        image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # 将点云投影到2D平面
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points)
        
        # 归一化坐标到图像空间
        normalized_points = (points - min_bound) / (max_bound - min_bound)
        pixel_coords = (normalized_points[:, :2] * (resolution - 1)).astype(int)
        
        # 填充图像
        for i, coord in enumerate(pixel_coords):
            if 0 <= coord[0] < resolution and 0 <= coord[1] < resolution:
                image[coord[1], coord[0]] = colors[i] * 255
        
        return image, min_bound, max_bound
    
    def detect_defects_2d(self, image):
        """使用YOLO进行2D缺陷检测"""
        results = self.yolo_model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls,
                        'class_name': self.yolo_model.names[cls]
                    })
        
        return detections
    
    def segment_defects_sam(self, image, detections):
        """使用SAM进行精确分割"""
        self.sam_predictor.set_image(image)
        segmented_defects = []
        
        for detection in detections:
            bbox = detection['bbox']
            # 使用边界框作为SAM输入
            masks, scores, logits = self.sam_predictor.predict(
                box=bbox,
                multimask_output=True
            )
            
            # 选择最佳掩码
            best_mask = masks[np.argmax(scores)]
            
            segmented_defects.append({
                'mask': best_mask,
                'bbox': bbox,
                'confidence': detection['confidence'],
                'class_name': detection['class_name']
            })
        
        return segmented_defects
    
    def back_project_to_3d(self, segmented_defects, min_bound, max_bound, resolution=1024):
        """将2D分割结果投影回3D点云"""
        defect_points_3d = []
        
        for defect in segmented_defects:
            mask = defect['mask']
            bbox = defect['bbox']
            
            # 找到掩码中的像素坐标
            y_coords, x_coords = np.where(mask)
            
            # 将像素坐标转换回3D坐标
            for y, x in zip(y_coords, x_coords):
                # 归一化坐标
                norm_x = x / (resolution - 1)
                norm_y = y / (resolution - 1)
                
                # 转换回3D坐标
                point_3d = min_bound[:2] + np.array([norm_x, norm_y]) * (max_bound[:2] - min_bound[:2])
                # 假设Z坐标为中间值
                point_3d = np.append(point_3d, (min_bound[2] + max_bound[2]) / 2)
                
                defect_points_3d.append({
                    'point': point_3d,
                    'defect_type': defect['class_name'],
                    'confidence': defect['confidence']
                })
        
        return defect_points_3d
    
    def detect_defects_pipeline(self, pcd):
        """完整的缺陷检测流程"""
        print("开始缺陷检测...")
        
        # 1. 点云转2D图像
        image, min_bound, max_bound = self.point_cloud_to_image(pcd)
        
        # 2. YOLO检测
        detections = self.detect_defects_2d(image)
        
        # 3. SAM精确分割
        segmented_defects = self.segment_defects_sam(image, detections)
        
        # 4. 投影回3D
        defect_points_3d = self.back_project_to_3d(segmented_defects, min_bound, max_bound)
        
        # 5. 创建缺陷点云
        defect_pcd = self.create_defect_point_cloud(defect_points_3d)
        
        print(f"检测完成，发现 {len(defect_points_3d)} 个缺陷点")
        return defect_pcd, segmented_defects
    
    def create_defect_point_cloud(self, defect_points_3d):
        """创建缺陷点云"""
        if not defect_points_3d:
            return None
        
        points = np.array([d['point'] for d in defect_points_3d])
        
        # 根据缺陷类型设置颜色
        colors = []
        for defect in defect_points_3d:
            defect_type = defect['defect_type']
            if 'crack' in defect_type.lower():
                colors.append([1, 0, 0])  # 红色 - 裂缝
            elif 'corrosion' in defect_type.lower():
                colors.append([0, 1, 0])  # 绿色 - 腐蚀
            elif 'deformation' in defect_type.lower():
                colors.append([0, 0, 1])  # 蓝色 - 变形
            else:
                colors.append([1, 1, 0])  # 黄色 - 其他
        
        colors = np.array(colors)
        
        defect_pcd = o3d.geometry.PointCloud()
        defect_pcd.points = o3d.utility.Vector3dVector(points)
        defect_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return defect_pcd
    
    def visualize_defects(self, original_pcd, defect_pcd):
        """可视化缺陷检测结果"""
        geometries = [original_pcd]
        if defect_pcd is not None:
            geometries.append(defect_pcd)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="钢结构缺陷检测结果",
            point_show_normal=True
        )

# 使用示例
if __name__ == "__main__":
    # 加载预处理后的点云
    from point_cloud_preprocessing import PointCloudPreprocessor
    
    preprocessor = PointCloudPreprocessor()
    steel_cloud, _ = preprocessor.preprocess_pipeline("example_point_cloud.ply")
    
    # 缺陷检测
    detector = SteelDefectDetector()
    defect_pcd, segmented_defects = detector.detect_defects_pipeline(steel_cloud)
    
    # 可视化结果
    detector.visualize_defects(steel_cloud, defect_pcd) 