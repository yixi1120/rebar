#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钢筋检测标注系统
集成LabelMe和点云标注功能
"""

import json
import os
import cv2
import numpy as np
from datetime import datetime
import open3d as o3d
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class RebarAnnotationSystem:
    def __init__(self):
        self.annotation_config = {
            'project_name': 'RebarNet',
            'version': '1.0',
            'annotation_tools': ['LabelMe', 'PointCloud_Annotation'],
            'rebar_categories': [
                'main_rebar',      # 主筋
                'stirrup',         # 箍筋
                'distribution_rebar', # 分布筋
                'bent_rebar',      # 弯筋
                'hook_end',        # 弯钩
                'binding_wire',    # 绑扎铁丝
                'intersection'     # 交叉点
            ],
            'inspection_items': [
                'diameter',        # 直径
                'spacing',         # 间距
                'binding_quality', # 绑扎质量
                'hook_formation',  # 弯钩形成
                'stirrup_spacing', # 箍筋间距
                'reinforcement_ratio' # 配筋比例
            ]
        }
        
        self.labelme_config = {
            'image_format': ['jpg', 'png', 'bmp'],
            'annotation_format': 'json',
            'label_format': 'polygon',
            'export_format': ['json', 'xml', 'yolo']
        }
        
        self.pointcloud_config = {
            'point_cloud_format': ['ply', 'pcd', 'xyz'],
            'annotation_format': 'json',
            'visualization_tool': 'open3d'
        }
        
    def setup_labelme_environment(self):
        """设置LabelMe标注环境"""
        print("设置LabelMe标注环境...")
        
        # 创建标注目录结构
        directories = [
            'annotations/labelme',
            'annotations/pointcloud',
            'images/raw',
            'images/processed',
            'models/trained',
            'models/checkpoints',
            'datasets/train',
            'datasets/val',
            'datasets/test'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        
        # 创建LabelMe配置文件
        labelme_config = {
            'auto_save': True,
            'display_label_popup': True,
            'store_data': True,
            'keep_prev': False,
            'flags': None,
            'label_colors': {
                'main_rebar': '#FF0000',
                'stirrup': '#00FF00', 
                'distribution_rebar': '#0000FF',
                'bent_rebar': '#FFFF00',
                'hook_end': '#FF00FF',
                'binding_wire': '#00FFFF',
                'intersection': '#FF8000'
            }
        }
        
        with open('annotations/labelme/config.json', 'w', encoding='utf-8') as f:
            json.dump(labelme_config, f, ensure_ascii=False, indent=2)
        
        print("LabelMe环境设置完成!")
        return labelme_config
    
    def create_annotation_guidelines(self):
        """创建标注指南"""
        guidelines = {
            'rebar_annotation_rules': {
                'main_rebar': {
                    'description': '主要受力钢筋，通常为纵向布置',
                    'labeling_rules': [
                        '沿钢筋轴线标注完整长度',
                        '标注钢筋直径和间距',
                        '注意弯钩部分的标注'
                    ],
                    'quality_standards': {
                        'diameter_tolerance': '±0.3mm',
                        'spacing_tolerance': '±5mm',
                        'bending_angle': '135°±1.5°'
                    }
                },
                'stirrup': {
                    'description': '横向箍筋，用于约束主筋',
                    'labeling_rules': [
                        '标注箍筋的完整形状',
                        '标注箍筋间距',
                        '注意箍筋与主筋的连接点'
                    ],
                    'quality_standards': {
                        'spacing_tolerance': '±3mm',
                        'shape_consistency': '矩形或圆形',
                        'connection_quality': '绑扎牢固'
                    }
                },
                'distribution_rebar': {
                    'description': '分布钢筋，用于控制裂缝',
                    'labeling_rules': [
                        '标注分布筋的布置方向',
                        '标注分布筋的间距',
                        '注意分布筋的直径'
                    ]
                },
                'hook_end': {
                    'description': '钢筋端部弯钩',
                    'labeling_rules': [
                        '标注弯钩的角度',
                        '标注弯钩的长度',
                        '注意弯钩的方向'
                    ],
                    'quality_standards': {
                        'hook_angle': '135°±3°',
                        'hook_length': '≥50mm'
                    }
                },
                'binding_wire': {
                    'description': '绑扎铁丝',
                    'labeling_rules': [
                        '标注绑扎点的位置',
                        '标注绑扎的质量',
                        '注意绑扎的牢固程度'
                    ]
                }
            },
            'inspection_annotation_rules': {
                'diameter': {
                    'measurement_method': '使用卡尺或激光测量',
                    'tolerance': '±0.3mm',
                    'annotation_format': '数值(mm)'
                },
                'spacing': {
                    'measurement_method': '测量钢筋中心线间距',
                    'tolerance': '±5mm',
                    'annotation_format': '数值(mm)'
                },
                'binding_quality': {
                    'assessment_criteria': [
                        '绑扎是否牢固',
                        '铁丝是否拧紧',
                        '交叉点是否扎牢'
                    ],
                    'annotation_format': '布尔值(合格/不合格)'
                },
                'reinforcement_ratio': {
                    'calculation_method': '钢筋面积/构件面积',
                    'tolerance': '±0.08%',
                    'annotation_format': '百分比(%)'
                }
            }
        }
        
        # 保存标注指南
        with open('annotations/annotation_guidelines.json', 'w', encoding='utf-8') as f:
            json.dump(guidelines, f, ensure_ascii=False, indent=2)
        
        print("标注指南创建完成!")
        return guidelines
    
    def convert_pointcloud_to_images(self, point_cloud_file):
        """将点云转换为2D图像用于标注"""
        print(f"转换点云文件: {point_cloud_file}")
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(point_cloud_file)
        points = np.asarray(pcd.points)
        
        # 创建多个视角的投影图像
        projections = []
        
        # 前视图 (X-Y平面)
        front_view = self.create_2d_projection(points, 'front')
        projections.append(('front', front_view))
        
        # 侧视图 (X-Z平面)
        side_view = self.create_2d_projection(points, 'side')
        projections.append(('side', side_view))
        
        # 俯视图 (Y-Z平面)
        top_view = self.create_2d_projection(points, 'top')
        projections.append(('top', top_view))
        
        # 保存投影图像
        for view_name, image in projections:
            filename = f"images/processed/{os.path.splitext(os.path.basename(point_cloud_file))[0]}_{view_name}.png"
            cv2.imwrite(filename, image)
            print(f"保存{view_name}视图: {filename}")
        
        return projections
    
    def create_2d_projection(self, points, view_type):
        """创建2D投影图像"""
        if view_type == 'front':
            # X-Y平面投影
            x_coords = points[:, 0]
            y_coords = points[:, 1]
        elif view_type == 'side':
            # X-Z平面投影
            x_coords = points[:, 0]
            y_coords = points[:, 2]
        elif view_type == 'top':
            # Y-Z平面投影
            x_coords = points[:, 1]
            y_coords = points[:, 2]
        
        # 归一化坐标到图像尺寸
        img_size = 800
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 计算缩放比例
        scale_x = img_size / (x_max - x_min)
        scale_y = img_size / (y_max - y_min)
        scale = min(scale_x, scale_y)
        
        # 转换坐标
        img_x = ((x_coords - x_min) * scale).astype(int)
        img_y = ((y_coords - y_min) * scale).astype(int)
        
        # 创建图像
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # 绘制点
        for x, y in zip(img_x, img_y):
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        
        return image
    
    def create_labelme_annotation(self, image_path, annotations):
        """创建LabelMe格式的标注文件"""
        print(f"创建LabelMe标注: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 创建LabelMe格式的标注
        labelme_annotation = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }
        
        # 添加标注形状
        for annotation in annotations:
            shape = {
                "label": annotation['label'],
                "points": annotation['points'],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_annotation["shapes"].append(shape)
        
        # 保存标注文件
        annotation_path = image_path.replace('.png', '.json')
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_annotation, f, ensure_ascii=False, indent=2)
        
        print(f"标注文件保存: {annotation_path}")
        return annotation_path
    
    def create_pointcloud_annotation(self, point_cloud_file, annotations):
        """创建点云标注文件"""
        print(f"创建点云标注: {point_cloud_file}")
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(point_cloud_file)
        points = np.asarray(pcd.points)
        
        # 创建点云标注格式
        pointcloud_annotation = {
            "point_cloud_file": point_cloud_file,
            "total_points": len(points),
            "annotations": [],
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "annotation_tool": "PointCloud_Annotation",
                "version": "1.0"
            }
        }
        
        # 添加标注
        for annotation in annotations:
            pointcloud_annotation["annotations"].append({
                "category": annotation['category'],
                "points_indices": annotation['point_indices'],
                "properties": annotation['properties'],
                "bounding_box": annotation['bbox'],
                "confidence": annotation.get('confidence', 1.0)
            })
        
        # 保存标注文件
        annotation_path = point_cloud_file.replace('.ply', '_annotations.json')
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(pointcloud_annotation, f, ensure_ascii=False, indent=2)
        
        print(f"点云标注文件保存: {annotation_path}")
        return annotation_path
    
    def generate_training_data(self, annotations_dir):
        """生成训练数据"""
        print("生成训练数据...")
        
        training_data = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # 遍历标注文件
        for root, dirs, files in os.walk(annotations_dir):
            for file in files:
                if file.endswith('.json'):
                    annotation_path = os.path.join(root, file)
                    
                    # 读取标注文件
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    # 分配数据集
                    if 'train' in annotation_path:
                        training_data['train'].append(annotation_path)
                    elif 'val' in annotation_path:
                        training_data['val'].append(annotation_path)
                    elif 'test' in annotation_path:
                        training_data['test'].append(annotation_path)
                    else:
                        # 随机分配
                        import random
                        split = random.choice(['train', 'val', 'test'])
                        training_data[split].append(annotation_path)
        
        # 保存数据集配置
        dataset_config = {
            'train_count': len(training_data['train']),
            'val_count': len(training_data['val']),
            'test_count': len(training_data['test']),
            'total_count': sum(len(split) for split in training_data.values()),
            'categories': self.annotation_config['rebar_categories'],
            'inspection_items': self.annotation_config['inspection_items']
        }
        
        with open('datasets/dataset_config.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_config, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据生成完成:")
        print(f"- 训练集: {dataset_config['train_count']} 个样本")
        print(f"- 验证集: {dataset_config['val_count']} 个样本")
        print(f"- 测试集: {dataset_config['test_count']} 个样本")
        
        return training_data, dataset_config
    
    def run_annotation_pipeline(self, point_cloud_files):
        """运行完整标注流程"""
        print("开始钢筋检测标注流程...")
        
        # 1. 设置标注环境
        labelme_config = self.setup_labelme_environment()
        
        # 2. 创建标注指南
        guidelines = self.create_annotation_guidelines()
        
        # 3. 转换点云为图像
        all_projections = []
        for pcd_file in point_cloud_files:
            projections = self.convert_pointcloud_to_images(pcd_file)
            all_projections.extend(projections)
        
        # 4. 生成示例标注（这里需要人工标注）
        print("\n标注说明:")
        print("1. 使用LabelMe标注2D投影图像")
        print("2. 标注钢筋类型、位置、尺寸")
        print("3. 标注检测项目（直径、间距等）")
        print("4. 导出标注文件")
        
        # 5. 生成训练数据
        training_data, dataset_config = self.generate_training_data('annotations')
        
        return {
            'labelme_config': labelme_config,
            'guidelines': guidelines,
            'projections': all_projections,
            'training_data': training_data,
            'dataset_config': dataset_config
        }

# 使用示例
if __name__ == "__main__":
    # 创建标注系统
    annotation_system = RebarAnnotationSystem()
    
    # 示例点云文件列表
    point_cloud_files = [
        'example_point_cloud.ply',
        'rebar_sample_1.ply',
        'rebar_sample_2.ply'
    ]
    
    # 运行标注流程
    result = annotation_system.run_annotation_pipeline(point_cloud_files)
    
    print("\n标注系统设置完成!")
    print("下一步：使用LabelMe进行人工标注") 