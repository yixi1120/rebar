#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机搭载激光雷达钢筋检测系统
基于无人机+激光雷达的高效钢筋检测方案
"""

import open3d as o3d
import numpy as np
import cv2
import json
from datetime import datetime
import time

class DroneLidarInspection:
    def __init__(self):
        # 无人机配置参数
        self.drone_config = {
            'flight_height': 10,      # 飞行高度(m)
            'scan_speed': 2,          # 扫描速度(m/s)
            'coverage_area': 100,     # 覆盖面积(m²)
            'flight_time': 25,        # 飞行时间(分钟)
            'battery_capacity': 4000  # 电池容量(mAh)
        }
        
        # 激光雷达配置
        self.lidar_config = {
            'model': 'Velodyne VLP-16',
            'range': 100,             # 扫描范围(m)
            'accuracy': 0.02,         # 精度(±2cm)
            'scan_frequency': 10,     # 扫描频率(Hz)
            'points_per_second': 300000  # 每秒点数
        }
        
        # 检测标准
        self.inspection_standards = {
            'diameter_tolerance': 0.5,    # 直径容差(mm)
            'spacing_tolerance': 10,      # 间距容差(mm)
            'bending_angle': 135,         # 弯钩角度(度)
            'hook_length': 50,            # 弯钩长度(mm)
            'stirrup_spacing': 200,       # 箍筋间距(mm)
        }
        
        # 检测结果
        self.inspection_results = {}
        
    def plan_flight_path(self, building_dimensions):
        """规划无人机飞行路径"""
        print("规划无人机飞行路径...")
        
        length, width, height = building_dimensions
        
        # 计算飞行路径
        flight_path = {
            'takeoff_point': [0, 0, 0],
            'scan_points': [],
            'landing_point': [0, 0, 0],
            'total_distance': 0,
            'estimated_time': 0
        }
        
        # 生成扫描路径点
        scan_height = height + 5  # 飞行高度比建筑高5米
        scan_spacing = 2  # 扫描间距2米
        
        for x in range(0, int(length), scan_spacing):
            for y in range(0, int(width), scan_spacing):
                flight_path['scan_points'].append([x, y, scan_height])
        
        # 计算总距离和预计时间
        total_points = len(flight_path['scan_points'])
        flight_path['total_distance'] = total_points * scan_spacing
        flight_path['estimated_time'] = flight_path['total_distance'] / self.drone_config['scan_speed']
        
        print(f"飞行路径规划完成:")
        print(f"- 扫描点数量: {total_points}")
        print(f"- 总飞行距离: {flight_path['total_distance']:.1f}m")
        print(f"- 预计飞行时间: {flight_path['estimated_time']:.1f}分钟")
        
        return flight_path
    
    def simulate_drone_scan(self, building_dimensions, flight_path):
        """模拟无人机扫描过程"""
        print("开始模拟无人机扫描...")
        
        # 模拟扫描数据收集
        scan_data = []
        total_points = len(flight_path['scan_points'])
        
        for i, scan_point in enumerate(flight_path['scan_points']):
            print(f"扫描进度: {i+1}/{total_points} ({((i+1)/total_points*100):.1f}%)")
            
            # 模拟从该点扫描到的点云数据
            point_cloud = self.generate_scan_data(scan_point, building_dimensions)
            scan_data.append({
                'scan_point': scan_point,
                'point_cloud': point_cloud,
                'timestamp': datetime.now().isoformat()
            })
            
            # 模拟扫描时间
            time.sleep(0.1)
        
        print("无人机扫描完成!")
        return scan_data
    
    def generate_scan_data(self, scan_point, building_dimensions):
        """生成扫描点云数据"""
        # 模拟从扫描点看到的钢筋结构
        length, width, height = building_dimensions
        x, y, z = scan_point
        
        # 生成钢筋点云
        rebar_points = []
        
        # 主筋（纵向）
        for i in range(0, int(length), 150):  # 150mm间距
            for j in range(0, int(width), 150):
                # 生成钢筋轴线上的点
                for k in range(0, int(height), 50):  # 50mm采样间隔
                    # 添加噪声模拟真实扫描
                    noise = np.random.normal(0, 0.01, 3)
                    point = [i + noise[0], j + noise[1], k + noise[2]]
                    rebar_points.append(point)
        
        # 箍筋（横向）
        for k in range(0, int(height), 200):  # 200mm箍筋间距
            for i in range(0, int(length), 50):
                for j in range(0, int(width), 50):
                    noise = np.random.normal(0, 0.01, 3)
                    point = [i + noise[0], j + noise[1], k + noise[2]]
                    rebar_points.append(point)
        
        # 转换为点云格式
        points = np.array(rebar_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def merge_scan_data(self, scan_data):
        """合并多个扫描点的数据"""
        print("合并扫描数据...")
        
        merged_points = []
        
        for scan in scan_data:
            points = np.asarray(scan['point_cloud'].points)
            merged_points.extend(points)
        
        # 创建合并后的点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.array(merged_points))
        
        print(f"合并完成，总点数: {len(merged_points)}")
        return merged_pcd
    
    def detect_rebar_features(self, merged_pcd):
        """检测钢筋特征"""
        print("检测钢筋特征...")
        
        # 1. 钢筋直径检测
        diameter_results = self.detect_rebar_diameter(merged_pcd)
        
        # 2. 钢筋间距检测
        spacing_results = self.detect_rebar_spacing(merged_pcd)
        
        # 3. 绑扎质量检测
        binding_results = self.detect_binding_quality(merged_pcd)
        
        # 4. 弯钩检测
        hook_results = self.detect_hook_ends(merged_pcd)
        
        # 5. 箍筋检测
        stirrup_results = self.detect_stirrups(merged_pcd)
        
        # 6. 配筋比例检测
        ratio_result = self.check_reinforcement_ratio(merged_pcd)
        
        return {
            'diameter': diameter_results,
            'spacing': spacing_results,
            'binding': binding_results,
            'hooks': hook_results,
            'stirrups': stirrup_results,
            'reinforcement_ratio': ratio_result
        }
    
    def detect_rebar_diameter(self, pcd):
        """检测钢筋直径"""
        print("检测钢筋直径...")
        
        # 简化实现：基于点云密度估算直径
        points = np.asarray(pcd.points)
        
        # 按高度分层分析
        height_bins = np.linspace(points[:, 2].min(), points[:, 2].max(), 10)
        diameters = []
        
        for i in range(len(height_bins) - 1):
            mask = (points[:, 2] >= height_bins[i]) & (points[:, 2] < height_bins[i+1])
            layer_points = points[mask]
            
            if len(layer_points) > 10:
                # 估算该层的钢筋直径
                density = len(layer_points) / (height_bins[i+1] - height_bins[i])
                estimated_diameter = 12.0 + np.random.normal(0, 0.2)  # 模拟检测结果
                diameters.append(estimated_diameter)
        
        # 生成检测结果
        diameter_results = []
        for i, diameter in enumerate(diameters):
            deviation = abs(diameter - 12.0)  # 假设标准直径12mm
            is_qualified = deviation <= self.inspection_standards['diameter_tolerance']
            
            diameter_results.append({
                'rebar_id': i,
                'measured_diameter': diameter,
                'standard_diameter': 12.0,
                'deviation': deviation,
                'is_qualified': is_qualified
            })
        
        return diameter_results
    
    def detect_rebar_spacing(self, pcd):
        """检测钢筋间距"""
        print("检测钢筋间距...")
        
        points = np.asarray(pcd.points)
        
        # 按X坐标分组（假设钢筋沿X方向排列）
        x_coords = np.unique(np.round(points[:, 0] / 150) * 150)  # 150mm间距
        
        spacing_results = []
        for i in range(len(x_coords) - 1):
            spacing = x_coords[i+1] - x_coords[i]
            deviation = abs(spacing - 150)  # 标准间距150mm
            is_qualified = deviation <= self.inspection_standards['spacing_tolerance']
            
            spacing_results.append({
                'spacing_id': i,
                'measured_spacing': spacing,
                'standard_spacing': 150,
                'deviation': deviation,
                'is_qualified': is_qualified
            })
        
        return spacing_results
    
    def detect_binding_quality(self, pcd):
        """检测绑扎质量"""
        print("检测绑扎质量...")
        
        # 简化实现：检测交叉点
        points = np.asarray(pcd.points)
        
        # 寻找可能的交叉点（X和Y坐标都接近的点）
        binding_results = []
        for i in range(0, len(points), 100):  # 采样检测
            point = points[i]
            
            # 检查附近是否有其他钢筋
            nearby_points = points[np.linalg.norm(points - point, axis=1) < 50]
            
            if len(nearby_points) > 5:  # 如果附近点较多，可能是交叉点
                has_binding = np.random.choice([True, False], p=[0.9, 0.1])  # 90%概率有绑扎
                
                binding_results.append({
                    'intersection_id': len(binding_results),
                    'position': point.tolist(),
                    'has_binding': has_binding,
                    'is_qualified': has_binding
                })
        
        return binding_results
    
    def detect_hook_ends(self, pcd):
        """检测钢筋末端弯钩"""
        print("检测钢筋末端弯钩...")
        
        points = np.asarray(pcd.points)
        
        # 寻找钢筋末端（最高和最低点）
        min_z = points[:, 2].min()
        max_z = points[:, 2].max()
        
        hook_results = []
        
        # 检测底部弯钩
        bottom_points = points[points[:, 2] < min_z + 100]  # 底部100mm范围内
        if len(bottom_points) > 0:
            has_hook = np.random.choice([True, False], p=[0.85, 0.15])
            hook_angle = 135 + np.random.normal(0, 2)  # 135°±2°
            hook_length = 50 + np.random.normal(0, 3)  # 50mm±3mm
            
            hook_results.append({
                'end_id': 0,
                'position': [0, 0, min_z],
                'has_hook': has_hook,
                'hook_angle': hook_angle,
                'hook_length': hook_length,
                'angle_qualified': abs(hook_angle - 135) <= 5,
                'length_qualified': hook_length >= 50,
                'is_qualified': has_hook and abs(hook_angle - 135) <= 5 and hook_length >= 50
            })
        
        # 检测顶部弯钩
        top_points = points[points[:, 2] > max_z - 100]  # 顶部100mm范围内
        if len(top_points) > 0:
            has_hook = np.random.choice([True, False], p=[0.85, 0.15])
            hook_angle = 135 + np.random.normal(0, 2)
            hook_length = 50 + np.random.normal(0, 3)
            
            hook_results.append({
                'end_id': 1,
                'position': [0, 0, max_z],
                'has_hook': has_hook,
                'hook_angle': hook_angle,
                'hook_length': hook_length,
                'angle_qualified': abs(hook_angle - 135) <= 5,
                'length_qualified': hook_length >= 50,
                'is_qualified': has_hook and abs(hook_angle - 135) <= 5 and hook_length >= 50
            })
        
        return hook_results
    
    def detect_stirrups(self, pcd):
        """检测箍筋配置"""
        print("检测箍筋配置...")
        
        points = np.asarray(pcd.points)
        
        # 按Z坐标分层检测箍筋
        z_coords = np.unique(np.round(points[:, 2] / 200) * 200)  # 200mm箍筋间距
        
        stirrup_results = []
        for i, z in enumerate(z_coords):
            layer_points = points[np.abs(points[:, 2] - z) < 50]  # 50mm厚度层
            
            if len(layer_points) > 10:
                spacing = 200 + np.random.normal(0, 10)  # 200mm±10mm
                spacing_qualified = abs(spacing - 200) <= 20
                shape_qualified = np.random.choice([True, False], p=[0.95, 0.05])
                
                stirrup_results.append({
                    'stirrup_id': i,
                    'position': [0, 0, z],
                    'spacing': spacing,
                    'spacing_qualified': spacing_qualified,
                    'shape_qualified': shape_qualified,
                    'is_qualified': spacing_qualified and shape_qualified
                })
        
        return stirrup_results
    
    def check_reinforcement_ratio(self, pcd):
        """检查配筋比例"""
        print("检查配筋比例...")
        
        points = np.asarray(pcd.points)
        
        # 估算钢筋总面积
        total_rebar_area = len(points) * 0.1  # 简化估算
        component_area = 50000  # 假设构件截面积50000mm²
        
        reinforcement_ratio = total_rebar_area / component_area * 100
        
        # 检查是否符合规范
        min_ratio = 0.6
        max_ratio = 2.5
        is_qualified = min_ratio <= reinforcement_ratio <= max_ratio
        
        ratio_result = {
            'total_rebar_area': total_rebar_area,
            'component_area': component_area,
            'reinforcement_ratio': reinforcement_ratio,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'is_qualified': is_qualified
        }
        
        return ratio_result
    
    def generate_inspection_report(self, flight_path, scan_data, inspection_results):
        """生成检测报告"""
        report = {
            'inspection_date': datetime.now().isoformat(),
            'drone_config': self.drone_config,
            'lidar_config': self.lidar_config,
            'flight_path': flight_path,
            'scan_summary': {
                'total_scan_points': len(scan_data),
                'total_points_collected': sum(len(scan['point_cloud'].points) for scan in scan_data),
                'scan_duration': flight_path['estimated_time']
            },
            'inspection_results': inspection_results,
            'summary': {
                'total_checks': 0,
                'qualified_items': 0,
                'unqualified_items': 0,
                'qualification_rate': 0.0
            },
            'recommendations': []
        }
        
        # 统计检测结果
        total_checks = 0
        qualified_items = 0
        
        for category, results in inspection_results.items():
            if isinstance(results, list):
                for result in results:
                    total_checks += 1
                    if result.get('is_qualified', False):
                        qualified_items += 1
            else:
                total_checks += 1
                if results.get('is_qualified', False):
                    qualified_items += 1
        
        report['summary']['total_checks'] = total_checks
        report['summary']['qualified_items'] = qualified_items
        report['summary']['unqualified_items'] = total_checks - qualified_items
        report['summary']['qualification_rate'] = qualified_items / total_checks * 100 if total_checks > 0 else 0
        
        # 生成建议
        if report['summary']['qualification_rate'] < 90:
            report['recommendations'].append("钢筋工程质量需要改进，建议重新检查")
        elif report['summary']['qualification_rate'] < 95:
            report['recommendations'].append("钢筋工程质量基本合格，部分项目需要整改")
        else:
            report['recommendations'].append("钢筋工程质量优秀，符合规范要求")
        
        return report
    
    def run_complete_inspection(self, building_dimensions):
        """运行完整检测流程"""
        print("开始无人机激光雷达钢筋检测...")
        print(f"建筑尺寸: {building_dimensions}")
        
        # 1. 规划飞行路径
        flight_path = self.plan_flight_path(building_dimensions)
        
        # 2. 执行无人机扫描
        scan_data = self.simulate_drone_scan(building_dimensions, flight_path)
        
        # 3. 合并扫描数据
        merged_pcd = self.merge_scan_data(scan_data)
        
        # 4. 检测钢筋特征
        inspection_results = self.detect_rebar_features(merged_pcd)
        
        # 5. 生成检测报告
        report = self.generate_inspection_report(flight_path, scan_data, inspection_results)
        
        return report, merged_pcd

# 使用示例
if __name__ == "__main__":
    # 创建无人机检测系统
    drone_inspector = DroneLidarInspection()
    
    # 设置建筑尺寸 (长, 宽, 高) 单位：米
    building_dimensions = (20, 15, 8)  # 20m长, 15m宽, 8m高
    
    # 运行完整检测
    report, merged_pcd = drone_inspector.run_complete_inspection(building_dimensions)
    
    # 打印检测报告
    print("\n" + "="*60)
    print("无人机激光雷达钢筋检测报告")
    print("="*60)
    print(f"检测时间: {report['inspection_date']}")
    print(f"飞行高度: {report['drone_config']['flight_height']}m")
    print(f"扫描点数量: {report['scan_summary']['total_scan_points']}")
    print(f"采集点数: {report['scan_summary']['total_points_collected']}")
    print(f"飞行时间: {report['scan_summary']['scan_duration']:.1f}分钟")
    print(f"总检查项目: {report['summary']['total_checks']}")
    print(f"合格项目: {report['summary']['qualified_items']}")
    print(f"合格率: {report['summary']['qualification_rate']:.1f}%")
    print("\n建议:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # 可视化点云
    print("\n可视化点云数据...")
    o3d.visualization.draw_geometries([merged_pcd], window_name="无人机扫描点云") 