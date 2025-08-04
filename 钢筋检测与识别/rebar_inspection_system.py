#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钢筋工程专用检测系统
基于激光点云的高精度钢筋检测
"""

import open3d as o3d
import numpy as np
import cv2
from scipy.spatial import cKDTree
import json
from datetime import datetime

class RebarInspectionSystem:
    def __init__(self):
        # 钢筋检测标准参数
        self.rebar_standards = {
            'diameter_tolerance': 0.5,  # 直径容差(mm)
            'spacing_tolerance': 10,    # 间距容差(mm)
            'bending_angle': 135,       # 弯钩角度(度)
            'hook_length': 50,          # 弯钩长度(mm)
            'stirrup_spacing': 200,     # 箍筋间距(mm)
        }
        
        # 检测结果存储
        self.inspection_results = {}
        
    def detect_rebar_diameter(self, point_cloud):
        """检测钢筋直径"""
        print("检测钢筋直径...")
        
        # 1. 提取钢筋点云
        rebar_points = self.extract_rebar_points(point_cloud)
        
        # 2. 拟合圆柱体计算直径
        diameters = []
        for rebar_segment in rebar_points:
            diameter = self.fit_cylinder_diameter(rebar_segment)
            diameters.append(diameter)
        
        # 3. 检查是否符合标准
        standard_diameter = 12  # 假设标准直径12mm
        diameter_results = []
        
        for i, diameter in enumerate(diameters):
            deviation = abs(diameter - standard_diameter)
            is_qualified = deviation <= self.rebar_standards['diameter_tolerance']
            
            diameter_results.append({
                'rebar_id': i,
                'measured_diameter': diameter,
                'standard_diameter': standard_diameter,
                'deviation': deviation,
                'is_qualified': is_qualified
            })
        
        self.inspection_results['diameter'] = diameter_results
        return diameter_results
    
    def detect_rebar_spacing(self, point_cloud):
        """检测钢筋间距"""
        print("检测钢筋间距...")
        
        # 1. 识别钢筋轴线
        rebar_axes = self.extract_rebar_axes(point_cloud)
        
        # 2. 计算间距
        spacing_results = []
        for i in range(len(rebar_axes) - 1):
            spacing = self.calculate_spacing(rebar_axes[i], rebar_axes[i+1])
            
            # 检查间距是否符合标准
            standard_spacing = 150  # 假设标准间距150mm
            deviation = abs(spacing - standard_spacing)
            is_qualified = deviation <= self.rebar_standards['spacing_tolerance']
            
            spacing_results.append({
                'spacing_id': i,
                'measured_spacing': spacing,
                'standard_spacing': standard_spacing,
                'deviation': deviation,
                'is_qualified': is_qualified
            })
        
        self.inspection_results['spacing'] = spacing_results
        return spacing_results
    
    def detect_binding_quality(self, point_cloud):
        """检测绑扎质量"""
        print("检测绑扎质量...")
        
        # 1. 检测交叉点
        intersection_points = self.detect_intersections(point_cloud)
        
        # 2. 检查绑扎点
        binding_results = []
        for intersection in intersection_points:
            # 检查是否有绑扎痕迹
            has_binding = self.check_binding_mark(intersection, point_cloud)
            
            binding_results.append({
                'intersection_id': len(binding_results),
                'position': intersection,
                'has_binding': has_binding,
                'is_qualified': has_binding
            })
        
        self.inspection_results['binding'] = binding_results
        return binding_results
    
    def detect_hook_ends(self, point_cloud):
        """检测钢筋末端弯钩"""
        print("检测钢筋末端弯钩...")
        
        # 1. 识别钢筋末端
        rebar_ends = self.extract_rebar_ends(point_cloud)
        
        # 2. 检查弯钩
        hook_results = []
        for end in rebar_ends:
            # 检查是否有弯钩
            has_hook = self.check_hook_formation(end, point_cloud)
            
            # 检查弯钩角度
            hook_angle = self.measure_hook_angle(end, point_cloud)
            angle_qualified = abs(hook_angle - self.rebar_standards['bending_angle']) <= 5
            
            # 检查弯钩长度
            hook_length = self.measure_hook_length(end, point_cloud)
            length_qualified = hook_length >= self.rebar_standards['hook_length']
            
            hook_results.append({
                'end_id': len(hook_results),
                'position': end,
                'has_hook': has_hook,
                'hook_angle': hook_angle,
                'hook_length': hook_length,
                'angle_qualified': angle_qualified,
                'length_qualified': length_qualified,
                'is_qualified': has_hook and angle_qualified and length_qualified
            })
        
        self.inspection_results['hooks'] = hook_results
        return hook_results
    
    def detect_stirrups(self, point_cloud):
        """检测箍筋配置"""
        print("检测箍筋配置...")
        
        # 1. 识别箍筋
        stirrups = self.extract_stirrups(point_cloud)
        
        # 2. 检查箍筋间距
        stirrup_results = []
        for i, stirrup in enumerate(stirrups):
            # 计算箍筋间距
            if i > 0:
                spacing = self.calculate_stirrup_spacing(stirrups[i-1], stirrup)
                spacing_qualified = abs(spacing - self.rebar_standards['stirrup_spacing']) <= 20
            else:
                spacing = 0
                spacing_qualified = True
            
            # 检查箍筋形状
            shape_qualified = self.check_stirrup_shape(stirrup)
            
            stirrup_results.append({
                'stirrup_id': i,
                'position': stirrup,
                'spacing': spacing,
                'spacing_qualified': spacing_qualified,
                'shape_qualified': shape_qualified,
                'is_qualified': spacing_qualified and shape_qualified
            })
        
        self.inspection_results['stirrups'] = stirrup_results
        return stirrup_results
    
    def check_reinforcement_ratio(self, point_cloud):
        """检查配筋比例"""
        print("检查配筋比例...")
        
        # 1. 计算钢筋总面积
        total_rebar_area = self.calculate_total_rebar_area(point_cloud)
        
        # 2. 计算构件截面积
        component_area = self.calculate_component_area(point_cloud)
        
        # 3. 计算配筋率
        reinforcement_ratio = total_rebar_area / component_area * 100
        
        # 4. 检查是否符合规范
        min_ratio = 0.6  # 最小配筋率0.6%
        max_ratio = 2.5  # 最大配筋率2.5%
        
        is_qualified = min_ratio <= reinforcement_ratio <= max_ratio
        
        ratio_result = {
            'total_rebar_area': total_rebar_area,
            'component_area': component_area,
            'reinforcement_ratio': reinforcement_ratio,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'is_qualified': is_qualified
        }
        
        self.inspection_results['reinforcement_ratio'] = ratio_result
        return ratio_result
    
    def run_complete_inspection(self, point_cloud_file):
        """运行完整检测流程"""
        print("开始钢筋工程完整检测...")
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(point_cloud_file)
        
        # 执行各项检测
        diameter_results = self.detect_rebar_diameter(pcd)
        spacing_results = self.detect_rebar_spacing(pcd)
        binding_results = self.detect_binding_quality(pcd)
        hook_results = self.detect_hook_ends(pcd)
        stirrup_results = self.detect_stirrups(pcd)
        ratio_result = self.check_reinforcement_ratio(pcd)
        
        # 生成综合报告
        report = self.generate_inspection_report()
        
        return report
    
    def generate_inspection_report(self):
        """生成检测报告"""
        report = {
            'inspection_date': datetime.now().isoformat(),
            'summary': {
                'total_checks': 0,
                'qualified_items': 0,
                'unqualified_items': 0,
                'qualification_rate': 0.0
            },
            'detailed_results': self.inspection_results,
            'recommendations': []
        }
        
        # 统计结果
        total_checks = 0
        qualified_items = 0
        
        for category, results in self.inspection_results.items():
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
    
    # 辅助方法（简化实现）
    def extract_rebar_points(self, pcd):
        """提取钢筋点云"""
        # 简化实现：假设钢筋在特定高度范围内
        points = np.asarray(pcd.points)
        rebar_mask = (points[:, 2] > 0.5) & (points[:, 2] < 2.0)
        return points[rebar_mask]
    
    def fit_cylinder_diameter(self, points):
        """拟合圆柱体计算直径"""
        # 简化实现：使用点云密度估算直径
        return 12.0  # 假设检测到12mm直径
    
    def extract_rebar_axes(self, pcd):
        """提取钢筋轴线"""
        # 简化实现
        return [[0, 0, 1], [0, 0, 1.5]]  # 假设两条钢筋轴线
    
    def calculate_spacing(self, axis1, axis2):
        """计算间距"""
        return np.linalg.norm(np.array(axis1) - np.array(axis2))
    
    def detect_intersections(self, pcd):
        """检测交叉点"""
        # 简化实现
        return [[0, 0, 1], [0, 0.15, 1]]  # 假设交叉点位置
    
    def check_binding_mark(self, intersection, pcd):
        """检查绑扎痕迹"""
        # 简化实现：检查交叉点附近是否有额外点云
        return True  # 假设有绑扎
    
    def extract_rebar_ends(self, pcd):
        """提取钢筋末端"""
        # 简化实现
        return [[0, 0, 0.5], [0, 0, 2.5]]  # 假设末端位置
    
    def check_hook_formation(self, end, pcd):
        """检查弯钩形成"""
        return True  # 假设有弯钩
    
    def measure_hook_angle(self, end, pcd):
        """测量弯钩角度"""
        return 135.0  # 假设135度
    
    def measure_hook_length(self, end, pcd):
        """测量弯钩长度"""
        return 50.0  # 假设50mm
    
    def extract_stirrups(self, pcd):
        """提取箍筋"""
        # 简化实现
        return [[0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.5]]  # 假设箍筋位置
    
    def calculate_stirrup_spacing(self, stirrup1, stirrup2):
        """计算箍筋间距"""
        return np.linalg.norm(np.array(stirrup1) - np.array(stirrup2))
    
    def check_stirrup_shape(self, stirrup):
        """检查箍筋形状"""
        return True  # 假设形状正确
    
    def calculate_total_rebar_area(self, pcd):
        """计算钢筋总面积"""
        return 1000.0  # 假设1000mm²
    
    def calculate_component_area(self, pcd):
        """计算构件截面积"""
        return 50000.0  # 假设50000mm²

# 使用示例
if __name__ == "__main__":
    inspector = RebarInspectionSystem()
    
    # 运行检测
    report = inspector.run_complete_inspection("example_point_cloud.ply")
    
    # 打印报告
    print("\n" + "="*50)
    print("钢筋工程检测报告")
    print("="*50)
    print(f"检测时间: {report['inspection_date']}")
    print(f"总检查项目: {report['summary']['total_checks']}")
    print(f"合格项目: {report['summary']['qualified_items']}")
    print(f"不合格项目: {report['summary']['unqualified_items']}")
    print(f"合格率: {report['summary']['qualification_rate']:.1f}%")
    print("\n建议:")
    for rec in report['recommendations']:
        print(f"- {rec}") 