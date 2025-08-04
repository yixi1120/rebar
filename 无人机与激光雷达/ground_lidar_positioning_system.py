#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地面激光雷达定位系统
集成RTK和定位精度控制
"""

import numpy as np
import json
from datetime import datetime
import open3d as o3d

class GroundLidarPositioningSystem:
    def __init__(self):
        self.positioning_config = {
            'rtk_config': {
                'receiver_type': 'DJI RTK',
                'accuracy': 0.01,  # ±1cm
                'update_rate': 10,  # 10Hz
                'baseline_length': 5000,  # 5km基线
                'solution_type': 'RTK_FIXED'
            },
            'lidar_config': {
                'model': 'DJI L2',
                'range': 450,  # 450m
                'accuracy': 0.01,  # ±1cm
                'scan_frequency': 20,  # 20Hz
                'points_per_second': 240000
            },
            'positioning_requirements': {
                'absolute_accuracy': 0.01,  # ±1cm
                'relative_accuracy': 0.005,  # ±0.5cm
                'attitude_accuracy': 0.001,  # ±0.1°
                'coordinate_system': 'WGS84'
            }
        }
        
        self.control_points = []
        self.scan_positions = []
        self.positioning_results = {}
        
    def setup_rtk_system(self):
        """设置RTK系统"""
        print("设置RTK定位系统...")
        
        rtk_config = self.positioning_config['rtk_config']
        
        # 模拟RTK基站设置
        base_station = {
            'position': [116.3974, 39.9093, 50.0],  # 北京坐标
            'accuracy': rtk_config['accuracy'],
            'status': 'RTK_FIXED',
            'satellites': 12,
            'hdop': 0.8,
            'vdop': 1.2
        }
        
        # 模拟RTK移动站
        rover_station = {
            'position': [116.3975, 39.9094, 50.0],
            'accuracy': rtk_config['accuracy'],
            'status': 'RTK_FIXED',
            'satellites': 12,
            'hdop': 0.8,
            'vdop': 1.2,
            'baseline_length': rtk_config['baseline_length']
        }
        
        print(f"RTK基站位置: {base_station['position']}")
        print(f"RTK移动站位置: {rover_station['position']}")
        print(f"定位精度: ±{rtk_config['accuracy']*1000}mm")
        print(f"基线长度: {rtk_config['baseline_length']}m")
        
        return base_station, rover_station
    
    def setup_control_points(self, building_dimensions):
        """设置控制点"""
        print("设置控制点...")
        
        length, width, height = building_dimensions
        
        # 在建筑周围设置控制点
        control_points = [
            {'id': 'CP1', 'position': [0, 0, 0], 'type': 'primary'},
            {'id': 'CP2', 'position': [length, 0, 0], 'type': 'primary'},
            {'id': 'CP3', 'position': [length, width, 0], 'type': 'primary'},
            {'id': 'CP4', 'position': [0, width, 0], 'type': 'primary'},
            {'id': 'CP5', 'position': [length/2, width/2, 0], 'type': 'secondary'},
            {'id': 'CP6', 'position': [length/4, width/4, 0], 'type': 'secondary'},
            {'id': 'CP7', 'position': [3*length/4, 3*width/4, 0], 'type': 'secondary'}
        ]
        
        # 为每个控制点添加RTK测量
        for cp in control_points:
            cp['rtk_measurement'] = {
                'position': cp['position'],
                'accuracy': self.positioning_config['rtk_config']['accuracy'],
                'timestamp': datetime.now().isoformat(),
                'status': 'RTK_FIXED'
            }
        
        self.control_points = control_points
        
        print(f"设置了 {len(control_points)} 个控制点")
        for cp in control_points:
            print(f"- {cp['id']}: {cp['position']} ({cp['type']})")
        
        return control_points
    
    def plan_scan_positions(self, building_dimensions, control_points):
        """规划扫描位置"""
        print("规划激光雷达扫描位置...")
        
        length, width, height = building_dimensions
        
        # 基于控制点规划扫描位置
        scan_positions = []
        scan_spacing = 2  # 2米间距
        
        for x in range(0, int(length), scan_spacing):
            for y in range(0, int(width), scan_spacing):
                # 计算到最近控制点的距离
                min_distance = float('inf')
                nearest_cp = None
                
                for cp in control_points:
                    if cp['type'] == 'primary':
                        distance = np.sqrt((x - cp['position'][0])**2 + 
                                        (y - cp['position'][1])**2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_cp = cp
                
                # 添加扫描位置
                scan_position = {
                    'id': f'SP_{len(scan_positions)}',
                    'position': [x, y, 1.5],  # 1.5m扫描高度
                    'nearest_cp': nearest_cp['id'],
                    'distance_to_cp': min_distance,
                    'rtk_status': 'RTK_FIXED',
                    'estimated_accuracy': self.calculate_positioning_accuracy(min_distance)
                }
                scan_positions.append(scan_position)
        
        self.scan_positions = scan_positions
        
        print(f"规划了 {len(scan_positions)} 个扫描位置")
        print(f"平均定位精度: ±{np.mean([sp['estimated_accuracy'] for sp in scan_positions])*1000:.1f}mm")
        
        return scan_positions
    
    def calculate_positioning_accuracy(self, distance_to_cp):
        """计算定位精度"""
        # 基于距离控制点的距离计算定位精度
        base_accuracy = self.positioning_config['rtk_config']['accuracy']
        distance_factor = 1 + distance_to_cp / 1000  # 每1000m增加1倍误差
        
        return base_accuracy * distance_factor
    
    def simulate_scan_with_positioning(self, scan_position, building_dimensions):
        """模拟带定位精度的扫描"""
        print(f"模拟扫描位置 {scan_position['id']}...")
        
        # 获取RTK定位信息
        rtk_position = scan_position['position']
        rtk_accuracy = scan_position['estimated_accuracy']
        
        # 模拟定位误差
        positioning_error = np.random.normal(0, rtk_accuracy, 3)
        actual_position = np.array(rtk_position) + positioning_error
        
        # 生成点云数据（考虑定位精度）
        point_cloud = self.generate_positioned_pointcloud(actual_position, building_dimensions)
        
        # 记录定位结果
        positioning_result = {
            'scan_id': scan_position['id'],
            'rtk_position': rtk_position,
            'actual_position': actual_position.tolist(),
            'positioning_error': positioning_error.tolist(),
            'accuracy': rtk_accuracy,
            'timestamp': datetime.now().isoformat(),
            'point_cloud_size': len(point_cloud.points)
        }
        
        return point_cloud, positioning_result
    
    def generate_positioned_pointcloud(self, position, building_dimensions):
        """生成带定位信息的点云"""
        length, width, height = building_dimensions
        x, y, z = position
        
        # 生成钢筋点云（考虑定位精度）
        rebar_points = []
        
        # 主筋（纵向）
        for i in range(0, int(length), 150):
            for j in range(0, int(width), 150):
                for k in range(0, int(height), 20):
                    # 添加定位误差影响
                    positioning_noise = np.random.normal(0, 0.003, 3)  # 3mm噪声
                    point = [i + positioning_noise[0], 
                           j + positioning_noise[1], 
                           k + positioning_noise[2]]
                    rebar_points.append(point)
        
        # 箍筋（横向）
        for k in range(0, int(height), 200):
            for i in range(0, int(length), 30):
                for j in range(0, int(width), 30):
                    positioning_noise = np.random.normal(0, 0.003, 3)
                    point = [i + positioning_noise[0], 
                           j + positioning_noise[1], 
                           k + positioning_noise[2]]
                    rebar_points.append(point)
        
        # 转换为点云格式
        points = np.array(rebar_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def merge_positioned_scans(self, scan_results):
        """合并带定位信息的扫描数据"""
        print("合并带定位信息的扫描数据...")
        
        merged_points = []
        positioning_summary = {
            'total_scans': len(scan_results),
            'average_accuracy': 0,
            'max_positioning_error': 0,
            'coordinate_system': 'WGS84'
        }
        
        total_accuracy = 0
        max_error = 0
        
        for scan_result in scan_results:
            point_cloud, positioning_result = scan_result
            
            # 合并点云
            points = np.asarray(point_cloud.points)
            merged_points.extend(points)
            
            # 统计定位精度
            total_accuracy += positioning_result['accuracy']
            max_error = max(max_error, np.max(np.abs(positioning_result['positioning_error'])))
        
        # 计算平均精度
        positioning_summary['average_accuracy'] = total_accuracy / len(scan_results)
        positioning_summary['max_positioning_error'] = max_error
        
        # 创建合并后的点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.array(merged_points))
        
        print(f"合并完成，总点数: {len(merged_points)}")
        print(f"平均定位精度: ±{positioning_summary['average_accuracy']*1000:.1f}mm")
        print(f"最大定位误差: ±{positioning_summary['max_positioning_error']*1000:.1f}mm")
        
        return merged_pcd, positioning_summary
    
    def assess_positioning_quality(self, positioning_summary):
        """评估定位质量"""
        print("评估定位质量...")
        
        quality_assessment = {
            'absolute_accuracy': {
                'required': self.positioning_config['positioning_requirements']['absolute_accuracy'],
                'achieved': positioning_summary['average_accuracy'],
                'status': 'PASS' if positioning_summary['average_accuracy'] <= 
                         self.positioning_config['positioning_requirements']['absolute_accuracy'] else 'FAIL'
            },
            'relative_accuracy': {
                'required': self.positioning_config['positioning_requirements']['relative_accuracy'],
                'achieved': positioning_summary['max_positioning_error'],
                'status': 'PASS' if positioning_summary['max_positioning_error'] <= 
                         self.positioning_config['positioning_requirements']['relative_accuracy'] else 'FAIL'
            },
            'overall_quality': {
                'score': 0,
                'grade': 'A',
                'recommendations': []
            }
        }
        
        # 计算质量分数
        absolute_score = 100 * (1 - positioning_summary['average_accuracy'] / 
                               self.positioning_config['positioning_requirements']['absolute_accuracy'])
        relative_score = 100 * (1 - positioning_summary['max_positioning_error'] / 
                               self.positioning_config['positioning_requirements']['relative_accuracy'])
        
        overall_score = (absolute_score + relative_score) / 2
        quality_assessment['overall_quality']['score'] = overall_score
        
        # 确定等级
        if overall_score >= 90:
            quality_assessment['overall_quality']['grade'] = 'A'
        elif overall_score >= 80:
            quality_assessment['overall_quality']['grade'] = 'B'
        elif overall_score >= 70:
            quality_assessment['overall_quality']['grade'] = 'C'
        else:
            quality_assessment['overall_quality']['grade'] = 'D'
        
        # 添加建议
        if quality_assessment['absolute_accuracy']['status'] == 'FAIL':
            quality_assessment['overall_quality']['recommendations'].append(
                '增加RTK基站数量或缩短基线长度'
            )
        
        if quality_assessment['relative_accuracy']['status'] == 'FAIL':
            quality_assessment['overall_quality']['recommendations'].append(
                '增加控制点密度或使用全站仪辅助定位'
            )
        
        print(f"定位质量评估:")
        print(f"- 绝对精度: {quality_assessment['absolute_accuracy']['status']}")
        print(f"- 相对精度: {quality_assessment['relative_accuracy']['status']}")
        print(f"- 总体质量: {quality_assessment['overall_quality']['grade']} ({overall_score:.1f}分)")
        
        return quality_assessment
    
    def run_complete_positioning_pipeline(self, building_dimensions):
        """运行完整定位流程"""
        print("开始地面激光雷达定位系统...")
        
        # 1. 设置RTK系统
        base_station, rover_station = self.setup_rtk_system()
        
        # 2. 设置控制点
        control_points = self.setup_control_points(building_dimensions)
        
        # 3. 规划扫描位置
        scan_positions = self.plan_scan_positions(building_dimensions, control_points)
        
        # 4. 执行带定位的扫描
        scan_results = []
        for scan_position in scan_positions:
            point_cloud, positioning_result = self.simulate_scan_with_positioning(
                scan_position, building_dimensions
            )
            scan_results.append((point_cloud, positioning_result))
        
        # 5. 合并扫描数据
        merged_pcd, positioning_summary = self.merge_positioned_scans(scan_results)
        
        # 6. 评估定位质量
        quality_assessment = self.assess_positioning_quality(positioning_summary)
        
        return {
            'base_station': base_station,
            'rover_station': rover_station,
            'control_points': control_points,
            'scan_positions': scan_positions,
            'merged_pointcloud': merged_pcd,
            'positioning_summary': positioning_summary,
            'quality_assessment': quality_assessment
        }

# 使用示例
if __name__ == "__main__":
    # 创建定位系统
    positioning_system = GroundLidarPositioningSystem()
    
    # 设置建筑尺寸
    building_dimensions = (50, 30, 15)  # 长50m, 宽30m, 高15m
    
    # 运行完整定位流程
    results = positioning_system.run_complete_positioning_pipeline(building_dimensions)
    
    print("\n地面激光雷达定位系统完成!")
    print(f"RTK基站: {results['base_station']['position']}")
    print(f"控制点数量: {len(results['control_points'])}")
    print(f"扫描位置数量: {len(results['scan_positions'])}")
    print(f"定位质量等级: {results['quality_assessment']['overall_quality']['grade']}")
    
    # 可视化点云
    print("\n可视化定位点云...")
    o3d.visualization.draw_geometries([results['merged_pointcloud']], 
                                     window_name="带定位信息的点云") 