import open3d as o3d
import numpy as np
import ifcopenshell
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

class BIMComparison:
    def __init__(self):
        self.standard_bim = None
        self.tolerance = 0.01  # 1cm容差
        self.defect_thresholds = {
            'crack': 0.005,  # 5mm裂缝阈值
            'corrosion': 0.01,  # 1cm腐蚀阈值
            'deformation': 0.02,  # 2cm变形阈值
            'weld_defect': 0.008,  # 8mm焊接缺陷阈值
        }
    
    def load_bim_model(self, ifc_file_path):
        """加载IFC格式的BIM模型"""
        try:
            ifc_file = ifcopenshell.open(ifc_file_path)
            print(f"成功加载BIM模型: {ifc_file_path}")
            self.standard_bim = ifc_file
            return True
        except Exception as e:
            print(f"加载BIM模型失败: {e}")
            return False
    
    def extract_bim_geometry(self):
        """从BIM模型中提取几何信息"""
        if not self.standard_bim:
            return None
        
        bim_points = []
        bim_elements = []
        
        # 遍历所有钢结构元素
        for entity in self.standard_bim.by_type('IfcStructuralMember'):
            if hasattr(entity, 'ObjectPlacement'):
                # 提取几何信息
                representation = entity.Representation
                if representation:
                    for item in representation.Representations:
                        if item.RepresentationType == 'Curve3D':
                            # 提取曲线几何
                            curve = item.Items[0]
                            # 这里需要根据具体IFC结构提取点坐标
                            # 简化示例
                            pass
        
        return bim_points, bim_elements
    
    def create_bim_point_cloud(self, bim_points):
        """创建BIM点云"""
        if not bim_points:
            return None
        
        bim_pcd = o3d.geometry.PointCloud()
        points = np.array(bim_points)
        bim_pcd.points = o3d.utility.Vector3dVector(points)
        
        # 设置BIM点云为绿色
        colors = np.ones((len(points), 3)) * [0, 1, 0]
        bim_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return bim_pcd
    
    def compare_with_bim(self, detected_pcd, bim_pcd):
        """将检测结果与BIM模型对比"""
        if bim_pcd is None:
            print("BIM模型未加载，跳过对比")
            return None
        
        # 配准点云
        transformation = self.register_point_clouds(detected_pcd, bim_pcd)
        
        # 计算偏差
        deviations = self.calculate_deviations(detected_pcd, bim_pcd, transformation)
        
        # 分析缺陷
        defect_analysis = self.analyze_defects(deviations)
        
        return {
            'transformation': transformation,
            'deviations': deviations,
            'defect_analysis': defect_analysis
        }
    
    def register_point_clouds(self, source_pcd, target_pcd):
        """点云配准"""
        # 使用ICP算法进行配准
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 
            max_correspondence_distance=self.tolerance,
            init=np.eye(4),
            criteria=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return icp_result.transformation
    
    def calculate_deviations(self, detected_pcd, bim_pcd, transformation):
        """计算偏差"""
        # 应用变换
        detected_pcd_transformed = detected_pcd.transform(transformation)
        
        # 计算点到点的距离
        distances = detected_pcd_transformed.compute_point_cloud_distance(bim_pcd)
        
        deviations = []
        for i, distance in enumerate(distances):
            if distance > self.tolerance:
                point = np.asarray(detected_pcd_transformed.points)[i]
                deviations.append({
                    'point': point,
                    'deviation': distance,
                    'threshold': self.tolerance
                })
        
        return deviations
    
    def analyze_defects(self, deviations):
        """分析缺陷类型和严重程度"""
        defect_analysis = {
            'total_defects': len(deviations),
            'defect_types': {},
            'severity_levels': {
                'low': 0,    # 0-1cm
                'medium': 0, # 1-2cm
                'high': 0    # >2cm
            }
        }
        
        for deviation in deviations:
            distance = deviation['deviation']
            
            # 判断缺陷类型（简化）
            if distance < 0.01:
                defect_type = 'surface_damage'
            elif distance < 0.02:
                defect_type = 'deformation'
            else:
                defect_type = 'structural_defect'
            
            # 统计缺陷类型
            if defect_type not in defect_analysis['defect_types']:
                defect_analysis['defect_types'][defect_type] = 0
            defect_analysis['defect_types'][defect_type] += 1
            
            # 判断严重程度
            if distance < 0.01:
                defect_analysis['severity_levels']['low'] += 1
            elif distance < 0.02:
                defect_analysis['severity_levels']['medium'] += 1
            else:
                defect_analysis['severity_levels']['high'] += 1
        
        return defect_analysis
    
    def generate_report(self, comparison_result, output_path="inspection_report.json"):
        """生成检测报告"""
        if not comparison_result:
            return
        
        report = {
            'inspection_date': datetime.now().isoformat(),
            'summary': {
                'total_defects': comparison_result['defect_analysis']['total_defects'],
                'defect_types': comparison_result['defect_analysis']['defect_types'],
                'severity_distribution': comparison_result['defect_analysis']['severity_levels']
            },
            'recommendations': self.generate_recommendations(comparison_result['defect_analysis']),
            'technical_details': {
                'tolerance_used': self.tolerance,
                'registration_accuracy': 'ICP配准',
                'analysis_method': '点云对比分析'
            }
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"检测报告已保存: {output_path}")
        return report
    
    def generate_recommendations(self, defect_analysis):
        """生成维修建议"""
        recommendations = []
        
        total_defects = defect_analysis['total_defects']
        high_severity = defect_analysis['severity_levels']['high']
        
        if total_defects == 0:
            recommendations.append("钢结构状态良好，无需维修")
        elif high_severity > 0:
            recommendations.append("发现严重缺陷，建议立即维修")
            recommendations.append("建议进行详细的结构安全评估")
        elif defect_analysis['severity_levels']['medium'] > 5:
            recommendations.append("发现多个中等缺陷，建议计划维修")
        else:
            recommendations.append("发现轻微缺陷，建议定期监测")
        
        return recommendations
    
    def visualize_comparison(self, detected_pcd, bim_pcd, comparison_result):
        """可视化对比结果"""
        geometries = []
        
        # 添加检测到的点云（红色）
        if detected_pcd:
            detected_colors = np.asarray(detected_pcd.colors)
            detected_colors[:] = [1, 0, 0]  # 红色
            detected_pcd.colors = o3d.utility.Vector3dVector(detected_colors)
            geometries.append(detected_pcd)
        
        # 添加BIM模型点云（绿色）
        if bim_pcd:
            geometries.append(bim_pcd)
        
        # 添加偏差点（黄色）
        if comparison_result and comparison_result['deviations']:
            deviation_points = np.array([d['point'] for d in comparison_result['deviations']])
            deviation_pcd = o3d.geometry.PointCloud()
            deviation_pcd.points = o3d.utility.Vector3dVector(deviation_points)
            deviation_colors = np.ones((len(deviation_points), 3)) * [1, 1, 0]  # 黄色
            deviation_pcd.colors = o3d.utility.Vector3dVector(deviation_colors)
            geometries.append(deviation_pcd)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="BIM对比分析结果",
            point_show_normal=True
        )

# 使用示例
if __name__ == "__main__":
    # 加载检测结果
    from defect_detection import SteelDefectDetector
    from point_cloud_preprocessing import PointCloudPreprocessor
    
    # 预处理
    preprocessor = PointCloudPreprocessor()
    steel_cloud, _ = preprocessor.preprocess_pipeline("example_point_cloud.ply")
    
    # 缺陷检测
    detector = SteelDefectDetector()
    defect_pcd, _ = detector.detect_defects_pipeline(steel_cloud)
    
    # BIM对比
    bim_comparison = BIMComparison()
    
    # 如果有BIM模型文件
    # bim_comparison.load_bim_model("steel_structure.ifc")
    # bim_points, _ = bim_comparison.extract_bim_geometry()
    # bim_pcd = bim_comparison.create_bim_point_cloud(bim_points)
    
    # 对比分析（这里用检测结果作为示例）
    comparison_result = bim_comparison.compare_with_bim(defect_pcd, steel_cloud)
    
    # 生成报告
    if comparison_result:
        report = bim_comparison.generate_report(comparison_result)
        print("检测报告:", report)
        
        # 可视化
        bim_comparison.visualize_comparison(defect_pcd, steel_cloud, comparison_result) 