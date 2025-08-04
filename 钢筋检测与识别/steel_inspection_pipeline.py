#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钢视智检完整技术流程
基于激光点云的高精度钢结构检测系统
"""

import os
import sys
import time
import logging
from datetime import datetime
import json
import gradio as gr

# 导入自定义模块
from point_cloud_preprocessing import PointCloudPreprocessor
from defect_detection import SteelDefectDetector
from bim_comparison import BIMComparison

class SteelInspectionPipeline:
    def __init__(self):
        # 设置日志
        self.setup_logging()
        
        # 初始化各个模块
        self.preprocessor = PointCloudPreprocessor()
        self.detector = SteelDefectDetector()
        self.bim_comparison = BIMComparison()
        
        # 结果存储
        self.results = {}
        
        self.logger.info("钢视智检系统初始化完成")
    
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('steel_inspection.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self, point_cloud_file, bim_file=None):
        """运行完整的检测流程"""
        start_time = time.time()
        self.logger.info("开始钢视智检完整流程")
        
        try:
            # 第一阶段：点云预处理
            self.logger.info("=== 第一阶段：点云预处理 ===")
            steel_cloud, ground_cloud = self.preprocessor.preprocess_pipeline(point_cloud_file)
            self.results['preprocessed_cloud'] = steel_cloud
            
            # 第二阶段：缺陷检测
            self.logger.info("=== 第二阶段：缺陷检测 ===")
            defect_pcd, segmented_defects = self.detector.detect_defects_pipeline(steel_cloud)
            self.results['defect_cloud'] = defect_pcd
            self.results['segmented_defects'] = segmented_defects
            
            # 第三阶段：BIM对比分析
            self.logger.info("=== 第三阶段：BIM对比分析 ===")
            comparison_result = None
            if bim_file and os.path.exists(bim_file):
                self.bim_comparison.load_bim_model(bim_file)
                bim_points, _ = self.bim_comparison.extract_bim_geometry()
                bim_pcd = self.bim_comparison.create_bim_point_cloud(bim_points)
                comparison_result = self.bim_comparison.compare_with_bim(defect_pcd, bim_pcd)
            else:
                # 如果没有BIM文件，使用检测结果作为对比基准
                comparison_result = self.bim_comparison.compare_with_bim(defect_pcd, steel_cloud)
            
            self.results['comparison_result'] = comparison_result
            
            # 第四阶段：生成报告
            self.logger.info("=== 第四阶段：生成报告 ===")
            report = self.bim_comparison.generate_report(comparison_result)
            self.results['report'] = report
            
            # 计算处理时间
            processing_time = time.time() - start_time
            self.logger.info(f"完整流程处理完成，耗时: {processing_time:.2f}秒")
            
            return True, report
            
        except Exception as e:
            self.logger.error(f"流程执行失败: {e}")
            return False, str(e)
    
    def visualize_results(self):
        """可视化所有结果"""
        if not self.results:
            self.logger.warning("没有可用的结果进行可视化")
            return
        
        try:
            # 可视化预处理结果
            if 'preprocessed_cloud' in self.results:
                o3d.visualization.draw_geometries(
                    [self.results['preprocessed_cloud']],
                    window_name="预处理后的钢结构点云"
                )
            
            # 可视化缺陷检测结果
            if 'defect_cloud' in self.results and self.results['defect_cloud']:
                o3d.visualization.draw_geometries(
                    [self.results['preprocessed_cloud'], self.results['defect_cloud']],
                    window_name="缺陷检测结果"
                )
            
            # 可视化BIM对比结果
            if 'comparison_result' in self.results:
                self.bim_comparison.visualize_comparison(
                    self.results.get('defect_cloud'),
                    self.results.get('preprocessed_cloud'),
                    self.results['comparison_result']
                )
                
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")
    
    def export_results(self, output_dir="results"):
        """导出所有结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出点云文件
        if 'preprocessed_cloud' in self.results:
            o3d.io.write_point_cloud(
                f"{output_dir}/preprocessed_cloud_{timestamp}.ply",
                self.results['preprocessed_cloud']
            )
        
        if 'defect_cloud' in self.results and self.results['defect_cloud']:
            o3d.io.write_point_cloud(
                f"{output_dir}/defect_cloud_{timestamp}.ply",
                self.results['defect_cloud']
            )
        
        # 导出报告
        if 'report' in self.results:
            with open(f"{output_dir}/inspection_report_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(self.results['report'], f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已导出到: {output_dir}")
    
    def create_gradio_interface(self):
        """创建Gradio Web界面"""
        def process_inspection(point_cloud_file, bim_file=None):
            success, result = self.run_full_pipeline(point_cloud_file.name, bim_file.name if bim_file else None)
            
            if success:
                # 导出结果
                self.export_results()
                
                # 返回报告摘要
                if isinstance(result, dict):
                    summary = f"""
## 检测报告摘要

**检测时间**: {result.get('inspection_date', 'N/A')}

**缺陷统计**:
- 总缺陷数: {result.get('summary', {}).get('total_defects', 0)}
- 缺陷类型: {result.get('summary', {}).get('defect_types', {})}
- 严重程度分布: {result.get('summary', {}).get('severity_distribution', {})}

**维修建议**:
{chr(10).join(result.get('recommendations', []))}

**技术参数**:
- 容差设置: {result.get('technical_details', {}).get('tolerance_used', 'N/A')}
- 配准方法: {result.get('technical_details', {}).get('registration_accuracy', 'N/A')}
- 分析方法: {result.get('technical_details', {}).get('analysis_method', 'N/A')}
                    """
                    return summary, "✅ 检测完成"
                else:
                    return str(result), "✅ 检测完成"
            else:
                return f"❌ 检测失败: {result}", "❌ 检测失败"
        
        # 创建界面
        iface = gr.Interface(
            fn=process_inspection,
            inputs=[
                gr.File(label="上传点云文件 (.ply, .pcd, .xyz)"),
                gr.File(label="上传BIM模型文件 (.ifc) - 可选")
            ],
            outputs=[
                gr.Markdown(label="检测报告"),
                gr.Textbox(label="处理状态")
            ],
            title="钢视智检系统",
            description="""
            ## 基于激光点云的高精度钢结构检测系统
            
            ### 功能特点:
            - 🎯 毫米级精度检测
            - 🔍 多类型缺陷识别
            - 📊 BIM模型对比分析
            - 📋 自动生成检测报告
            - 🎨 3D可视化结果
            
            ### 使用说明:
            1. 上传激光点云文件
            2. 可选择上传BIM模型文件
            3. 点击提交开始检测
            4. 查看检测报告和可视化结果
            """,
            theme=gr.themes.Soft()
        )
        
        return iface

def main():
    """主函数"""
    print("=" * 60)
    print("钢视智检系统 - 基于激光点云的高精度钢结构检测")
    print("=" * 60)
    
    # 创建检测流程实例
    pipeline = SteelInspectionPipeline()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            # 启动Web界面
            print("启动Web界面...")
            iface = pipeline.create_gradio_interface()
            iface.launch(share=True, server_name="0.0.0.0", server_port=7860)
        elif sys.argv[1] == "--demo":
            # 运行演示
            print("运行演示模式...")
            success, result = pipeline.run_full_pipeline("example_point_cloud.ply")
            if success:
                print("演示完成!")
                pipeline.visualize_results()
                pipeline.export_results()
            else:
                print(f"演示失败: {result}")
        else:
            # 处理指定文件
            point_cloud_file = sys.argv[1]
            bim_file = sys.argv[2] if len(sys.argv) > 2 else None
            
            print(f"处理文件: {point_cloud_file}")
            success, result = pipeline.run_full_pipeline(point_cloud_file, bim_file)
            
            if success:
                print("检测完成!")
                pipeline.visualize_results()
                pipeline.export_results()
            else:
                print(f"检测失败: {result}")
    else:
        # 交互模式
        print("\n请选择运行模式:")
        print("1. Web界面模式 (--web)")
        print("2. 演示模式 (--demo)")
        print("3. 文件处理模式 (文件路径)")
        print("\n示例:")
        print("python steel_inspection_pipeline.py --web")
        print("python steel_inspection_pipeline.py --demo")
        print("python steel_inspection_pipeline.py your_point_cloud.ply")

if __name__ == "__main__":
    main() 