# 钢筋检测项目

基于AI的智能钢筋检测系统，集成Stable Diffusion图像生成、YOLOv8目标检测、SAM分割等技术。

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-username/rebar-detection.git
cd rebar-detection
```

### 2. 下载模型文件
```bash
python download_models.py
```

### 3. 安装依赖
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install ultralytics
pip install segment-anything
pip install requests pillow tqdm
```

### 4. 运行项目
```bash
# 交互式钢筋图像生成
python 图像生成与AI模型/quick_start_diffusion_conda.py --mode rebar_interactive

# 生成单个钢筋类型
python 图像生成与AI模型/quick_start_diffusion_conda.py --mode single_rebar --rebar-type main_rebar --num-images 1
```

## 📁 项目结构

```
rebar-detection/
├── 🖼️ 图像生成与AI模型/          # Stable Diffusion和LoRA训练
│   ├── quick_start_diffusion_conda.py
│   ├── lora_training_prep.py
│   ├── lora_training_advanced.py
│   └── lora_training/ (完整目录)
│
├── 🔍 钢筋检测与识别/            # 核心检测算法
│   ├── rebar_inspection_system.py
│   ├── steel_inspection_pipeline.py
│   ├── defect_detection.py
│   └── rebar_annotation_system.py
│
├── 🚁 无人机与激光雷达/          # 激光雷达数据处理
│   ├── drone_lidar_inspection.py
│   ├── ground_lidar_dataset_system.py
│   ├── ground_lidar_positioning_system.py
│   └── point_cloud_preprocessing.py
│
├── 🏗️ BIM与工程管理/            # BIM模型比较
│   ├── bim_comparison.py
│   └── l2_m350_config.py
│
├── 📋 技术文档与方案/            # 项目文档
│   ├── 钢筋工程检测技术方案.md
│   ├── 无人机激光雷达钢筋检测方案.md
│   ├── 地面激光雷达数据集技术方案.md
│   ├── 钢视智检技术流程说明.md
│   ├── L2_M350钢筋检测技术方案.md
│   ├── 钢筋检测项目完整工作流程.md
│   ├── 项目文件结构说明.md
│   ├── 环境需求说明.md
│   ├── 项目文件分类整理.md
│   ├── 文件功能详细说明表.md
│   ├── 大文件上传指南.md
│   ├── rebar_dataset_paper_outline.md
│   ├── rebar_technology_paper_outline.md
│   └── 提取自国际赛道钢视智检商业计划书.docx
│
├── 🗂️ 模型与数据/              # 模型文件（需要下载）
│   ├── models/ (目录)
│   ├── outputs/ (目录)
│   ├── yolov8n.pt (6.5MB)
│   └── sam_vit_h_4b8939.pth (2.4GB)
│
├── 🔧 测试与验证/               # 测试脚本
│   ├── test_simple.py
│   └── gpu_test.py
│
├── download_models.py           # 模型下载脚本
├── .gitignore                  # Git忽略文件
└── README.md                   # 项目说明
```

## 📋 模型文件说明

- **yolov8n.pt** (6.5MB) - YOLOv8检测模型，用于钢筋目标检测
- **sam_vit_h_4b8939.pth** (2.4GB) - SAM分割模型，用于精确分割
- **Stable Diffusion模型** - 用于钢筋图像生成

## 🎯 主要功能

### 1. 图像生成
- 基于Stable Diffusion生成钢筋图像
- 支持多种钢筋类型（主筋、箍筋、分布筋等）
- 交互式生成和批量生成
- LoRA微调训练

### 2. 钢筋检测
- YOLOv8目标检测
- SAM精确分割
- 缺陷检测（裂纹、锈蚀等）
- 批量处理支持

### 3. 激光雷达处理
- 无人机激光雷达检测
- 地面激光雷达数据处理
- 点云预处理和配准
- 精确定位系统

### 4. BIM集成
- BIM模型与实际施工对比
- 偏差检测和分析
- 3D可视化

## 🔧 环境要求

- **Python**: 3.8+
- **GPU**: NVIDIA GPU (推荐8GB+显存)
- **内存**: 16GB+
- **存储**: 10GB+ 可用空间

## 📊 性能指标

- **检测精度**: >90%
- **处理速度**: 实时处理
- **图像生成**: 2-4分钟/张
- **支持分辨率**: 最高8K

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目主页: [GitHub Repository](https://github.com/your-username/rebar-detection)
- 问题反馈: [Issues](https://github.com/your-username/rebar-detection/issues)
- 邮箱: your-email@example.com

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [Meta AI](https://github.com/facebookresearch/segment-anything) - SAM
- [Hugging Face](https://huggingface.co/) - Stable Diffusion
- [Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库

---

⭐ 如果这个项目对你有帮助，请给它一个星标！ 