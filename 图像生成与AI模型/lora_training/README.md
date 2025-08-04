# 钢筋检测LoRA微调项目

## 项目结构
```
lora_training/
├── data/                    # 训练数据
│   ├── images/             # 图像文件
│   └── captions/           # 标注文件
├── config/                 # 配置文件
│   └── training_config.json
├── output/                 # 训练输出
├── logs/                   # 训练日志
├── train_lora.py          # 训练脚本
├── inference_lora.py      # 推理脚本
├── requirements.txt        # 依赖文件
└── README.md              # 说明文档
```

## 使用步骤

### 1. 环境准备
```bash
# 创建conda环境
conda create -n lora_training python=3.8
conda activate lora_training

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
- 收集真实的钢筋图像（建议50-100张）
- 将图像放入 `data/images/` 目录
- 为每张图像创建对应的标注文件

### 3. 开始训练
```bash
python train_lora.py --config config/training_config.json --epochs 100
```

### 4. 推理测试
```bash
python inference_lora.py --lora-path output/lora_checkpoint_epoch_100.safetensors --num-images 3
```

## 训练参数说明

- **学习率**: 1e-4
- **批次大小**: 1
- **分辨率**: 768x768
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **训练轮数**: 100

## 注意事项

1. 确保有足够的GPU内存（建议8GB+）
2. 训练时间较长，建议使用稳定的电源
3. 定期保存检查点，避免训练中断
4. 使用真实的钢筋图像进行训练效果最佳

## 预期效果

经过LoRA微调后，模型应该能够：
- 更准确地生成钢筋图像
- 避免生成布条、竹竿等错误内容
- 保持金属质感和真实感
- 支持多种钢筋类型（主筋、箍筋、分布筋等）
