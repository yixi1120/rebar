#!/usr/bin/env python3
"""
模型文件下载脚本
用于下载项目所需的大模型文件
"""

import os
import requests
from pathlib import Path
import sys

def download_model(url, filename, chunk_size=8192):
    """下载模型文件"""
    model_dir = Path("模型与数据")
    model_dir.mkdir(exist_ok=True)
    
    filepath = model_dir / filename
    
    if filepath.exists():
        print(f"✅ {filename} 已存在")
        return True
    
    print(f"📥 下载 {filename}...")
    print(f"🔗 下载地址: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 显示进度
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📊 进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        print(f"\n✅ {filename} 下载完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载 {filename} 失败: {e}")
        return False

def main():
    """下载所有模型文件"""
    print("🚀 开始下载模型文件...")
    
    # 模型文件配置
    models = {
        "yolov8n.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "description": "YOLOv8检测模型 (6.5MB)"
        },
        "sam_vit_h_4b8939.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "description": "SAM分割模型 (2.4GB)"
        }
    }
    
    success_count = 0
    total_count = len(models)
    
    for filename, config in models.items():
        print(f"\n📋 {filename} - {config['description']}")
        
        if download_model(config['url'], filename):
            success_count += 1
    
    print(f"\n📊 下载完成: {success_count}/{total_count} 个文件")
    
    if success_count == total_count:
        print("🎉 所有模型文件下载成功！")
    else:
        print("⚠️ 部分文件下载失败，请检查网络连接")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 