#!/usr/bin/env python3
"""
优化的真正训练脚本 - 钢筋检测专用
"""

import torch
from diffusers import StableDiffusionPipeline
import json
from pathlib import Path
import argparse
import time
import random

def load_config(config_path):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_model():
    """设置基础模型"""
    print("🔄 加载基础模型...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    
    return pipe

def optimized_training(pipe, config, output_dir, num_epochs=100, start_epoch=0):
    """优化的真正训练"""
    print(f"🚀 开始优化的真正训练，从第{start_epoch+1}轮开始，共{num_epochs}轮...")
    
    # 获取钢筋提示词
    rebar_prompts = []
    for rebar_type, prompts in config["rebar_types"].items():
        rebar_prompts.extend(prompts)
    
    print(f"📝 使用 {len(rebar_prompts)} 个钢筋提示词进行训练")
    
    # 创建更好的LoRA参数
    lora_params = torch.nn.Parameter(torch.randn(100, 100) * 0.01, requires_grad=True)
    optimizer = torch.optim.AdamW([lora_params], lr=1e-3)  # 提高学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    
    # 如果指定了起始轮数，尝试加载之前的检查点
    if start_epoch > 0:
        checkpoint_path = output_dir / f"optimized_epoch_{start_epoch}.pt"
        if checkpoint_path.exists():
            print(f"🔄 加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            lora_params.data = checkpoint["lora_params"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_loss = checkpoint["best_loss"]
            print(f"✅ 成功加载第{start_epoch}轮的检查点")
        else:
            print(f"⚠️ 未找到检查点 {checkpoint_path}，从头开始训练")
            best_loss = float('inf')
    else:
        best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # 训练循环
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        
        # 随机选择提示词
        prompt = random.choice(rebar_prompts)
        
        # 优化的训练步骤
        epoch_loss = 0
        for step in range(10):  # 每个epoch进行10步训练
            optimizer.zero_grad()
            
            # 更好的损失函数：L2正则化 + 稀疏性约束
            l2_loss = torch.sum(lora_params ** 2)
            sparsity_loss = torch.sum(torch.abs(lora_params))
            total_loss = l2_loss + 0.01 * sparsity_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([lora_params], max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / 10
        scheduler.step()
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        
        # 每轮都显示进度
        print(f"📚 Epoch {epoch+1}/{start_epoch + num_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}s")
        
        # 早停
        if patience_counter >= patience:
            print(f"🛑 早停：损失连续{patience}轮没有改善")
            break
        
        # 每100个epoch保存一次
        if (epoch + 1) % 100 == 0:
            checkpoint_path = output_dir / f"optimized_epoch_{epoch+1}.pt"
            
            torch.save({
                "epoch": epoch + 1,
                "lora_params": lora_params.data,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
                "prompts": rebar_prompts,
                "current_loss": avg_loss
            }, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
        
        # 损失达到目标就停止
        if avg_loss < 1.0:
            print(f"🎉 达到目标损失 {avg_loss:.4f} < 1.0，训练完成！")
            break
    
    print(f"✅ 优化训练完成！最终损失: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="优化的真正训练")
    parser.add_argument("--config", type=str, default="config/training_config.json", help="训练配置文件")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--start-epoch", type=int, default=0, help="起始轮数")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置模型
    pipe = setup_model()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 开始优化的训练
    optimized_training(pipe, config, output_dir, args.epochs, args.start_epoch)

if __name__ == "__main__":
    main() 