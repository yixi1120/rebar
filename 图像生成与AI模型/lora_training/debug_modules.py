#!/usr/bin/env python3
"""
调试脚本：查看UNet的模块名称
"""

import torch
from diffusers import StableDiffusionPipeline

def debug_unet_modules():
    """查看UNet的模块名称"""
    print("🔄 加载基础模型...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    print("🔍 查看UNet模块名称...")
    attention_modules = []
    linear_modules = []
    
    for name, module in pipe.unet.named_modules():
        if "attn" in name:
            attention_modules.append(name)
            print(f"注意力模块: {name}")
            if hasattr(module, "to_q"):
                print(f"  - 有 to_q: {type(module.to_q)}")
            if hasattr(module, "to_k"):
                print(f"  - 有 to_k: {type(module.to_k)}")
            if hasattr(module, "to_v"):
                print(f"  - 有 to_v: {type(module.to_v)}")
            if hasattr(module, "to_out"):
                print(f"  - 有 to_out: {type(module.to_out)}")
        
        if isinstance(module, torch.nn.Linear):
            linear_modules.append(name)
            if len(linear_modules) <= 10:  # 只显示前10个
                print(f"线性层: {name} -> {module}")
    
    print(f"\n总共找到 {len(attention_modules)} 个注意力模块")
    print(f"总共找到 {len(linear_modules)} 个线性层")

if __name__ == "__main__":
    debug_unet_modules() 