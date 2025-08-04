#!/usr/bin/env python3
"""
简单测试脚本 - 验证优化后的参数
"""

import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import time

def test_optimized_params():
    """测试优化后的参数"""
    print("🔍 测试优化后的超高质量参数")
    print("=" * 50)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA不可用")
        return
    
    # 加载模型
    model_path = "models/CompVis/stable-diffusion-v1-4"
    print(f"🔄 加载模型: {model_path}")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        print("✅ 模型加载成功")
        
        # 测试优化后的参数
        prompt = "steel rebar with ribbed surface, concrete reinforcement, construction site, close up, detailed texture, metal surface"
        negative_prompt = "fabric, cloth, carpet, rug, wool, cotton, textile, soft, fluffy, fuzzy, pipe, tube, hollow, smooth surface, plastic, wood, glass, blurry, low quality, distorted, unrealistic, cartoon, painting, drawing, sketch, watermark, text, logo, signature, low resolution, pixelated, noise, artifacts, blur, out of focus"
        
        print("🎨 生成测试图像...")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print("参数: 50步, 8.0引导, 768x768分辨率")
        
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,    # 优化后的步数
            guidance_scale=8.0,        # 优化后的引导强度
            height=768,                # 优化后的分辨率
            width=768,                 # 优化后的分辨率
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        
        # 保存图像
        output_path = Path("outputs") / "test_optimized.png"
        output_path.parent.mkdir(exist_ok=True)
        image.save(output_path)
        
        print(f"✅ 测试完成！")
        print(f"生成时间: {generation_time:.2f}秒")
        print(f"图像保存: {output_path}")
        
        # 显示GPU使用情况
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_optimized_params() 