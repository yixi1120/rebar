#!/usr/bin/env python3
"""
LoRA推理脚本 - 钢筋检测专用
使用训练好的LoRA模型生成钢筋图像
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from pathlib import Path
import argparse

def load_lora_model(base_model_path, lora_path):
    """加载LoRA模型"""
    print("🔄 加载LoRA模型...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # 加载LoRA权重
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    
    return pipe

def generate_rebar_images(pipe, prompts, output_dir, num_images=1):
    """生成钢筋图像"""
    print("🎨 生成钢筋图像...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"生成图像 {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        image = pipe(
            prompt=prompt,
            negative_prompt="fabric, textile, bamboo, wood, straw, rope, plastic, cartoon, painting, blurry, lowres, artifacts, watermark, text, logo",
            num_inference_steps=40,
            guidance_scale=7.0,
            height=768,
            width=768,
            num_images_per_prompt=num_images
        ).images[0]
        
        output_path = output_dir / f"lora_rebar_{i+1:02d}.png"
        image.save(output_path)
        print(f"✅ 图像已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="LoRA推理")
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5", help="基础模型路径")
    parser.add_argument("--lora-path", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--output", type=str, default="lora_outputs", help="输出目录")
    parser.add_argument("--num-images", type=int, default=1, help="每类生成图像数量")
    
    args = parser.parse_args()
    
    # 加载模型
    pipe = load_lora_model(args.base_model, args.lora_path)
    
    # 钢筋提示词
    rebar_prompts = [
        "Ultra-realistic photo of construction site main rebar — tall vertical ribbed steel bars, metallic surface with rusty orange patches and silver highlights, hard reflections, densely packed in a reinforcement cage above fresh grey concrete slab, background blurred high-rise buildings, overcast daylight, photorealism, 8 k, high-contrast, shallow depth of field",
        "Ultra-realistic photo of steel stirrup rebar — rectangular ribbed steel bars with metallic surface, rusty orange patches and silver highlights, hard reflections, bent into rectangular shape, densely packed in a reinforcement cage, construction site, photorealism, 8 k, high-contrast",
        "Ultra-realistic photo of steel distribution rebar — horizontal ribbed steel bars with metallic surface, rusty orange patches and silver highlights, hard reflections, densely packed in a grid pattern, construction site, photorealism, 8 k, high-contrast"
    ]
    
    # 生成图像
    generate_rebar_images(pipe, rebar_prompts, args.output, args.num_images)

if __name__ == "__main__":
    main()
