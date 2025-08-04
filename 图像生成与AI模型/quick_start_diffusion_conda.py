#!/usr/bin/env python3
"""
Diffusion 快速启动脚本 - Conda版本
用于快速生成图像和钢筋检测相关图像
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class DiffusionQuickStart:
    def __init__(self):
        self.model_path = "models/CompVis/stable-diffusion-v1-4"
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """加载Stable Diffusion模型"""
        print("🔄 正在加载模型...")
        
        # 检查模型文件是否存在
        if not Path(self.model_path).exists():
            print(f"❌ 模型路径不存在: {self.model_path}")
            print("请先下载模型文件到正确位置")
            return None
            
        try:
            # 记录加载开始时间
            start_time = time.time()
            
            # 检查GPU初始状态
            if torch.cuda.is_available():
                print(f"🔍 GPU初始状态:")
                print(f"  - GPU名称: {torch.cuda.get_device_name()}")
                print(f"  - GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"  - GPU内存已用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                print(f"  - GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
            
            # 加载模型（diffusers内部会显示组件加载进度）
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            
            # 检查CUDA可用性
            if torch.cuda.is_available():
                print("✅ 使用CUDA加速")
                pipe = pipe.to("cuda")
                # 启用内存优化
                pipe.enable_attention_slicing()
                
                # 检查加载后的GPU状态
                print(f"🔍 模型加载后GPU状态:")
                print(f"  - GPU内存已用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                print(f"  - GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
            else:
                print("⚠️ 使用CPU模式（较慢）")
            
            # 计算加载时间
            load_time = time.time() - start_time
            print(f"✅ 模型加载成功！(耗时: {load_time:.2f}秒)")
            return pipe
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def generate_image(self, pipe, prompt, output_name="generated_image.png"):
        """生成单张图像"""
        print(f"🎨 生成图像: {prompt}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 检查生成前的GPU状态
            if torch.cuda.is_available():
                print(f"🔍 生成前GPU内存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # 生成图像（优化参数提升质量）
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,  # 增加步数提升质量
                guidance_scale=7.5,      # 增加引导强度
                height=512,              # 设置高度
                width=512,               # 设置宽度
                num_images_per_prompt=1  # 每次生成1张
            ).images[0]
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            # 检查生成后的GPU状态
            if torch.cuda.is_available():
                print(f"🔍 生成后GPU内存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # 保存图像
            output_path = self.output_dir / output_name
            image.save(output_path)
            print(f"✅ 图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
            
            return image
            
        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            return None
    
    def show_prompt_translation(self):
        """显示提示词的中文翻译"""
        print("🔍 当前钢筋提示词的中文翻译：")
        print("=" * 60)
        
        # 翻译当前使用的提示词
        translations = [
            "专业建筑照片，混凝土中嵌入的钢主筋，带肋表面的纵向钢筋，高分辨率，详细纹理，建筑工地，主筋，逼真，8K质量",
            "混凝土柱中垂直钢主筋的特写视图，带可见肋纹理的结构钢，建筑细节，主筋，专业摄影，高质量，详细",
            "混凝土梁中水平钢主筋，带清晰肋表面的结构加固，建筑工地，主筋，逼真，详细纹理，专业照片",
            
            "专业建筑照片，围绕主筋的钢箍筋，带清晰肋纹理的横向加固，混凝土柱，建筑细节，箍筋，特写，逼真，8K质量",
            "纵向钢筋周围钢箍筋的详细视图，带肋表面的横向钢加固，混凝土结构，箍筋，专业摄影，高分辨率",
            "矩形钢箍筋，带肋纹理的横向绑扎，混凝土梁加固，建筑，箍筋，逼真，详细，专业照片",
            
            "专业建筑照片，钢分布筋，带肋表面的次要加固，混凝土板，建筑细节，分布筋，逼真，详细纹理，8K质量",
            "混凝土板中水平钢分布筋的特写，带清晰肋纹理的次要钢加固，建筑，分布筋，专业摄影，高质量",
            "钢分布筋网格，带肋表面的次要钢筋，混凝土地板，建筑，分布筋，逼真，详细，专业照片",
            
            "专业建筑照片，弯曲钢加固筋，带肋纹理的弯曲钢筋，混凝土梁端，建筑细节，弯筋，逼真，详细视图，8K质量",
            "梁柱连接处弯曲钢筋的详细视图，带肋表面的弯曲加固，混凝土结构，弯筋，建筑，专业摄影，高质量",
            "混凝土拐角处的弯曲钢加固，带清晰肋纹理的弯曲钢筋，结构细节，弯筋，逼真，详细，专业照片",
            
            "专业建筑照片，带弯钩的钢加固筋，带肋纹理的钢筋钩，混凝土加固，建筑细节，弯钩，特写，逼真，8K质量",
            "带90度弯钩的钢筋详细视图，带肋表面的加固弯钩端，混凝土结构，弯钩，专业摄影，高分辨率",
            "梁端带弯钩的钢加固，带肋纹理的钢筋终止，混凝土建筑，弯钩，逼真，详细，专业照片",
            
            "专业建筑照片，用绑扎铁丝绑扎的钢加固筋，带肋纹理的绑扎钢筋，混凝土加固，绑扎铁丝，建筑细节，逼真，8K质量",
            "用绑扎铁丝固定的钢筋特写，带肋表面的加固绑扎，混凝土结构，绑扎铁丝，专业摄影，详细纹理",
            "用绑扎铁丝绑扎的钢加固网格，带肋纹理的绑扎钢网格，混凝土建筑，绑扎铁丝，逼真，详细，专业照片",
            
            "专业建筑照片，钢加固筋交叉点，带肋纹理的交叉钢筋，混凝土加固，交叉点，建筑细节，逼真，8K质量",
            "钢筋交叉点详细视图，带肋表面的加固交叉，混凝土结构，交叉点，专业摄影，高分辨率",
            "钢加固网格交叉点，带肋纹理的交叉钢筋，混凝土建筑，交叉点，逼真，详细，专业照片"
        ]
        
        for i, translation in enumerate(translations, 1):
            print(f"{i:2d}. {translation}")
        
        print("\n❌ 问题分析：")
        print("- 提示词过于复杂，AI可能误解为钢管")
        print("- 'steel bars' 可能被理解为钢管而不是钢筋")
        print("- 缺少明确的'钢筋'特征描述")
        print("- 需要更强调'ribbed surface'（肋纹表面）")
        
        return translations
    
    def generate_single_rebar_type(self, pipe, rebar_type, num_images=3):
        """生成单个钢筋类型"""
        print(f"🏗️ 生成 {num_images} 张 {rebar_type} 钢筋图像...")
        
        # 按钢筋类型分类的提示词 - 基于真实钢筋图像重写
        rebar_type_prompts = {
            "main_rebar": [
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people"
            ],
            "stirrup": [
                "rectangular steel stirrup rebar with ribbed surface, rusty brown color, construction site, bent rebar, detailed metal texture",
                "steel rebar stirrup ties, rectangular shape, ribbed surface, rusty color, construction reinforcement, detailed view",
                "bent steel rebar stirrup, cylindrical with ridges, rusty brown, construction site, detailed metal texture"
            ],
            "distribution_rebar": [
                "steel distribution rebar with ribbed surface, rusty brown color, construction grid, individual rebar bars, detailed texture",
                "horizontal rebar mesh, cylindrical steel bars with ridges, rusty surface, construction site, detailed metal texture",
                "steel rebar grid, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed view"
            ],
            "bent_rebar": [
                "bent steel rebar with ribbed surface, rusty brown color, curved cylindrical bars, construction site, detailed metal texture",
                "curved rebar with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view",
                "steel rebar bend, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture"
            ],
            "hook_end": [
                "steel rebar with hook end, ribbed surface, rusty brown color, cylindrical bars, construction site, detailed metal texture",
                "rebar hook with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view",
                "steel rebar termination with hook, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture"
            ],
            "binding_wire": [
                "steel rebar tied with wire, ribbed surface, rusty brown color, cylindrical bars, construction grid, detailed metal texture",
                "rebar binding wire, cylindrical steel bars with ridges, rusty surface, construction site, detailed view",
                "steel rebar mesh with wire ties, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture"
            ],
            "intersection": [
                "steel rebar intersection, ribbed surface, rusty brown color, cylindrical bars crossing, construction site, detailed metal texture",
                "crossing rebar with ridges, cylindrical steel bars, rusty surface, construction grid, detailed view",
                "steel rebar crossing point, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture"
            ]
        }
        
        if rebar_type not in rebar_type_prompts:
            print(f"❌ 未知的钢筋类型: {rebar_type}")
            print("可用的钢筋类型: main_rebar, stirrup, distribution_rebar, bent_rebar, hook_end, binding_wire, intersection")
            return []
        
        prompts = rebar_type_prompts[rebar_type]
        
        # 负面提示词 - 避免生成混凝土结构、管道等
        negative_prompt = "watermark, logo, text, blurry, lowres, overexposed, artifacts, lens flare, deformation, duplicate limbs, CGI, cartoon, illustration, unrealistic colors, pipes, formwork panels, scaffolding, workers, stirrup highlighted, binding wire emphasized"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(prompts))), desc=f"生成{rebar_type}钢筋"):
            prompt = prompts[i]
            output_name = f"{rebar_type}_{i+1:02d}.png"
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 生成高质量图像 - 提升参数
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,    # 增加步数
                    guidance_scale=8.5,        # 提高引导强度
                    height=768,                # 提高分辨率
                    width=768,                 # 提高分辨率
                    num_images_per_prompt=1
                ).images[0]
                
                # 计算生成时间
                generation_time = time.time() - start_time
                
                # 保存图像
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ {rebar_type}钢筋图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ {rebar_type}钢筋图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张 {rebar_type} 钢筋图像")
        return generated_images
    
    def generate_ultra_quality_rebar(self, pipe, rebar_type, num_images=2):
        """生成超高质量钢筋图像"""
        print(f"🏗️ 生成 {num_images} 张超高质量 {rebar_type} 钢筋图像...")
        
        # 超高质量钢筋提示词 - 更明确、更直接
        ultra_quality_prompts = {
            "main_rebar": [
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people"
            ],
            "stirrup": [
                "steel stirrup rebar with ribbed surface, concrete column reinforcement, rectangular shape, metal texture",
                "rectangular stirrup rebar, concrete beam reinforcement, ribbed steel, detailed metal surface",
                "steel rebar stirrup ties, concrete structure, metal texture, sharp focus, detailed view"
            ],
            "distribution_rebar": [
                "steel distribution rebar with ribbed surface, concrete slab reinforcement, metal texture, detailed view",
                "horizontal rebar mesh, concrete floor reinforcement, ribbed steel bars, metal surface, sharp focus",
                "steel rebar grid, concrete construction, distribution rebar, detailed metal texture, professional photo"
            ],
            "bent_rebar": [
                "bent steel rebar with ribbed surface, concrete beam reinforcement, curved metal, detailed texture",
                "curved rebar with ridges, concrete column connection, metal surface, sharp focus, detailed view",
                "steel rebar bend, concrete structure, ribbed texture, metal detail, professional photo"
            ],
            "hook_end": [
                "steel rebar with hook end, ribbed surface, concrete reinforcement, metal texture, detailed view",
                "rebar hook with ridges, concrete beam end, metal surface, sharp focus, detailed texture",
                "steel rebar termination with hook, concrete construction, metal detail, professional photo"
            ],
            "binding_wire": [
                "steel rebar tied with wire, ribbed surface, concrete reinforcement, metal texture, detailed view",
                "rebar binding wire, concrete structure, metal surface, sharp focus, detailed texture",
                "steel rebar mesh with wire ties, concrete construction, metal detail, professional photo"
            ],
            "intersection": [
                "steel rebar intersection, ribbed surface, concrete reinforcement, metal texture, detailed view",
                "crossing rebar with ridges, concrete structure, metal surface, sharp focus, detailed texture",
                "steel rebar crossing point, concrete construction, metal detail, professional photo"
            ]
        }
        
        if rebar_type not in ultra_quality_prompts:
            print(f"❌ 未知的钢筋类型: {rebar_type}")
            print("可用的钢筋类型: main_rebar, stirrup, distribution_rebar, bent_rebar, hook_end, binding_wire, intersection")
            return []
        
        prompts = ultra_quality_prompts[rebar_type]
        
        # 负面提示词 - 更严格的质量控制，避免生成毛绒绒的东西
        negative_prompt = "watermark, logo, text, blurry, lowres, overexposed, artifacts, lens flare, deformation, duplicate limbs, CGI, cartoon, illustration, unrealistic colors, pipes, formwork panels, scaffolding, workers, stirrup highlighted, binding wire emphasized"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(prompts))), desc=f"生成超高质量{rebar_type}钢筋"):
            prompt = prompts[i]
            output_name = f"ultra_{rebar_type}_{i+1:02d}.png"
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 生成超高质量图像 - 优化参数，减少时间
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,    # 减少步数，从100降到50
                    guidance_scale=8.0,        # 稍微降低引导强度
                    height=768,                # 降低分辨率，从1024降到768
                    width=768,                 # 降低分辨率，从1024降到768
                    num_images_per_prompt=1
                ).images[0]
                
                # 计算生成时间
                generation_time = time.time() - start_time
                
                # 保存图像
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ 超高质量{rebar_type}钢筋图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ 超高质量{rebar_type}钢筋图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张超高质量 {rebar_type} 钢筋图像")
        return generated_images
    
    def interactive_rebar_generator(self, pipe):
        """交互式钢筋生成器"""
        print("\n🏗️ 交互式钢筋生成器")
        print("=" * 50)
        print("可用的钢筋类型:")
        print("1. main_rebar (主筋)")
        print("2. stirrup (箍筋)")
        print("3. distribution_rebar (分布筋)")
        print("4. bent_rebar (弯筋)")
        print("5. hook_end (弯钩)")
        print("6. binding_wire (绑扎铁丝)")
        print("7. intersection (交叉点)")
        print("8. all (生成所有类型)")
        print("9. quality_mode (质量模式选择)")
        print("10. quit (退出)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🎨 请选择钢筋类型 (1-10): ").strip()
                
                if user_input.lower() == 'quit' or user_input == '10':
                    print("👋 退出钢筋生成器")
                    break
                elif user_input == '8' or user_input.lower() == 'all':
                    print("🏗️ 生成所有钢筋类型...")
                    for rebar_type in ["main_rebar", "stirrup", "distribution_rebar", "bent_rebar", "hook_end", "binding_wire", "intersection"]:
                        print(f"\n--- 生成 {rebar_type} ---")
                        self.generate_single_rebar_type(pipe, rebar_type, 2)
                    print("✅ 所有钢筋类型生成完成！")
                    break
                elif user_input == '9' or user_input.lower() == 'quality_mode':
                    self.quality_mode_selection(pipe)
                elif user_input in ['1', '2', '3', '4', '5', '6', '7']:
                    # 映射数字到钢筋类型
                    type_mapping = {
                        '1': 'main_rebar',
                        '2': 'stirrup', 
                        '3': 'distribution_rebar',
                        '4': 'bent_rebar',
                        '5': 'hook_end',
                        '6': 'binding_wire',
                        '7': 'intersection'
                    }
                    
                    rebar_type = type_mapping[user_input]
                    num_images = input(f"生成几张 {rebar_type} 图像? (默认3张): ").strip()
                    num_images = int(num_images) if num_images.isdigit() else 3
                    
                    self.generate_single_rebar_type(pipe, rebar_type, num_images)
                else:
                    print("❌ 无效选择，请输入1-10")
                    
            except KeyboardInterrupt:
                print("\n👋 退出钢筋生成器")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def quality_mode_selection(self, pipe):
        """质量模式选择"""
        print("\n🎯 质量模式选择")
        print("=" * 40)
        print("1. 标准质量 (快速生成，512x512)")
        print("2. 高质量 (中等速度，768x768)")
        print("3. 超高质量 (较慢速度，768x768，更多步数)")
        print("4. 返回主菜单")
        print("=" * 40)
        
        while True:
            try:
                quality_input = input("\n🎨 请选择质量模式 (1-4): ").strip()
                
                if quality_input == '4':
                    print("返回主菜单")
                    break
                elif quality_input in ['1', '2', '3']:
                    print("\n🏗️ 请选择钢筋类型:")
                    print("1. main_rebar (主筋)")
                    print("2. stirrup (箍筋)")
                    print("3. distribution_rebar (分布筋)")
                    print("4. bent_rebar (弯筋)")
                    print("5. hook_end (弯钩)")
                    print("6. binding_wire (绑扎铁丝)")
                    print("7. intersection (交叉点)")
                    print("8. all (生成所有类型)")
                    
                    type_input = input("请选择钢筋类型 (1-8): ").strip()
                    
                    if type_input == '8' or type_input.lower() == 'all':
                        rebar_types = ["main_rebar", "stirrup", "distribution_rebar", "bent_rebar", "hook_end", "binding_wire", "intersection"]
                        num_images = input("每种类型生成几张图像? (默认2张): ").strip()
                        num_images = int(num_images) if num_images.isdigit() else 2
                        
                        for rebar_type in rebar_types:
                            print(f"\n--- 生成 {rebar_type} ---")
                            if quality_input == '1':
                                self.generate_standard_quality_rebar(pipe, rebar_type, num_images)
                            elif quality_input == '2':
                                self.generate_high_quality_rebar(pipe, rebar_type, num_images)
                            elif quality_input == '3':
                                self.generate_ultra_quality_rebar(pipe, rebar_type, num_images)
                        
                        print("✅ 所有钢筋类型生成完成！")
                        break
                    elif type_input in ['1', '2', '3', '4', '5', '6', '7']:
                        type_mapping = {
                            '1': 'main_rebar',
                            '2': 'stirrup', 
                            '3': 'distribution_rebar',
                            '4': 'bent_rebar',
                            '5': 'hook_end',
                            '6': 'binding_wire',
                            '7': 'intersection'
                        }
                        
                        rebar_type = type_mapping[type_input]
                        num_images = input(f"生成几张 {rebar_type} 图像? (默认3张): ").strip()
                        num_images = int(num_images) if num_images.isdigit() else 3
                        
                        if quality_input == '1':
                            self.generate_standard_quality_rebar(pipe, rebar_type, num_images)
                        elif quality_input == '2':
                            self.generate_high_quality_rebar(pipe, rebar_type, num_images)
                        elif quality_input == '3':
                            self.generate_ultra_quality_rebar(pipe, rebar_type, num_images)
                        break
                    else:
                        print("❌ 无效选择")
                else:
                    print("❌ 无效选择，请输入1-4")
                    
            except KeyboardInterrupt:
                print("\n返回主菜单")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def generate_standard_quality_rebar(self, pipe, rebar_type, num_images=3):
        """生成标准质量钢筋图像"""
        print(f"🏗️ 生成 {num_images} 张标准质量 {rebar_type} 钢筋图像...")
        
        # 标准质量钢筋提示词
        standard_prompts = {
            "main_rebar": [
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people"
            ],
            "stirrup": [
                "rectangular steel stirrup rebar with ribbed surface, rusty brown color, construction site",
                "steel rebar stirrup ties, rectangular shape, ribbed surface, rusty color",
                "bent steel rebar stirrup, cylindrical with ridges, rusty brown, construction site"
            ],
            "distribution_rebar": [
                "steel distribution rebar with ribbed surface, rusty brown color, construction grid",
                "horizontal rebar mesh, cylindrical steel bars with ridges, rusty surface",
                "steel rebar grid, cylindrical bars with ribbed surface, rusty brown color"
            ],
            "bent_rebar": [
                "bent steel rebar with ribbed surface, rusty brown color, curved cylindrical bars",
                "curved rebar with ridges, cylindrical steel bars, rusty surface",
                "steel rebar bend, cylindrical with ribbed surface, rusty brown color"
            ],
            "hook_end": [
                "steel rebar with hook end, ribbed surface, rusty brown color, cylindrical bars",
                "rebar hook with ridges, cylindrical steel bars, rusty surface",
                "steel rebar termination with hook, cylindrical with ribbed surface, rusty brown color"
            ],
            "binding_wire": [
                "steel rebar tied with wire, ribbed surface, rusty brown color, cylindrical bars",
                "rebar binding wire, cylindrical steel bars with ridges, rusty surface",
                "steel rebar mesh with wire ties, cylindrical bars with ribbed surface, rusty brown color"
            ],
            "intersection": [
                "steel rebar intersection, ribbed surface, rusty brown color, cylindrical bars crossing",
                "crossing rebar with ridges, cylindrical steel bars, rusty surface",
                "steel rebar crossing point, cylindrical bars with ribbed surface, rusty brown color"
            ]
        }
        
        if rebar_type not in standard_prompts:
            print(f"❌ 未知的钢筋类型: {rebar_type}")
            return []
        
        prompts = standard_prompts[rebar_type]
        negative_prompt = "concrete structure, pipe, tube, hollow, smooth surface, plastic, wood, glass, blurry, low quality"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(prompts))), desc=f"生成标准质量{rebar_type}钢筋"):
            prompt = prompts[i]
            output_name = f"standard_{rebar_type}_{i+1:02d}.png"
            
            try:
                start_time = time.time()
                
                # 标准质量参数
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    num_images_per_prompt=1
                ).images[0]
                
                generation_time = time.time() - start_time
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ 标准质量{rebar_type}钢筋图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ 标准质量{rebar_type}钢筋图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张标准质量 {rebar_type} 钢筋图像")
        return generated_images
    
    def generate_high_quality_rebar(self, pipe, rebar_type, num_images=3):
        """生成高质量钢筋图像"""
        print(f"🏗️ 生成 {num_images} 张高质量 {rebar_type} 钢筋图像...")
        
        # 高质量钢筋提示词
        high_quality_prompts = {
            "main_rebar": [
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
                "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people"
            ],
            "stirrup": [
                "rectangular steel stirrup rebar with ribbed surface, rusty brown color, construction site, bent rebar, detailed metal texture, high resolution",
                "steel rebar stirrup ties, rectangular shape, ribbed surface, rusty color, construction reinforcement, detailed view, professional photography",
                "bent steel rebar stirrup, cylindrical with ridges, rusty brown, construction site, detailed metal texture, sharp focus"
            ],
            "distribution_rebar": [
                "steel distribution rebar with ribbed surface, rusty brown color, construction grid, individual rebar bars, detailed texture, high resolution",
                "horizontal rebar mesh, cylindrical steel bars with ridges, rusty surface, construction site, detailed metal texture, professional photography",
                "steel rebar grid, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed view, sharp focus"
            ],
            "bent_rebar": [
                "bent steel rebar with ribbed surface, rusty brown color, curved cylindrical bars, construction site, detailed metal texture, high resolution",
                "curved rebar with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view, professional photography",
                "steel rebar bend, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture, sharp focus"
            ],
            "hook_end": [
                "steel rebar with hook end, ribbed surface, rusty brown color, cylindrical bars, construction site, detailed metal texture, high resolution",
                "rebar hook with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view, professional photography",
                "steel rebar termination with hook, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture, sharp focus"
            ],
            "binding_wire": [
                "steel rebar tied with wire, ribbed surface, rusty brown color, cylindrical bars, construction grid, detailed metal texture, high resolution",
                "rebar binding wire, cylindrical steel bars with ridges, rusty surface, construction site, detailed view, professional photography",
                "steel rebar mesh with wire ties, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture, sharp focus"
            ],
            "intersection": [
                "steel rebar intersection, ribbed surface, rusty brown color, cylindrical bars crossing, construction site, detailed metal texture, high resolution",
                "crossing rebar with ridges, cylindrical steel bars, rusty surface, construction grid, detailed view, professional photography",
                "steel rebar crossing point, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture, sharp focus"
            ]
        }
        
        if rebar_type not in high_quality_prompts:
            print(f"❌ 未知的钢筋类型: {rebar_type}")
            return []
        
        prompts = high_quality_prompts[rebar_type]
        negative_prompt = "concrete structure, pipe, tube, hollow, smooth surface, plastic, wood, glass, blurry, low quality, distorted, unrealistic, cartoon, painting, drawing, sketch, watermark, text, logo, signature, low resolution, pixelated, noise, artifacts"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(prompts))), desc=f"生成高质量{rebar_type}钢筋"):
            prompt = prompts[i]
            output_name = f"hq_{rebar_type}_{i+1:02d}.png"
            
            try:
                start_time = time.time()
                
                # 高质量参数
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=8.5,
                    height=768,
                    width=768,
                    num_images_per_prompt=1
                ).images[0]
                
                generation_time = time.time() - start_time
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ 高质量{rebar_type}钢筋图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ 高质量{rebar_type}钢筋图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张高质量 {rebar_type} 钢筋图像")
        return generated_images
    
    def generate_simple_rebar_images(self, pipe, num_images=5):
        """生成简单明确的钢筋图像"""
        print(f"🏗️ 生成 {num_images} 张简单钢筋图像...")
        
        # 简化的钢筋提示词 - 基于真实钢筋图像重写
        simple_rebar_prompts = [
            # 主筋 - 简化版本
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            
            # 箍筋 - 简化版本
            "rectangular steel stirrup rebar with ribbed surface, rusty brown color, construction site",
            "steel rebar stirrup ties, rectangular shape, ribbed surface, rusty color",
            "bent steel rebar stirrup, cylindrical with ridges, rusty brown, construction site",
            
            # 分布筋 - 简化版本
            "steel distribution rebar with ribbed surface, rusty brown color, construction grid",
            "horizontal rebar mesh, cylindrical steel bars with ridges, rusty surface",
            "steel rebar grid, cylindrical bars with ribbed surface, rusty brown color",
            
            # 弯筋 - 简化版本
            "bent steel rebar with ribbed surface, rusty brown color, curved cylindrical bars",
            "curved rebar with ridges, cylindrical steel bars, rusty surface",
            "steel rebar bend, cylindrical with ribbed surface, rusty brown color",
            
            # 弯钩 - 简化版本
            "steel rebar with hook end, ribbed surface, rusty brown color, cylindrical bars",
            "rebar hook with ridges, cylindrical steel bars, rusty surface",
            "steel rebar termination with hook, cylindrical with ribbed surface, rusty brown color",
            
            # 绑扎铁丝 - 简化版本
            "steel rebar tied with wire, ribbed surface, rusty brown color, cylindrical bars",
            "rebar binding wire, cylindrical steel bars with ridges, rusty surface",
            "steel rebar mesh with wire ties, cylindrical bars with ribbed surface, rusty brown color",
            
            # 交叉点 - 简化版本
            "steel rebar intersection, ribbed surface, rusty brown color, cylindrical bars crossing",
            "crossing rebar with ridges, cylindrical steel bars, rusty surface",
            "steel rebar crossing point, cylindrical bars with ribbed surface, rusty brown color"
        ]
        
        # 负面提示词 - 避免生成混凝土结构、管道等
        negative_prompt = "watermark, logo, text, blurry, lowres, overexposed, artifacts, lens flare, deformation, duplicate limbs, CGI, cartoon, illustration, unrealistic colors, pipes, formwork panels, scaffolding, workers, stirrup highlighted, binding wire emphasized"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(simple_rebar_prompts))), desc="生成简单钢筋图像"):
            prompt = simple_rebar_prompts[i]
            output_name = f"simple_rebar_{i+1:02d}.png"
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 生成图像
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    num_images_per_prompt=1
                ).images[0]
                
                # 计算生成时间
                generation_time = time.time() - start_time
                
                # 保存图像
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ 简单钢筋图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ 简单钢筋图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张简单钢筋图像")
        return generated_images
    
    def generate_rebar_images(self, pipe, num_images=5):
        """生成钢筋检测相关图像"""
        print(f"🏗️ 生成 {num_images} 张钢筋检测图像...")
        
        # 优化后的钢筋提示词 - 更专业、更详细
        rebar_prompts = [
            # 主筋 (main_rebar) - 更详细的描述
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            
            # 箍筋 (stirrup) - 更精确的描述
            "professional construction photo, steel stirrup reinforcement around main bars, transverse reinforcement with clear ribbed texture, concrete column, construction detail, stirrup, close up, photorealistic, 8k quality",
            "detailed view of steel stirrup ties around longitudinal bars, transverse steel reinforcement with ribbed surface, concrete structure, stirrup, professional photography, high resolution",
            "rectangular steel stirrup reinforcement, transverse ties with ribbed texture, concrete beam reinforcement, construction, stirrup, photorealistic, detailed, professional photo",
            
            # 分布筋 (distribution_rebar) - 更专业的描述
            "professional construction photo, steel distribution reinforcement bars, secondary reinforcement with ribbed surface, concrete slab, construction detail, distribution_rebar, photorealistic, detailed texture, 8k quality",
            "close-up of horizontal steel distribution bars in concrete slab, secondary steel reinforcement with clear ribbed texture, construction, distribution_rebar, professional photography, high quality",
            "steel distribution reinforcement mesh, secondary steel bars with ribbed surface, concrete floor, construction, distribution_rebar, photorealistic, detailed, professional photo",
            
            # 弯筋 (bent_rebar) - 更详细的描述
            "professional construction photo, steel bent reinforcement bars, curved steel bars with ribbed texture, concrete beam end, construction detail, bent_rebar, photorealistic, detailed view, 8k quality",
            "detailed view of steel bent bars at beam-column connection, curved reinforcement with ribbed surface, concrete structure, bent_rebar, construction, professional photography, high quality",
            "steel bent reinforcement at concrete corner, curved steel bars with clear ribbed texture, structural detail, bent_rebar, photorealistic, detailed, professional photo",
            
            # 弯钩 (hook_end) - 更精确的描述
            "professional construction photo, steel reinforcement bars with hook ends, steel bar hooks with ribbed texture, concrete reinforcement, construction detail, hook_end, close up, photorealistic, 8k quality",
            "detailed view of steel bars with 90-degree hooks, reinforcement hook ends with ribbed surface, concrete structure, hook_end, professional photography, high resolution",
            "steel reinforcement with hook ends at beam ends, steel bar termination with ribbed texture, concrete construction, hook_end, photorealistic, detailed, professional photo",
            
            # 绑扎铁丝 (binding_wire) - 更详细的描述
            "professional construction photo, steel reinforcement bars tied with binding wire, tied steel bars with ribbed texture, concrete reinforcement, binding_wire, construction detail, photorealistic, 8k quality",
            "close-up of steel bars secured with binding wire, reinforcement ties with ribbed surface, concrete structure, binding_wire, professional photography, detailed texture",
            "steel reinforcement mesh tied with binding wire, tied steel grid with ribbed texture, concrete construction, binding_wire, photorealistic, detailed, professional photo",
            
            # 交叉点 (intersection) - 更专业的描述
            "professional construction photo, steel reinforcement bars intersection, crossing steel bars with ribbed texture, concrete reinforcement, intersection, construction detail, photorealistic, 8k quality",
            "detailed view of steel bars crossing at intersection point, reinforcement intersection with ribbed surface, concrete structure, intersection, professional photography, high resolution",
            "steel reinforcement grid intersection, crossing steel bars with ribbed texture, concrete construction, intersection, photorealistic, detailed, professional photo"
        ]
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(rebar_prompts))), desc="生成钢筋图像"):
            prompt = rebar_prompts[i]
            output_name = f"rebar_image_{i+1:02d}.png"
            
            image = self.generate_image(pipe, prompt, output_name)
            if image:
                generated_images.append(image)
        
        print(f"✅ 生成了 {len(generated_images)} 张钢筋检测图像")
        return generated_images
    
    def generate_high_quality_rebar_images(self, pipe, num_images=5):
        """生成高质量钢筋检测图像"""
        print(f"🏗️ 生成 {num_images} 张高质量钢筋检测图像...")
        
        # 高质量钢筋提示词 - 基于真实钢筋图像重写
        high_quality_prompts = [
            # 主筋高质量版本
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            "Ultra-realistic photograph of main rebar (主筋) — tall, vertical, ribbed HRB400 steel bars Ø25 mm, slightly rust-colored brown–orange, densely arrayed in a column cage, bound at regular 200 mm intervals by smaller transverse stirrups; bars protrude upward 1–2 m above a freshly cast grey concrete slab; background shows high-rise building façade and protective steel mesh; soft overcast daylight, industrial work-site atmosphere, high-contrast sharp focus, depth-of-field bokeh, 8 K resolution, photorealism, no people",
            
            # 箍筋高质量版本
            "rectangular steel stirrup rebar with ribbed surface, rusty brown color, construction site, bent rebar, detailed metal texture, high resolution",
            "steel rebar stirrup ties, rectangular shape, ribbed surface, rusty color, construction reinforcement, detailed view, professional photography",
            "bent steel rebar stirrup, cylindrical with ridges, rusty brown, construction site, detailed metal texture, ultra sharp focus",
            
            # 分布筋高质量版本
            "steel distribution rebar with ribbed surface, rusty brown color, construction grid, individual rebar bars, detailed texture, high resolution",
            "horizontal rebar mesh, cylindrical steel bars with ridges, rusty surface, construction site, detailed metal texture, professional photography",
            "steel rebar grid, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed view, ultra sharp focus",
            
            # 弯筋高质量版本
            "bent steel rebar with ribbed surface, rusty brown color, curved cylindrical bars, construction site, detailed metal texture, high resolution",
            "curved rebar with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view, professional photography",
            "steel rebar bend, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture, ultra sharp focus",
            
            # 弯钩高质量版本
            "steel rebar with hook end, ribbed surface, rusty brown color, cylindrical bars, construction site, detailed metal texture, high resolution",
            "rebar hook with ridges, cylindrical steel bars, rusty surface, construction reinforcement, detailed view, professional photography",
            "steel rebar termination with hook, cylindrical with ribbed surface, rusty brown color, construction site, detailed texture, ultra sharp focus",
            
            # 绑扎铁丝高质量版本
            "steel rebar tied with wire, ribbed surface, rusty brown color, cylindrical bars, construction grid, detailed metal texture, high resolution",
            "rebar binding wire, cylindrical steel bars with ridges, rusty surface, construction site, detailed view, professional photography",
            "steel rebar mesh with wire ties, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture, ultra sharp focus",
            
            # 交叉点高质量版本
            "steel rebar intersection, ribbed surface, rusty brown color, cylindrical bars crossing, construction site, detailed metal texture, high resolution",
            "crossing rebar with ridges, cylindrical steel bars, rusty surface, construction grid, detailed view, professional photography",
            "steel rebar crossing point, cylindrical bars with ribbed surface, rusty brown color, construction reinforcement, detailed texture, ultra sharp focus"
        ]
        
        # 负面提示词 - 避免生成混凝土结构、管道等
        negative_prompt = "watermark, logo, text, blurry, lowres, overexposed, artifacts, lens flare, deformation, duplicate limbs, CGI, cartoon, illustration, unrealistic colors, pipes, formwork panels, scaffolding, workers, stirrup highlighted, binding wire emphasized"
        
        generated_images = []
        for i in tqdm(range(min(num_images, len(high_quality_prompts))), desc="生成高质量钢筋图像"):
            prompt = high_quality_prompts[i]
            output_name = f"hq_rebar_image_{i+1:02d}.png"
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 生成高质量图像
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,    # 更多步数
                    guidance_scale=8.5,        # 更高引导强度
                    height=768,                # 更高分辨率
                    width=768,                 # 更高分辨率
                    num_images_per_prompt=1
                ).images[0]
                
                # 计算生成时间
                generation_time = time.time() - start_time
                
                # 保存图像
                output_path = self.output_dir / output_name
                image.save(output_path)
                print(f"✅ 高质量图像已保存: {output_path} (耗时: {generation_time:.2f}秒)")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"❌ 高质量图像生成失败: {e}")
        
        print(f"✅ 生成了 {len(generated_images)} 张高质量钢筋检测图像")
        return generated_images
    
    def generate_custom_image(self, pipe, prompt, output_name=None):
        """生成自定义图像"""
        if output_name is None:
            output_name = f"custom_{prompt[:20].replace(' ', '_')}.png"
        
        return self.generate_image(pipe, prompt, output_name)
    
    def generate_images_with_detailed_progress(self, pipe, prompts, output_prefix="detailed"):
        """生成图像并显示详细进度"""
        print(f"🎯 开始生成 {len(prompts)} 张图像...")
        
        # 创建总进度条
        with tqdm(total=len(prompts), desc="总体进度", unit="张") as pbar:
            for i, prompt in enumerate(prompts):
                # 更新进度条描述
                pbar.set_description(f"生成第 {i+1}/{len(prompts)} 张")
                
                # 记录开始时间
                start_time = time.time()
                
                # 生成图像
                output_name = f"{output_prefix}_{i+1:02d}.png"
                image = self.generate_image(pipe, prompt, output_name)
                
                if image:
                    # 计算并显示详细信息
                    generation_time = time.time() - start_time
                    pbar.set_postfix({
                        '时间': f"{generation_time:.2f}s",
                        '文件': output_name
                    })
                
                # 更新进度
                pbar.update(1)
        
        print("✅ 所有图像生成完成！")
    
    def interactive_mode(self, pipe):
        """交互模式"""
        print("\n🎮 进入交互模式")
        print("输入 'quit' 退出")
        print("输入 'help' 查看帮助")
        
        while True:
            try:
                user_input = input("\n🎨 请输入图像描述: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 退出交互模式")
                    break
                elif user_input.lower() == 'help':
                    print("📖 帮助信息:")
                    print("- 输入图像描述来生成图像")
                    print("- 输入 'quit' 退出")
                    print("- 输入 'help' 查看此帮助")
                    continue
                elif not user_input:
                    continue
                
                # 生成图像
                output_name = f"interactive_{len(list(self.output_dir.glob('interactive_*.png')))+1:02d}.png"
                self.generate_image(pipe, user_input, output_name)
                
            except KeyboardInterrupt:
                print("\n👋 退出交互模式")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Diffusion 快速启动")
    parser.add_argument("--mode", choices=["demo", "interactive", "rebar", "hq_rebar", "simple_rebar", "translate", "rebar_interactive", "single_rebar", "ultra_rebar", "quality_mode"], 
                       default="demo", help="运行模式")
    parser.add_argument("--prompt", type=str, help="图像描述")
    parser.add_argument("--output", type=str, help="输出文件名")
    parser.add_argument("--num-images", type=int, default=5, help="生成图像数量")
    parser.add_argument("--rebar-type", type=str, help="钢筋类型 (main_rebar, stirrup, distribution_rebar, bent_rebar, hook_end, binding_wire, intersection)")
    
    args = parser.parse_args()
    
    # 创建快速启动实例
    quick_start = DiffusionQuickStart()
    
    # 如果是翻译模式，直接显示翻译
    if args.mode == "translate":
        quick_start.show_prompt_translation()
        return
    
    # 加载模型
    pipe = quick_start.load_model()
    if pipe is None:
        print("❌ 模型加载失败，程序退出")
        return
    
    # 根据模式运行
    if args.mode == "demo":
        print("🎯 演示模式")
        
        # 生成示例图像
        demo_prompts = [
            "a beautiful sunset over mountains, high quality, detailed",
            "a cute cat sitting on a windowsill, photorealistic",
            "steel bridge over river, construction, detailed"
        ]
        
        for i, prompt in enumerate(demo_prompts):
            output_name = f"demo_image_{i+1:02d}.png"
            quick_start.generate_image(pipe, prompt, output_name)
    
    elif args.mode == "interactive":
        quick_start.interactive_mode(pipe)
    
    elif args.mode == "rebar":
        quick_start.generate_rebar_images(pipe, args.num_images)
    
    elif args.mode == "hq_rebar":
        print("🏗️ 高质量钢筋检测模式")
        quick_start.generate_high_quality_rebar_images(pipe, args.num_images)
    
    elif args.mode == "simple_rebar":
        print("🏗️ 简单钢筋检测模式")
        quick_start.generate_simple_rebar_images(pipe, args.num_images)
    
    elif args.mode == "rebar_interactive":
        print("🏗️ 交互式钢筋生成器")
        quick_start.interactive_rebar_generator(pipe)
    
    elif args.mode == "single_rebar":
        if not args.rebar_type:
            print("❌ 请指定钢筋类型，使用 --rebar-type 参数")
            print("可用的钢筋类型: main_rebar, stirrup, distribution_rebar, bent_rebar, hook_end, binding_wire, intersection")
            return
        
        print(f"🏗️ 生成单个钢筋类型: {args.rebar_type}")
        quick_start.generate_single_rebar_type(pipe, args.rebar_type, args.num_images)
    
    elif args.mode == "ultra_rebar":
        if not args.rebar_type:
            print("❌ 请指定钢筋类型，使用 --rebar-type 参数")
            print("可用的钢筋类型: main_rebar, stirrup, distribution_rebar, bent_rebar, hook_end, binding_wire, intersection")
            return
        
        print(f"🏗️ 生成超高质量钢筋图像: {args.rebar_type}")
        quick_start.generate_ultra_quality_rebar(pipe, args.rebar_type, args.num_images)
    
    elif args.mode == "quality_mode":
        print("🏗️ 质量模式选择")
        quick_start.quality_mode_selection(pipe)
    
    # 如果提供了自定义提示
    if args.prompt:
        output_name = args.output or f"custom_{args.prompt[:20].replace(' ', '_')}.png"
        quick_start.generate_custom_image(pipe, args.prompt, output_name)

if __name__ == "__main__":
    main() 