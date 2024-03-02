from enum import Enum
import gc
import numpy as np
import tomesd
import torch
from torch import cuda

from diffusers import UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from text_to_video_pipeline import TextToVideoPipeline

import utils
import gradio_utils
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


class ModelType(Enum):  # 枚举
    Text2Video = 2,


class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device  # 显卡
        self.dtype = dtype  # 数据类型
        self.generator = torch.Generator()  # 随机数生成器
        self.pipe_dict = {
            ModelType.Text2Video: TextToVideoPipeline,  # pipeline
        }
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)  # 交叉注意力

        self.pipe = None  # 选择哪个pipeline
        self.model_type = None

        self.model_name = ""  # 模型名字

    def set_model(self, model_type: ModelType, model_id: str, **kwargs):
        if hasattr(self, "pipe") and self.pipe is not None:  # 清空pipe
            del self.pipe
            self.pipe = None
        cuda.empty_cache()
        gc.collect()  # 垃圾回收
        safety_checker = kwargs.pop('safety_checker', None)
        self.pipe = self.pipe_dict[model_type].from_pretrained(  # text2video的pipeline 并存入self.pipe
            model_id, safety_checker=safety_checker, **kwargs).to(self.device).to(self.dtype)
        self.model_type = model_type
        self.model_name = model_id

    def inference_chunk(self, frame_ids, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        prompt = np.array(kwargs.pop('prompt'))
        negative_prompt = np.array(kwargs.pop('negative_prompt', ''))
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)  # 将该chunk的视频长度设置为frame_ids的长度
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(prompt=prompt[frame_ids].tolist(),  # 使用 frame_ids 中的整数索引从 prompt 中提取相应位置的字符
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):  # chunk_size表示了一次处理多少帧，越低gpu要求越低，不会影响质量
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")  # 应用压缩，值越大，内存用的越少，但是质量越低

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)  # 程序加速
        seed = kwargs.pop('seed', 0)  # 随机种子
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop('generator', '')

        if 'image' in kwargs:
            f = kwargs['image'].shape[0]
        else:
            f = kwargs['video_length']  # 视频总长度，总帧数

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f  # 为什么要把文本输入乘这么多次
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)  # （0，8，7） 就是0-7八个，chunk是7， 所以结果是【0，7】
            result = []
            for i in range(len(chunk_ids)):  # 迭代
                # 如果chunk_ids里面只有一个值【0】，那么start就是0，end就是8；如果不是这样，start就是0，end是下一个chunk_ids
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))  # 【0，0，1，2，3，4，5，6】；【0，7】
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                result.append(self.inference_chunk(frame_ids=frame_ids,
                                                   prompt=prompt,
                                                   negative_prompt=negative_prompt,
                                                   **kwargs).images[1:])  # 将inference_chunk返回的images去除第一个元素添加到result
                frames_counter += len(chunk_ids) - 1  # 计算生成了多少帧
                if on_huggingspace and frames_counter >= 80:
                    break
            result = np.concatenate(result)  # 将所有图像连成一个numpy数组
            return result
        else:
            self.generator.manual_seed(seed)
            return self.pipe(prompt=prompt, negative_prompt=negative_prompt, generator=self.generator, **kwargs).images

    def process_text2video(self,
                           prompt,
                           model_name="runwayml/stable-diffusion-v1-5",
                           # runwayml/stable-diffusion-v1-5,prompthero/openjourney,dreamlike-art/dreamlike-diffusion-1.0,CompVis/stable-diffusion-v1-4, dreamlike-art/dreamlike-photoreal-2.0
                           motion_field_strength_x=12,
                           motion_field_strength_y=12,
                           t0=44,
                           t1=47,
                           n_prompt="",
                           chunk_size=8,
                           video_length=8,
                           watermark='Picsart AI Research',
                           merging_ratio=0.0,
                           seed=0,
                           resolution=512,
                           fps=2,
                           use_cf_attn=True,
                           use_motion_field=True,
                           smooth_bg=False,
                           smooth_bg_strength=0.4,
                           path=None):
        print("Module Text2Video")
        if self.model_type != ModelType.Text2Video or model_name != self.model_name:
            print("Model update")
            unet = UNet2DConditionModel.from_pretrained(  # SD中的unet模块
                model_name, subfolder="unet")
            self.set_model(ModelType.Text2Video,  # 跳转到set model函数
                           model_id=model_name, unet=unet)
            self.pipe.scheduler = DDIMScheduler.from_config(  # 修改pipeline的scheduler，更换调度器但使用相同config
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(  # 修改unet中的attention
                    processor=self.text2video_attn_proc)
        self.generator.manual_seed(seed)  # 生成随机种子

        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        prompt = prompt.rstrip()
        if len(prompt) > 0 and (prompt[-1] == "," or prompt[-1] == "."):
            prompt = prompt.rstrip()[:-1]
        prompt = prompt.rstrip()
        prompt = prompt + ", " + added_prompt
        if len(n_prompt) > 0:
            negative_prompt = n_prompt
        else:
            negative_prompt = None

        result = self.inference(prompt=prompt,
                                video_length=video_length,
                                height=resolution,
                                width=resolution,
                                num_inference_steps=50,
                                guidance_scale=7.5,
                                guidance_stop_step=1.0,
                                t0=t0,
                                t1=t1,
                                motion_field_strength_x=motion_field_strength_x,
                                motion_field_strength_y=motion_field_strength_y,
                                use_motion_field=use_motion_field,
                                smooth_bg=smooth_bg,
                                smooth_bg_strength=smooth_bg_strength,
                                seed=seed,
                                output_type='numpy',
                                negative_prompt=negative_prompt,
                                merging_ratio=merging_ratio,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark))  # 生成视频
