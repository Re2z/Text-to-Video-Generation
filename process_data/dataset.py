import os
import decord
import numpy as np
import random
import torchvision.transforms as T
import torch

from glob import glob
from itertools import islice
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat


def normalize_input(
        item,
        mean=[0.5, 0.5, 0.5],  # Imagenet [0.485, 0.456, 0.406]
        std=[0.5, 0.5, 0.5],  # Imagenet [0.229, 0.224, 0.225]
        use_simple_norm=False
):
    if item.dtype == torch.uint8 and not use_simple_norm:
        item = rearrange(item, 'f c h w -> f h w c')

        item = item.float() / 255.0
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        out = rearrange((item - mean) / std, 'f h w c -> f c h w')

        return out
    else:

        item = rearrange(item, 'f c h w -> f h w c')
        return rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')


def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    return prompt_ids


def read_caption_file(caption_file):
    with open(caption_file, 'r', encoding="utf8") as t:
        return t.read()


def get_text_prompt(
        text_prompt: str = '',
        fallback_prompt: str = '',
        file_path: str = '',
        ext_types=['.mp4'],
        use_caption=False
):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file):
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)

            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt


def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr


class VideoFolderDataset(Dataset):
    def __init__(
            self,
            tokenizer=None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 16,
            fps: int = 8,
            path: str = "./dataset",
            fallback_prompt: str = "",
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()

        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)

        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
            vid_path,
            self.use_bucketing,
            self.width,
            self.height,
            self.get_frame_buckets,
            self.get_frame_batch
        )
        return video, vr

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__():
        return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):

        video, _ = self.process_video_wrapper(self.video_files[index])

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        else:
            prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": normalize_input(video[0]), "prompt_ids": prompt_ids, "text_prompt": prompt,
                'dataset': self.__getname__()}
