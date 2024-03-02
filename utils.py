import os

import numpy as np
import torch
import torchvision
import imageio
from einops import rearrange
from PIL import Image


def add_watermark(image, watermark_path, wm_rel_size=1 / 16, boundary=5):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ('RGBA', 'LA') else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:  # 设置保存路径
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)  # 将输入的图像（存储在张量 x 中）按照每行 4 个的方式排列成一个网格。可视化一批图像
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


@torch.no_grad()
def compute_clip_score(
        model, model_processor, images, texts, local_bs=32, rescale=False
):
    if rescale:
        images = (images + 1.0) / 2.0  # -1,1 -> 0,1
    images = (images * 255).to(torch.uint8)
    clip_scores = []
    for start_idx in range(0, images.shape[0], local_bs):
        img_batch = images[start_idx: start_idx + local_bs]
        batch_size = img_batch.shape[0]  # shape: [b c t h w]
        img_batch = rearrange(img_batch, "b c t h w -> (b t) c h w")
        outputs = []
        for i in range(len(img_batch)):
            images_part = img_batch[i: i + 1]
            model_inputs = model_processor(
                text=texts, images=list(images_part), return_tensors="pt", padding=True
            )
            model_inputs = {
                k: v.to(device=model.device, dtype=model.dtype)
                if k in ["pixel_values"]
                else v.to(device=model.device)
                for k, v in model_inputs.items()
            }
            logits = model(**model_inputs)["logits_per_image"]
            # For consistency with `torchmetrics.functional.multimodal.clip_score`.
            logits = logits / model.logit_scale.exp()
            outputs.append(logits)
        logits = torch.cat(outputs)
        logits = rearrange(logits, "(b t) p -> t b p", b=batch_size)
        frame_sims = []
        for logit in logits:
            frame_sims.append(logit.diagonal())
        frame_sims = torch.stack(frame_sims)  # [t, b]
        clip_scores.append(frame_sims.mean(dim=0))
    return torch.cat(clip_scores)
