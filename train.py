import argparse
import copy
import datetime
import gc
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from process_data.dataset import CachedDataset, VideoJsonDataset, SingleVideoDataset, VideoFolderDataset
from models.unet import UNet3DConditionModel
from text_to_video_pipeline import TextToVideoPipeline
from utils import create_video
from einops import rearrange

already_printed_trainables = False
logger = get_logger(__name__, log_level="INFO")


def get_train_dataset(dataset_types, train_data, tokenizer):
    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")


def unet_and_text_g_c(unet, unet_enable):
    unet._set_gradient_checkpointing(value=unet_enable)


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        cached_latent_dir=None,
        shuffle=False
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()

    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None
    )
    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            for k, v in batch.items(): batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents


def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1) \
        and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        # lora_manager: LoraHandler,
        # unet_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_save = copy.deepcopy(unet.cpu())
    text_encoder_save = copy.deepcopy(text_encoder.cpu())

    unet_out = copy.deepcopy(accelerator.unwrap_model(unet_save, keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder_save, keep_fp32_wrapper=False))

    pipeline = TextToVideoPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)

    # lora_manager.save_lora_weights(model=pipeline, save_path=save_path, step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = (
                "spatial_attn",
                "temp_attn",
                "conv1_3d",
                "conv2_3d"
        ),
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 3e-5,
        shuffle: bool = True,
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        dataset_types: Tuple[str] = ('json'),
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        cache_latents: bool = False,
        cached_latent_dir=None,
        # lora_version: LORA_VERSIONS = LORA_VERSIONS[0],
        lora_bias: str = 'none',
        use_unet_lora: bool = False,
        unet_lora_modules: Tuple[str] = ["UNet3DConditionModel"],
        lora_rank: int = 16,
        lora_path: str = '',
        lora_unet_dropout: float = 0.1,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        resume_step: Optional[int] = None,
        offset_noise_strength: float = 0.1,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        save_pretrained_model: bool = True,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        seed: Optional[int] = None,
        logger_type: str = 'tensorboard',
        **kwargs,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = KarrasDiffusionSchedulers.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

        # Use LoRA if enabled.
    # lora_manager = LoraHandler(
    #     version=lora_version,
    #     use_unet_lora=use_unet_lora,
    #     unet_replace_modules=unet_lora_modules,
    #     lora_bias=lora_bias
    # )

    # unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
    #     use_unet_lora, unet, lora_manager.unet_replace_modules, lora_unet_dropout, lora_path, r=lora_rank)

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the training dataset
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle
    )

    # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        cached_latent_dir
    )

    if cached_data_loader is not None:
        train_dataloader = cached_data_loader

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        gradient_checkpointing,
    )

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_unet(batch):
        nonlocal use_offset_noise
        nonlocal rescale_schedule

        # Convert videos to latent space
        pixel_values = batch["pixel_values"]

        latents = tensor_to_vae_latent(pixel_values, vae)

        # Get video length
        video_length = latents.shape[2]

        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()

        # Encode text embeddings
        token_ids = batch['prompt_ids']

        # Assume extra batch dimnesion.
        if len(token_ids.shape) > 2:
            token_ids = token_ids[0]

        encoder_hidden_states = text_encoder(token_ids)[0]

        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss, latents

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):

                text_prompt = batch['text_prompt'][0]

                with accelerator.autocast():
                    loss, latents = finetune_unet(batch)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}")
                    continue

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        text_encoder,
                        vae,
                        output_dir,
                        # lora_manager,
                        # unet_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if global_step == 1: print("Performing validation prompt.")
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            unet.eval()
                            unet_and_text_g_c(unet, False)
                            # lora_manager.deactivate_lora_train(unet, True)

                            pipeline = TextToVideoPipeline.from_pretrained(
                                pretrained_model_path,
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            prompt = text_prompt if len(validation_data.prompt) <= 0 else validation_data.prompt

                            curr_dataset_name = batch['dataset']
                            save_filename = f"{global_step}_dataset-{curr_dataset_name}_{prompt}"

                            out_file = f"{output_dir}/samples/{save_filename}.mp4"

                            with torch.no_grad():
                                video_frames = pipeline(
                                    prompt,
                                    width=validation_data.width,
                                    height=validation_data.height,
                                    num_frames=validation_data.num_frames,
                                    num_inference_steps=validation_data.num_inference_steps,
                                    guidance_scale=validation_data.guidance_scale
                                ).frames
                            create_video(video_frames, train_data.get('fps', 8), out_file)

                            del pipeline
                            torch.cuda.empty_cache()

                    logger.info(f"Saved a new sample to {out_file}")

                    unet_and_text_g_c(
                        unet,
                        gradient_checkpointing,
                    )

                    # lora_manager.deactivate_lora_train(unet, False)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_pipe(
                pretrained_model_path,
                global_step,
                accelerator,
                unet,
                text_encoder,
                vae,
                output_dir,
                # lora_manager,
                # unet_lora_modules,
                is_checkpoint=False,
                save_pretrained_model=save_pretrained_model
            )
        accelerator.end_training()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
        args = parser.parse_args()

        main(**OmegaConf.load(args.config))
