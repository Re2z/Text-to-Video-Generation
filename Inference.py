import torch
from model import Model

model = Model(device="cuda", dtype=torch.float16)

prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47, "motion_field_strength_x": 12, "motion_field_strength_y": 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ', '_')}.mp4", 4
model.process_text2video(prompt, fps=fps, path=out_path, **params)
