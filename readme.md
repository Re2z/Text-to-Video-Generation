# Simple and Temporal Latent Diffusion Framework for Text-to-Video Generation

This code is used for the training, evaluation and inference for T2V generation. The code is based on the code of Stable Diffusion from the Diffusers.

First, you need to install the related libraries:
```
pip install -r requirements.txt 
```

# Train

To train the code, you first need to downtown the pretrained model of **Stable Diffusion V1.4** from https://huggingface.co/CompVis/stable-diffusion-v1-4

In addition, prepare a **dataset** folder for the training. 

All the configs of the training including the mentioned above are set in the config.yaml file. **Before starting training, you need to set the filepath to your own location in config.yaml.**

The filepath includes *"pretrained_model_path", "output_dir", "path"* in config.yaml

To start training:
```
python train.py --config config.yaml 
```

## Inference

To make an inference, set the *"pretrained_model_path"* to Stable Diffusion V1.4 and *"my_model_path"* to your trained model, using:
```
python Inference.py
```
