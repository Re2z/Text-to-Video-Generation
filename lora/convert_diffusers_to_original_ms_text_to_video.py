# =================#
# UNet Conversion #
# =================#

print ('Initializing the conversion map')

unet_conversion_map = [
    # (ModelScope, HF Diffusers)

    # from Vanilla ModelScope/StableDiffusion
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),


    # from Vanilla ModelScope/StableDiffusion
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),


    # from Vanilla ModelScope/StableDiffusion
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (ModelScope, HF Diffusers)

    # SD
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),

    # MS
    #("temopral_conv", "temp_convs"), # ROFL, they have a typo here --kabachuha
]

unet_conversion_map_layer = []

# Convert input TemporalTransformer
unet_conversion_map_layer.append(('input_blocks.0.1', 'transformer_in'))


def convert_unet_state_dict(unet_state_dict, strict_mapping=False):
    print('Converting the UNET')
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}

    for sd_name, hf_name in unet_conversion_map:
        if strict_mapping:
            if hf_name in mapping:
                mapping[hf_name] = sd_name
        else:
            mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
        # elif "temp_convs" in k:
        #     for sd_part, hf_part in unet_conversion_map_resnet:
        #         v = v.replace(hf_part, sd_part)
        #     mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v

    # there must be a pattern, but I don't want to bother atm
    do_not_unsqueeze = [f'output_blocks.{i}.1.proj_out.weight' for i in range(3, 12)] + [
        f'output_blocks.{i}.1.proj_in.weight' for i in range(3, 12)] + ['middle_block.1.proj_in.weight',
                                                                        'middle_block.1.proj_out.weight'] + [
                           f'input_blocks.{i}.1.proj_out.weight' for i in [1, 2, 4, 5, 7, 8]] + [
                           f'input_blocks.{i}.1.proj_in.weight' for i in [1, 2, 4, 5, 7, 8]]
    print(do_not_unsqueeze)

    new_state_dict = {v: (
        unet_state_dict[k].unsqueeze(-1) if ('proj_' in k and ('bias' not in k) and (k not in do_not_unsqueeze)) else
        unet_state_dict[k]) for k, v in mapping.items()}
    # HACK: idk why the hell it does not work with list comprehension
    for k, v in new_state_dict.items():
        has_k = False
        for n in do_not_unsqueeze:
            if k == n:
                has_k = True

        if has_k:
            v = v.squeeze(-1)
        new_state_dict[k] = v

    return new_state_dict
