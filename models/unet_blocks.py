from torch import nn
from mid_resnet3d import ResnetBlock3D
from mid_attention3d import Transformer3DModel
from utils import checkpoint, zero_module


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, num_layers=4, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim

        # conv layers
        convs = []
        prev_dim, next_dim = in_dim, out_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                next_dim = out_dim
            convs.extend(
                [
                    nn.GroupNorm(32, prev_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv3d(prev_dim, next_dim, (3, 1, 1), padding=(1, 0, 0)),
                ]
            )
            prev_dim, next_dim = next_dim, prev_dim
        self.convs = nn.ModuleList(convs)

    def forward(self, hidden_states):
        video_length = hidden_states.shape[2]

        identity = hidden_states
        for conv in self.convs:
            hidden_states = conv(hidden_states)

        # ignore effects of temporal layers on image inputs
        hidden_states = (
            identity + hidden_states
            if video_length > 1
            else identity + 0.0 * hidden_states
        )

        return hidden_states


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels=1,
            output_scale_factor=1.0,
            cross_attention_dim=1280,
            dual_cross_attention=False,
            use_linear_projection=False,
            upcast_attention=False,
            add_temp_attn=False,
            prepend_first_frame=False,
            add_temp_embed=False,
            add_temp_conv=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if add_temp_conv:
            self.temp_convs = None
            temp_convs = [TemporalConvLayer(in_channels, in_channels, dropout=0.1)]
            temp_convs[-1].convs[-1] = zero_module(temp_convs[-1].convs[-1])

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        prepend_first_frame=prepend_first_frame,
                        add_temp_embed=add_temp_embed,
                    )
                )
            else:
                raise NotImplementedError
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if hasattr(self, "temp_convs"):
                temp_convs.append(
                    TemporalConvLayer(in_channels, in_channels, dropout=0.1)
                )
                temp_convs[-1].convs[-1] = zero_module(temp_convs[-1].convs[-1])

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if hasattr(self, "temp_convs"):
            self.temp_convs = nn.ModuleList(temp_convs)

    def forward(
            self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs=None
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        if hasattr(self, "temp_convs"):
            hidden_states = self.temp_convs[0](hidden_states)
        for layer_idx in range(len(self.attentions)):
            attn = self.attentions[layer_idx]
            resnet = self.resnets[layer_idx + 1]
            hidden_states = attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states
            )[0]
            hidden_states = resnet(hidden_states, temb)
            if hasattr(self, "temp_convs"):
                temp_conv = self.temp_convs[layer_idx + 1]
                hidden_states = temp_conv(hidden_states)

        return hidden_states
