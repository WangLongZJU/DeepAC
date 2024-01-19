from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

# from mylib.pytorch_lightning.base_module import load_pretrained_dict
from .base_model import BaseModel
# from .mobileone import MobileOneBlock
# import segmentation_models_pytorch as smp

from segmentation_models_pytorch.encoders import get_encoder, get_preprocessing_fn, get_preprocessing_params
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock, CenterBlock
# from segmentation_models_pytorch.decoders.fpn.decoder

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        xs = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            xs.append(x)

        return xs


class BackboneUnet(BaseModel):
    default_conf = {
        'num_output_layer': 3,
        'output_dim': [16, 16, 16],
        'encoder': 'mobileone_s0',
        'encoder_depth': 5,
        'decoder': 'UnetDecoder',
        'decoder_channels': [256, 128, 64, 32, 16],
        'align_data_to_pretrain': False,
        'pretrained_weights': 'imagenet',
        # 'compute_uncertainty': False,
    }

    required_data_keys = {

    }

    def _init(self, conf):
        # self.test_unet = smp.Unet(encoder_name='mobileone_s0', 
        #                          encoder_weights='imagenet',
        #                          in_channels=3, classes=16)
        self.conf = conf
        encoder_depth = conf.encoder_depth
        decoder_channels = conf.decoder_channels
        decoder_use_batchnorm = True
        decoder_attention_type = None
        pretrained_weights = conf.pretrained_weights

        if conf.align_data_to_pretrain:
            self.preprocess_params = get_preprocessing_params(conf.encoder, pretrained=pretrained_weights)
            # self.preprocess_input = get_preprocessing_fn(conf.encoder, pretrained=pretrained_weights)
            self.mean = torch.from_numpy(np.asarray(self.preprocess_params['mean'])).float()
            self.std = torch.from_numpy(np.asarray(self.preprocess_params['std'])).float()
        else:
            self.preprocess_params = None
            # self.preprocess_input = None

        self.encoder = get_encoder(
            conf.encoder,
            in_channels=3,
            depth=encoder_depth,
            weights=pretrained_weights,
        )
        
        decoder_class = getattr(sys.modules[__name__], conf.decoder)
        self.decoder = decoder_class(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.smooth_layers = nn.ModuleList()
        for i, output_dim in enumerate(conf.output_dim):
            idx = conf.num_output_layer - i
            self.smooth_layers.append(nn.Conv2d(in_channels=decoder_channels[-idx], out_channels=output_dim,
                                                kernel_size=3, stride=1, padding=1))

    # for deploy
    def forward(self, x):
        # x0 = self.preprocess_input(x)
        # x = self.preprocess_input(x) if self.conf.align_data_to_pretrain else x
        if self.conf.align_data_to_pretrain:
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x - mean[None, :, None, None]
            x = x / std[None, :, None, None]
        x1 = self.encoder(x)
        x2 = self.decoder(*x1)

        output_layers = x2[-self.conf.num_output_layer:]
        outputs = []
        for smooth_layer, output_layer in zip(self.smooth_layers, output_layers):
            outputs.append(smooth_layer(output_layer))

        return outputs[0], outputs[1], outputs[2]

    def _forward(self, x):
        # x0 = self.preprocess_input(x)
        # x = self.preprocess_input(x) if self.conf.align_data_to_pretrain else x
        if self.conf.align_data_to_pretrain:
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x - mean[None, :, None, None]
            x = x / std[None, :, None, None]
        x1 = self.encoder(x)
        x2 = self.decoder(*x1)

        output_layers = x2[-self.conf.num_output_layer:]
        outputs = []
        for smooth_layer, output_layer in zip(self.smooth_layers, output_layers):
            outputs.append(smooth_layer(output_layer))

        return outputs

    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError