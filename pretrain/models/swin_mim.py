# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
rearrange, _ = optional_import("einops", name="rearrange")
from monai.networks.layers import DropPath, trunc_normal_
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from collections.abc import Sequence
from torch.nn import LayerNorm

def random_mask(image):
    H, W, D = image.shape[0], image.shape[1], image.shape[2]
    # Determine how many 32x32 patches fit along each dimension
    num_patches_h = H // 32
    num_patches_w = W // 32
    num_patches_d = D // 32
    random_grid = np.random.randint(0, 2, size=(num_patches_h, num_patches_w, num_patches_d))

    mask = np.kron(random_grid, np.ones((16, 16, 16)))
    mask = mask[:H, :W, :D] 
    return mask

class SwinTransformerMasked(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, normalize=True):
        x0 = self.patch_embed(x)
        B, L, H, W, D = x0.shape
        mask_tokens = self.mask_token.expand(B, L, H, W, D)
        w = mask.type_as(mask_tokens)
        
        x0 = x0 * (1. - w) + mask_tokens * w
        
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]

class SwinEncoder(nn.Module):
    def __init__(self, spatial_dims, in_channels, feature_size, dropout_path_rate=0.1, use_checkpoint=False):
        super(SwinEncoder, self).__init__()
        self.in_chans = in_channels
        self.num_features = feature_size
        
        self.patch_size = 2
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinTransformerMasked(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=16 * feature_size,
        #     out_channels=8 * feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 8,
        #     out_channels=feature_size * 4,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 4,
        #     out_channels=feature_size * 2,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 2,
        #     out_channels=feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder1 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size,
        #     out_channels=feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

    # def forward_encs(self, encs):
    #     b = encs[0].size()[0]
    #     outs = []
    #     for enc in encs:
    #         out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
    #         outs.append(out.view(b, -1))
    #     outs = torch.cat(outs, dim=1)
    #     return outs

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        # logits = self.out(out)
        encs = [enc0, enc1, enc2, enc3, dec4]

        return encs
    
    def forward_mask(self, x_in, mask):
        hidden_states_out = self.swinViT(x_in, mask)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        # logits = self.out(out)
        encs = [enc0, enc1, enc2, enc3, dec4]
        return encs

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.upscale_factor
        assert channels % (r**3) == 0, "Channels must be divisible by upscale_factor^3"
        new_c = channels // (r**3)
        x = x.view(batch_size, new_c, r, r, r, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.contiguous().view(batch_size, new_c, depth * r, height * r, width * r)
        return x

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super(SimMIM, self).__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.encoder.num_features * 16,
                out_channels=self.encoder_stride ** 3 * 1, kernel_size=1),
            PixelShuffle3D(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        
    def random_mask(self, x):
        B, C, H, W, D = x.size()
        # create random mask for each image in the batch with probability 0.7
        mask = torch.rand(B, 1, H // 2, W // 2, D // 2) < 0.7
        return mask
        
    def forward(self, x, mask):
        # mask = self.random_mask(x)
        mask = mask.to(x.device)
        z = self.encoder.forward_mask(x, mask)
        x_rec = self.decoder(z[-1])
        mask = mask.repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).repeat_interleave(self.patch_size, 4).contiguous()
        loss_recon = F.l1_loss(x_rec, x, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        mask = mask.bool()
        x_rec = x * (~ mask) + x_rec * mask
        return loss, x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
    
if __name__ == "__main__":
    swin_encoder = SwinEncoder(spatial_dims=3,
            in_channels=1,
            feature_size=48)
    swin_mim = SimMIM(swin_encoder, 32)