import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SwinEncoder(nn.Module):
    def __init__(self, spatial_dims, in_channels, feature_size, dropout_path_rate=0.1, use_checkpoint=False):
        super(SwinEncoder, self).__init__()
        self.in_chans = in_channels
        self.num_features = feature_size
        
        self.patch_size = 2
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinViT(
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
        #     spatial_dims=args.spatial_dims,
        #     in_channels=16 * args.feature_size,
        #     out_channels=8 * args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 8,
        #     out_channels=args.feature_size * 4,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 4,
        #     out_channels=args.feature_size * 2,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 2,
        #     out_channels=args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder1 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size,
        #     out_channels=args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]

        # for enc in encs:
        #     print(enc.shape)

        out = self.forward_encs(encs)
        return out, encs
    
    
if __name__ == '__main__':
    model = SwinEncoder(spatial_dims=3, 
                        in_channels=1, 
                        feature_size=48,
                        dropout_path_rate=0.1,)