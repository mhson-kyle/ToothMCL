import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from external_libs.pointops.functions import pointops

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pt_blocks import TransitionDown, TransitionUp

class PointTransformerEncoder(nn.Module):
    def __init__(self, 
                 block, 
                 blocks, 
                 in_channels, 
                 mask_head=None, 
                 planes=None):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_channels
        self.mask_head = mask_head
        # fdims
        planes = [32, 64, 128, 256, 512]
        # shared head in att
        # if 'share_planes' not in config:
        #     config.share_planes = 8
        # share_planes = config.share_planes

        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(
            block, 
            planes[0], 
            blocks[0], 
            share_planes, 
            stride=stride[0], 
            nsample=nsample[0]
            )  # N/1   - planes(fdims)=32,  blocks=2, nsample=8
        
        self.enc2 = self._make_enc(
            block, 
            planes[1], 
            blocks[1], 
            share_planes, 
            stride=stride[1], 
            nsample=nsample[1]
            )  # N/4   - planes(fdims)=64,  blocks=3, nsample=16
        
        self.enc3 = self._make_enc(
            block, 
            planes[2], 
            blocks[2], 
            share_planes, 
            stride=stride[2], 
            nsample=nsample[2]
            )  # N/16  - planes(fdims)=128, blocks=4, nsample=16KKK
        
        self.enc4 = self._make_enc(
            block, 
            planes[3], 
            blocks[3], 
            share_planes, 
            stride=stride[3], 
            nsample=nsample[3]
            )  # N/64  - planes(fdims)=256, blocks=6, nsample=16
        
        self.enc5 = self._make_enc(
            block, 
            planes[4], 
            blocks[4], 
            share_planes, 
            stride=stride[4], 
            nsample=nsample[4]
            )  # N/256 - planes(fdims)=512, blocks=3, nsample=16
        
        self.dec5 = self._make_dec(block, 
            planes[4], 2, share_planes, nsample=nsample[4], is_head=True
            )  # transform p5
        
        self.dec4 = self._make_dec(block, 
            planes[3], 2, share_planes, nsample=nsample[3] 
            )  # fusion p5 and p4
        
        self.dec3 = self._make_dec(block, 
            planes[2], 2, share_planes, nsample=nsample[2]
            )  # fusion p4 and p3
        
        self.dec2 = self._make_dec(block, 
            planes[1], 2, share_planes, nsample=nsample[1]
            )  # fusion p3 and p2
        
        self.dec1 = self._make_dec(block, 
            planes[0], 2, share_planes, nsample=nsample[0]
            )  # fusion p2 and p1
        

        # self.config = config
        # config.num_layers = block_num
        # config.num_classes = num_classes
        # if 'multi' in config:
        #     self.mask_head = MultiHead(planes, config.multi, config, k=2)
        #     self.cls_head = MultiHead(planes, config.multi, config, k=self.num_classes)
        #     self.offset_head = MultiHead(planes, config.multi, config, k=3)
        # else:
        #     self.cls = nn.Sequential(
        #         nn.Linear(planes[0], planes[0]), 
        #         nn.BatchNorm1d(planes[0]), 
        #         nn.ReLU(inplace=True), 
        #         nn.Linear(planes[0], in_channels))
        #     self.offset_head = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 3))


    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        stride = 1 => TransitionDown = mlp, [block, ...]
        stride > 1 => 
        """
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
            ]
        self.in_planes = planes * block.expansion  # expansion default to 1
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
                )
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, points, features, offset):
    # def forward(self, inputs):
        """
        input:
            inputs[0] -> pxo -> batch_size, channel, 24000
            inputs[1] -> target -> batch_size, 24000
            inputs[2] -> pxo_prev -> batch_size, channel(32), 24000
        """
        # B, C, N = inputs[0].shape
        # print(B)
        # pxo = inputs[0].permute(0, 2, 1) # (B, N, C)
        # x0 = pxo.reshape(-1, C) # (B * N, C)
        # p0 = pxo[:, :, :3].reshape(-1, 3).contiguous() # (B * N, 3)
        # o0 = torch.arange(1, B + 1, dtype=torch.int, device=inputs[0].device) # (B, ) (1, 2, ..., B)
        # o0 *= N # (B, ) (N, 2N, ..., BN)

        # stage_list = {'inputs': inputs}

        p0 = points.reshape(-1, 3).contiguous() # (B * N, 3)
        x0 = features.reshape(-1, 6) # (B * N, C)
        o0 = offset.cuda()
        
        print(p0.shape, x0.shape, o0.shape)
        stage_list = {'inputs': [p0, x0, o0]}
        
        p1, x1, o1 = self.enc1([p0, x0, o0]); print(p0.shape, x0.shape, o0.shape)
        p2, x2, o2 = self.enc2([p1, x1, o1]); print(p1.shape, x1.shape, o1.shape)
        p3, x3, o3 = self.enc3([p2, x2, o2]); print(p2.shape, x2.shape, o2.shape)
        p4, x4, o4 = self.enc4([p3, x3, o3]); print(p3.shape, x3.shape, o3.shape)
        p5, x5, o5 = self.enc5([p4, x4, o4]); print(p4.shape, x4.shape, o4.shape)
        print(p5.shape, x4.shape, o5.shape)

        down_list = [
            # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
        ]
        # for i, s in enumerate(down_list):
        #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
        stage_list['down'] = down_list

        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]  # no upsample - concat with per-cloud mean: mlp[ x, mlp[mean(x)] ]
        # x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        # x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        # x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        # x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        # up_list = [
        #     {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
        #     {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
        #     {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
        #     {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
        #     {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
        # ]
        # stage_list['up'] = up_list

        # if self.cls_head is not None:
        #     cls_results, stage_list = self.cls_head(stage_list)
        #     if B==1:
        #         offset_results, _ = self.offset_head(stage_list)
        #     else:
        #         offset_results = None
        # else:
        #     cls_results = self.cls(x1)
        #     offset_results = self.offset(x1)
        
        # output = []
        # if len(inputs) == 2:
        #     target = inputs[1].reshape(-1) # target: n
        #     target = target.type(torch.long)
        #     target = target + 1 # -1 is gingiva now with out subtraction
        #     info_loss = self.criterion(cls_results, target, stage_list)
        #     output.append(info_loss)

        # cls_results = cls_results.view(B, N, self.num_classes).permute(0,2,1)
        # if B==1:
        #     offset_results = offset_results.view(B, N, 3).permute(0,2,1)
        # else:
        #     offset_results = None
        # output.append(cls_results)
        # output.append(offset_results)
        # output.append(None)
        # output.append(x1)
        
        return down_list

# -------------------------------------------------------------------
# 1. Masking function: splits the point cloud into visible and masked subsets
# -------------------------------------------------------------------
def random_mask_points(points, o, mask_ratio):
    """
    p: Tensor of shape (N, 3) with point coordinates.
    o: Tensor of shape (B,) with cumulative offsets for each batch.
    mask_ratio: Fraction of points to mask (e.g., 0.3 for 30%).
    
    Returns:
        visible_idx: LongTensor of indices for points that remain visible.
        masked_idx: LongTensor of indices for points that are masked.
    """
    N = 24000
    M = 64 
    
    offset = torch.tensor([N], dtype=torch.int32).cuda()       # for the entire point cloud
    new_offset = torch.tensor([M], dtype=torch.int32).cuda()     # for the sampled seeds

    center_idx = pointops.furthestsampling(points, offset, new_offset).long()
    print("Selected seed indices:", center_idx)
    print(len(center_idx))

    center_points = points[center_idx.detach().cpu().numpy(), :]  # shape: (M, 3)
    print("M:", M)  # Debug print
    center_offset = torch.tensor([M], dtype=torch.int32).cuda()
    points_offset = torch.tensor([N], dtype=torch.int32).cuda()

    neighbor_idx, neighbor_dist = pointops.knnquery(
        1, 
        center_points, 
        points, 
        center_offset,
        points_offset
        )
    cluster_labels = neighbor_idx[:, 0].cpu().numpy()  # shape: (N,)
    
    # mask around 8 clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Determine the number of clusters to mask (approximately 20%).
    num_clusters_to_mask = int(len(unique_clusters) * mask_ratio)
    
    # Randomly select clusters to mask.
    masked_clusters = np.random.choice(unique_clusters, num_clusters_to_mask, replace=False)
    
    # Create a boolean mask: True if the point's cluster label is in the masked clusters.
    mask = np.isin(cluster_labels, masked_clusters)
    
    return mask, ~mask


# -------------------------------------------------------------------
# 3. Masked Autoencoder using the point transformer encoder and a lightweight decoder
# -------------------------------------------------------------------
class MaskedPointTransformerMAE(nn.Module):
    def __init__(self, encoder, decoder_embed_dim, mask_ratio=0.3, reconstruction_dim=3):
        """
        encoder: An instance of PointTransformerEncoder (or a similar encoder).
        decoder_embed_dim: Dimension of the tokens entering the decoder (should equal encoder output dim).
        mask_ratio: Fraction of points to mask.
        reconstruction_dim: Dimension of the target to reconstruct (e.g., 3 for point coordinates).
        """
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        in_channels = 512
        hidden_channels = 256
        out_channels = 6
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, point, features, offset=None):
        """
        pxo: Tuple (p, x, o) where:
            p: Tensor of shape (N, 3) with point positions.
            x: Tensor of shape (N, C) with point features.
            o: Tensor of shape (B,) with cumulative point counts per batch.
        """
        B, C, N = point.shape
        x0 = features.reshape(-1, 6) # (B * N, C)
        p0 = point.reshape(-1, 3).contiguous() # (B * N, 3)
        o0 = torch.arange(1, B + 1, dtype=torch.int32, device=point.device)  * 24000 # (B, ) (1, 2, ..., B)
        mask_idx, visible_idx = random_mask_points(p0, o0, self.mask_ratio)
        # Create a linear mlp to project the input features to the desired dimension.

        
        print('P_VISIBLE SHAPE: ', p0.shape)
        print('X_VISIBLE SHAPE: ', x0.shape)
        x0[mask_idx] = 0 # Zero out the features of masked points.
        print('P_VISIBLE SHAPE: ', p0.shape)
        print('X_VISIBLE SHAPE: ', x0.shape)
        print('O0 SHAPE: ', o0)        
        # Encode only the visible points.
        z = self.encoder(p0, x0, o0)
        
        latent_full = z[-1]['f_out']  # Expected shape: (N_visible, D)
        p_coarse = z[-1]['p_out']  # Expected shape: (N_visible, D)
        o0_coarse = z[-1]['offset']  # Expected shape: (B,)

        f_dense = pointops.interpolation(
            xyz=p_coarse, 
            new_xyz=p0, 
            feat=latent_full, 
            offset=o0_coarse, 
            new_offset=o0
        ) # (N / 256, 512) -> (N, 512)   
        
        x_rec = self.mlp(f_dense) # (N, 512) -> (N, 6)

        loss_recon = F.l1_loss(x_rec, x0, reduction='none')
        loss_recon = loss_recon[mask_idx].mean()
        x_rec[visible_idx] = x0[visible_idx]
        return loss_recon, x_rec, mask_idx, visible_idx, x0

# -------------------------------------------------------------------
# Example usage:
# Assume that `point_transformer_encoder` is an instance of PointTransformerEncoder.
# decoder_embed_dim should match the dimension of the encoder’s output (e.g., 512).
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Dummy inputs for demonstration
    B, N, C = 1, 24000, 3  # batch size, number of points, feature dimension
    dummy_point = torch.randn(B, N, 3).cuda()
    dummy_features = torch.randn(B, N, 6).cuda()
    # Create offsets: e.g., for two examples, offsets [N, 2N]
    offsets = torch.arange(1, B + 1, dtype=torch.int32) * N
    offsets = offsets.cuda()
    cluster_label = random_mask_points(dummy_point, offsets, mask_ratio=0.3)
    print(cluster_label)
    from .pt_blocks import PointTransformerBlock
    from .pt_encoder import PointTransformerEncoder

    point_transformer_encoder = PointTransformerEncoder(
        block=PointTransformerBlock, 
        blocks=[2, 3, 4, 6, 3],
        in_channels=6
        ).cuda()
    
    # Create the masked autoencoder instance.
    mae_model = MaskedPointTransformerMAE(
        encoder=point_transformer_encoder,
        decoder_embed_dim=512,
        mask_ratio=0.3,
        reconstruction_dim=3
    ).cuda()
    
    # Forward pass.
    loss, reconstruction, visible_idx, masked_idx, x0 = mae_model(dummy_point, dummy_features, offsets)
    print("Reconstruction loss:", loss.item())
    
