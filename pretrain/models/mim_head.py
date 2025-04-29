import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import einops
from external_libs.pointops.functions import pointops

from models.pt_blocks import PointTransformerBlock
from models.pt_encoder import PointTransformerEncoder
from models.pt_mim import random_mask_points
from models.swin_mim import SwinEncoder, PixelShuffle3D


class ImagePCDMIMModel(nn.Module):
    def __init__(self, 
                 img_spatial_dim, 
                 img_in_channels, 
                 img_feature_size, 
                 dropout_path_rate,

                 pcd_in_channels=6,
                 pcd_blocks=[2, 3, 4, 6, 3],
                 pcd_stride=[1, 4, 4, 4, 4],
                 pcd_nstride=[2, 2, 2, 2],
                 pcd_nsample=[36, 24, 24, 24, 24],
                 pcd_planes=[32, 64, 128, 256, 512],
                 contain_weight=False,
                 crop_sample_size=3072,
                 mask_ratio=0.7,
                 device='cuda'):
        super(ImagePCDMIMModel, self).__init__()
        
        self.img_spatial_dim = img_spatial_dim
        self.img_in_channels = img_in_channels
        self.img_feature_size = img_feature_size
        self.dropout_path_rate = dropout_path_rate

        self.pcd_in_channels = pcd_in_channels
        self.pcd_blocks = pcd_blocks
        self.pcd_stride = pcd_stride
        self.pcd_nstride = pcd_nstride
        self.pcd_nsample = pcd_nsample
        self.pcd_planes = pcd_planes
        self.contain_weight = contain_weight
        self.crop_sample_size = crop_sample_size
        self.mask_ratio = mask_ratio

        #### Image #####
        self.image_encoder = SwinEncoder(
            spatial_dims=self.img_spatial_dim, 
            in_channels=self.img_in_channels, 
            feature_size=self.img_feature_size,
            dropout_path_rate=self.dropout_path_rate
            ).cuda()
        
        self.encoder_stride = 32 # encoder_stride

        self.image_decoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.image_encoder.num_features * 16,
                out_channels=self.encoder_stride ** 3 * 1, kernel_size=1),
            PixelShuffle3D(self.encoder_stride),
        )
        #### Point Cloud ####
        self.pcd_lower_encoder = PointTransformerEncoder(
            block=PointTransformerBlock, 
            blocks=self.pcd_blocks,
            in_channels=pcd_in_channels
            ).cuda()
        
        self.pcd_upper_encoder = PointTransformerEncoder(
            block=PointTransformerBlock, 
            blocks=self.pcd_blocks,
            in_channels=pcd_in_channels,
            ).cuda()
        
        in_channels = 512
        hidden_channels = 256
        out_channels = 6
        self.pcd_lower_decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.pcd_upper_decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def random_mask(self, image):
        B, C, H, W, D = image.shape
        # Determine how many 32x32 patches fit along each dimension
        num_patches_h = H // 32
        num_patches_w = W // 32
        num_patches_d = D // 32
        random_grid = np.random.randint(0, 2, size=(num_patches_h, num_patches_w, num_patches_d))

        mask = np.kron(random_grid, np.ones((16, 16, 16)))
        mask = mask[None, None, :H, :W, :D]
        mask = torch.from_numpy(mask).float().to(image.device) 
        return mask

    def forward(self, image, label, pcd_lower, pcd_upper):
        #### Masking and Encoding ####
        ######## Image ########
        image_mask = self.random_mask(image)
        image_features = self.image_encoder.forward_mask(image, image_mask) # latent, feature list

        ######## Point Cloud ########
        B, C, N = pcd_lower.shape
        pcd_lower_x0 = pcd_lower.reshape(-1, 6) # (B * N, C)
        pcd_lower_p0 = pcd_lower[:, 3].reshape(-1, 3).contiguous() # (B * N, 3)
        pcd_lower_o0 = torch.arange(1, B + 1, dtype=torch.int32, device=pcd_lower.device) * N # (B, ) (1, 2, ..., B)
        mask_idx, visible_idx = random_mask_points(pcd_lower_p0, pcd_lower_o0, self.mask_ratio)
        pcd_lower_x0[mask_idx] = 0 # Zero out the features of masked points.
        pcd_lower_z = self.pcd_lower_encoder(pcd_lower_p0, pcd_lower_x0, pcd_lower_o0)
        
        pcd_upper_x0 = pcd_upper.reshape(-1, 6) # (B * N, C)
        pcd_upper_p0 = pcd_upper[:, 3].reshape(-1, 3).contiguous() # (B * N, 3)
        pcd_upper_o0 = torch.arange(1, B + 1, dtype=torch.int32, device=pcd_upper.device) * N # (B, ) (1, 2, ..., B)
        mask_idx, visible_idx = random_mask_points(pcd_lower_p0, pcd_upper_o0, self.mask_ratio)
        pcd_upper_x0[mask_idx] = 0 # Zero out the features of masked points.
        pcd_upper_z = self.pcd_upper_encoder(pcd_upper_p0, pcd_upper_x0, pcd_upper_o0)
        
        #### Reconstruction #### 
        ######## Image ########
        image_rec = self.image_decoder(image_features[-1])
        mask = mask.repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).repeat_interleave(self.patch_size, 4).contiguous()
        loss_recon = F.l1_loss(image_rec, image, reduction='none')
        image_loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        mask = mask.bool()
        image_rec = image * (~ mask) + image_rec * mask
        
        ######## Point Cloud ########
        pcd_lower_latent_full = pcd_lower_z[-1]['f_out']  # (N // 256, D)
        pcd_lower_p_coarse = pcd_lower_z[-1]['p_out']  # (N // 256, D)
        pcd_lower_o0_coarse = pcd_lower_z[-1]['offset']  # (B,) * N

        pcd_upper_latent_full = pcd_upper_z[-1]['f_out']  # (N // 256, D)
        pcd_upper_p_coarse = pcd_upper_z[-1]['p_out']  # (N // 256, D)
        pcd_upper_o0_coarse = pcd_upper_z[-1]['offset']  # (B,) * N
        
        pcd_lower_f_dense = pointops.interpolation(
            xyz=pcd_lower_p_coarse, 
            new_xyz=pcd_lower_p0, 
            feat=pcd_lower_latent_full, 
            offset=pcd_lower_o0_coarse, 
            new_offset=pcd_lower_o0
        ) # (N // 256, 512) -> (N, 512)
        
        pcd_upper_f_dense = pointops.interpolation(
            xyz=pcd_upper_p_coarse, 
            new_xyz=pcd_upper_p0, 
            feat=pcd_upper_latent_full, 
            offset=pcd_upper_o0_coarse, 
            new_offset=pcd_upper_o0
        ) # (N // 256, 512) -> (N, 512)
        pcd_lower_rec = self.pcd_lower_decoder(pcd_lower_f_dense) # (N, 512) -> (N, 6)
        pcd_upper_rec = self.pcd_lower_decoder(pcd_upper_f_dense) # (N, 512) -> (N, 6)
        
        pcd_lower_loss = F.l1_loss(pcd_lower_rec, pcd_lower)
        pcd_upper_loss = F.l1_loss(pcd_upper_rec, pcd_upper)
        
        total_loss = image_loss + pcd_lower_loss + pcd_upper_loss
        
        return total_loss, image_loss, pcd_lower_loss, pcd_upper_loss, image_rec, pcd_lower_rec, pcd_upper_rec


if __name__ == '__main__':
    image_file = 'sample_data/DBT1_0001_0000.nii.gz'
    label_file = 'sample_data/DBT1_0001.nii.gz'
    pcd_lower_file = 'sample_data/DBT1_0001_lower_sampled_points.npy'
    pcd_upper_file = 'sample_data/DBT1_0001_upper_sampled_points.npy'
    import nibabel as nib

    image_data = nib.load(image_file)
    image = image_data.get_fdata()
    image = image[None, None, :128, :128, :32]
    print(image.shape)

    label_data = nib.load(label_file)
    label = label_data.get_fdata()
    label = label[None, None, :128, :128, :32]

    pcd_lower = np.load(pcd_lower_file)
    pcd_upper = np.load(pcd_upper_file)

    image = torch.from_numpy(image).float().cuda()
    label = torch.from_numpy(label).float().cuda()
    pcd_lower = torch.from_numpy(pcd_lower).float().cuda().unsqueeze(0).permute(0, 2, 1)
    pcd_upper = torch.from_numpy(pcd_upper).float().cuda().unsqueeze(0).permute(0, 2, 1)

    print(image.shape, pcd_lower.shape)
    model = ImagePCDMIMModel(
                img_spatial_dim=3,
                img_in_channels=1,
                img_feature_size=48,
                dropout_path_rate=0.1,
                pcd_in_channels=6,
                pcd_blocks=[2, 3, 4, 6, 3],
                pcd_stride=[1, 4, 4, 4, 4],
                pcd_nstride=[2, 2, 2, 2],
                pcd_nsample=[36, 24, 24, 24, 24],
                pcd_planes=[32, 64, 128, 256, 512],
                contain_weight=False,
                mask_ratio=0.3,
                crop_sample_size=3072,
                device='cuda'
            )
    model = model.cuda()
    output = model(image, label, pcd_lower, pcd_upper)