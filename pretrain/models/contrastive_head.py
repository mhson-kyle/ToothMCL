import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import einops

from models.pt_blocks import PointTransformerBlock
from models.pt_encoder import PointTransformerEncoder
from models.swin_encoder import SwinEncoder


class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class ImagePCDModel(nn.Module):
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
                 device='cuda'):
        super(ImagePCDModel, self).__init__()
        
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

        self.image_encoder = SwinEncoder(spatial_dims=self.img_spatial_dim, 
                                         in_channels=self.img_in_channels, 
                                         feature_size=self.img_feature_size,
                                         dropout_path_rate=self.dropout_path_rate)
        
        self.pcd_lower_encoder = PointTransformerEncoder(
            block=PointTransformerBlock, 
            blocks=self.pcd_blocks,
            in_channels=pcd_in_channels
            )
        
        self.pcd_upper_encoder = PointTransformerEncoder(
            block=PointTransformerBlock, 
            blocks=self.pcd_blocks,
            in_channels=pcd_in_channels,
            )
        
        # self.projection_image = projection_head(in_dim=768, hidden_dim=2048, out_dim=512)
        # self.projection_pcd = projection_head(in_dim=512, hidden_dim=2048, out_dim=512)
        self.projection_image = nn.Conv1d(768, 512, 1)
        self.projection_pcd = nn.Conv1d(512, 512, 1)
        INITIAL_SCALE = (512 ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * INITIAL_SCALE)
        self.logit_bias  = nn.Parameter(torch.zeros([]))  # or initialize differently


    def forward(self, image, image_label, pcd_lower, pcd_upper, pcd_label):
        image_features = self.image_encoder(image) # latent, feature list
        image_feature = image_features[1][-1] # (B, C, H, W, D) -> (B, feature_size * 16, H // 32, W // 32, D // 32)
        image_feature = einops.rearrange(image_feature, 'b c h w d -> b c (h w d)') # e.g. (B, 768, 8, 8, 2) -> (B, 768, 128)
        
        pcd_lower_features = self.pcd_lower_encoder([pcd_lower]) # 5 Features [point, feature, offset]
        pcd_lower_feature = pcd_lower_features[-1]['f_out'] # (B, N, C) -> (B, N // 256, feature_size (=512),) e.g. (B, 93, 512))
        pcd_lower_feature = einops.rearrange(pcd_lower_feature.unsqueeze(0), 'b n c -> b c n') # (B, 512, 93) -> (B, 512, 93)
        
        pcd_upper_features = self.pcd_upper_encoder([pcd_upper]) # 5 Features [point, feature, offset]
        pcd_upper_feature = pcd_upper_features[-1]['f_out'] # (B, N, C) -> (B, N // 256, feature_size (=512),) e.g. (B, 93, 512))
        pcd_upper_feature = einops.rearrange(pcd_upper_feature.unsqueeze(0), 'b n c -> b c n') # (B, 512, 93) -> (B, 512, 93)

        pcd_feature = torch.cat([pcd_lower_feature, pcd_upper_feature], dim=2) # (B, 512, 93) + (B, 512, 93) -> (B, 512, 186)
        
        image_feature = self.projection_image(image_feature) # (B, 768, 128) -> (B, 512, 128)
        pcd_feature = self.projection_pcd(pcd_feature) # (B, 512, 186) -> (B, 512, 186)
        features = [image_feature, pcd_feature]
        
        # contrastive loss
        device = image_feature.device
        image_label,pcd_label = image_label.to(device), pcd_label.to(device)
        image_loss, pcd_loss, image_pcd_loss, total_loss = self.contrastive_loss(image_feature, pcd_feature, image_label, pcd_label)
        loss = [image_loss, pcd_loss, image_pcd_loss, total_loss]
        return loss, features

    def compute_loss(self, features):
        image_feature, pcd_feature = features
        image_loss, pcd_loss, image_pcd_loss, total_loss = self.contrastive_loss(image_feature, pcd_feature)
        loss = [image_loss, pcd_loss, image_pcd_loss, total_loss]
        return loss

    def contrastive_loss(self, image_feature, pcd_feature, image_label, pcd_label):
        # image_feature (B, C, N) (1, 512, 128)
        # pcd_feature (B, C, N) (1, 512, 93)
        alpha = 0.5
        B, C, N = image_feature.shape
        image_mask = self.build_mask_from_labels(image_label)
        image_mask = image_mask.unsqueeze(0).repeat(B,1,1)  # (B, N, N)
        image_loss = self.intra_modal_loss(image_feature, pos_mask=image_mask)
        
        B, C, N = pcd_feature.shape
        pcd_mask = self.build_mask_from_labels(pcd_label)
        pcd_mask = pcd_mask.unsqueeze(0).repeat(B,1,1)  # (B, N, N)
        pcd_loss = self.intra_modal_loss(pcd_feature, pos_mask=pcd_mask)
        
        image_pcd_loss = self.inter_modal_loss_dense(image_feature, pcd_feature)
        total_loss = image_loss + pcd_loss + alpha * image_pcd_loss
        return image_loss, pcd_loss, image_pcd_loss, total_loss
    
    def inter_modal_loss_dense(self, image_feature, pcd_feature):
        """
        Dense cross-modal logistic loss without Hungarian matching.
        """

        B, C, N_img = image_feature.shape
        _, _, N_pcd = pcd_feature.shape

        total_loss = 0.0

        for b in range(B):
            img_feat = F.normalize(image_feature[b], dim=0)  # (C, N_img)
            pc_feat = F.normalize(pcd_feature[b], dim=0)     # (C, N_pcd)

            sim_matrix = torch.matmul(img_feat.t(), pc_feat) # (N_img, N_pcd)

            # Build z^{ij}: +1 for matching pairs (assume same indices), -1 otherwise
            # Assume N_img == N_pcd == N
            device = sim_matrix.device
            L = min(N_img, N_pcd)
            pos_mask = torch.zeros((N_img, N_pcd), dtype=torch.bool, device=device)
            pos_mask[torch.arange(L), torch.arange(L)] = True    # diagonal for i<min

            # now z = +1 for those, –1 for all others
            z = pos_mask.float().mul(2).sub(1)                   # shape (N_img, N_pcd)

            # proceed with logits of shape (N_img, N_pcd)
            t = self.logit_scale
            b = self.logit_bias
            logits = -t * sim_matrix + b                        # still (N_img, N_pcd)

            # and your logistic loss over all entries:
            A = torch.log1p(torch.exp(z * logits)).neg()        # (N_img, N_pcd)
            sample_loss = - A.mean()

            total_loss += sample_loss

        total_loss = total_loss / B
        return total_loss
    
    def inter_modal_loss(self, image_feature, pcd_feature, margin=0.5, return_sim=False):
        B, C, N_img = image_feature.shape
        _, _, N_pcd = pcd_feature.shape
        total_loss = 0.0
        # Optional storage for visualization
        sim_matrix_for_vis = None

        for b in range(B):
            img_feat = F.normalize(image_feature[b], dim=0)  # (C, N_img)
            pc_feat = F.normalize(pcd_feature[b], dim=0)     # (C, N_pcd)
            sim_matrix = torch.matmul(img_feat.t(), pc_feat) # (N_img, N_pcd)

            # Optionally store the similarity matrix (only for first sample to keep it simple)
            if b == 0 and return_sim:
                sim_matrix_for_vis = sim_matrix.detach().cpu()

            cost_matrix = 1 - sim_matrix
            cost_matrix_np = cost_matrix.detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            matched_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            row_ind_tensor = torch.tensor(row_ind, device=sim_matrix.device)
            col_ind_tensor = torch.tensor(col_ind, device=sim_matrix.device)
            matched_mask[row_ind_tensor, col_ind_tensor] = True

            pos_loss = F.relu(margin - sim_matrix)[matched_mask]
            neg_loss = F.relu(sim_matrix - margin)[~matched_mask]

            pos_loss = pos_loss.mean() if pos_loss.numel() > 0 else torch.tensor(0.0, device=sim_matrix.device)
            neg_loss = neg_loss.mean() if neg_loss.numel() > 0 else torch.tensor(0.0, device=sim_matrix.device)

            sample_loss = pos_loss + neg_loss
            total_loss += sample_loss

        total_loss = total_loss / B
        return (total_loss, sim_matrix_for_vis) if return_sim else total_loss
    
    def build_simple_mask(self, N, device):
        """
        Build (N, N) mask:
        True for same patches (i == j),
        False for different patches (i != j).
        """
        eye = torch.eye(N, dtype=torch.bool, device=device)  # (N, N)
        return eye
    
    def build_neighbor_mask(self, N, window=1, device='cuda'):
        """
        Positive if i==j or i close to j within window size.
        """
        idx = torch.arange(N, device=device)
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window
        return mask
    
    def build_mask_from_labels(self, labels):
        N = len(labels)
        device = labels.device
        
        # Create pairwise comparison matrix
        label_i = labels.unsqueeze(0).expand(N, N)  # (N, N)
        label_j = labels.unsqueeze(1).expand(N, N)  # (N, N)
        
        # Same class = positive pair
        pos_mask = (label_i == label_j) & (labels != -1).unsqueeze(0) & (labels != -1).unsqueeze(1)
        
        return pos_mask
    
    def build_tooth_only_mask(self, labels):
        N = len(labels)
        
        # Only tooth patches (label=1) are positive with each other
        tooth_mask = (labels == 1)
        pos_mask = tooth_mask.unsqueeze(0) & tooth_mask.unsqueeze(1)  # (N, N)
        
        # Ignore patches with label=-1
        valid_mask = (labels != -1)
        pos_mask = pos_mask & valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)
        
        return pos_mask

    def intra_modal_loss(self, features, pos_mask, hard_negative=False):
        """
        features:    (B, C, N)
        pos_mask:    (B, N, N)  boolean mask or {+1,-1} mask
                    True (or +1) for positive pairs, False (or -1) for negative
        Returns:     scalar loss
        """
        B, C, N = features.shape

        # 1) normalize patch vectors
        feat_norm = F.normalize(features.transpose(1, 2), dim=-1)  # (B, N, C)
        sim       = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # (B, N, N)

        # 2) build z^{ij} = +1 / -1
        eye = torch.eye(N, device=sim.device, dtype=torch.bool).unsqueeze(0)
        pos_mask = pos_mask & (~eye)
        z = pos_mask.float().mul(2).sub(1)  # convert boolean to +1/-1

        # 3) logistic alignment: A_{ij} = log[ 1 / (1 + exp( z * ( -t*(sim + b) ) )) ]
        t = torch.clamp(self.logit_scale, 0, 100)  # Clamp scale parameter
        b = self.logit_bias
        logits = -t * sim + b      # (B, N, N)
        A = torch.log(1.0 / (1.0 + torch.exp(z * logits)))  # (B, N, N)  # log(1/(1+exp(z·logits)))

        # 4) for each query patch j, take max over i (hardest match)
        #    shape: (B, N); then average over patches & batch, with negative sign
        if hard_negative:
            per_patch = A.max(dim=1).values  # (B, N)
            loss = -per_patch.mean()
        else:
            loss = -A.mean()

        return loss


if __name__ == '__main__':
    model = ImagePCDModel(
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
        crop_sample_size=3072,
        device='cuda'
    )
    model.cuda()
    cbct_image = torch.randn(1, 1, 256, 256, 64).cuda()
    cbct_image = torch.randn(1, 1, 256, 256, 64).cuda()
    pcd = torch.randn(1, 6, 24000).cuda()
    pcd = torch.randn(1, 6, 24000).cuda()
    output = model(cbct_image, cbct_image, pcd, pcd)