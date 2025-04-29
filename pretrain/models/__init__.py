from .contrastive_head import ImagePCDModel
from .mim_head import ImagePCDMIMModel

class ModelManager:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
    def get_model(self):
        if self.config.get('model') == 'ImagePCDModel':
            self.model = ImagePCDModel(
                img_spatial_dim=self.config.get('spatial_dim', 3),
                img_in_channels=self.config.get('in_channels', 1),
                img_feature_size=self.config.get('feature_size', 48),
                dropout_path_rate=self.config.get('dropout_path_rate', 0.1),
                pcd_in_channels=self.config.get('input_feat', 6),
                pcd_blocks=self.config.get('blocks', [2, 3, 4, 6, 3]),
                pcd_stride=self.config.get('stride', [1, 4, 4, 4, 4]),
                pcd_nstride=self.config.get('nstride', [2, 2, 2, 2]),
                pcd_nsample=self.config.get('nsample', [36, 24, 24, 24, 24]),
                pcd_planes=self.config.get('planes', [32, 64, 128, 256, 512]),
                contain_weight=self.config.get('contain_weight', False),
                crop_sample_size=self.config.get('crop_sample_size', 3072),
                device=self.device
            )
        elif self.config.get('model') == 'ImagePCDMIMModel':
            self.model = ImagePCDMIMModel(
                img_spatial_dim=self.config.get('spatial_dim', 3),
                img_in_channels=self.config.get('in_channels', 1),
                img_feature_size=self.config.get('feature_size', 48),
                dropout_path_rate=self.config.get('dropout_path_rate', 0.1),
                pcd_in_channels=self.config.get('input_feat', 6),
                pcd_blocks=self.config.get('blocks', [2, 3, 4, 6, 3]),
                pcd_stride=self.config.get('stride', [1, 4, 4, 4, 4]),
                pcd_nstride=self.config.get('nstride', [2, 2, 2, 2]),
                pcd_nsample=self.config.get('nsample', [36, 24, 24, 24, 24]),
                pcd_planes=self.config.get('planes', [32, 64, 128, 256, 512]),
                contain_weight=self.config.get('contain_weight', False),
                mask_ratio=0.3,
                crop_sample_size=self.config.get('crop_sample_size', 3072),
                device=self.device
            )
        elif self.config.get('model') == 'SwinEncoder':
            from .pt_blocks import PointTransformerBlock
            from .pt_encoder import PointTransformerSeg
            self.model = PointTransformerSeg(block=PointTransformerBlock, 
                                    blocks=[2, 3, 4, 6, 3],
                                    in_channels=1,
                                    num_classes=17)
        else:
            raise ValueError(f"Model {self.config.get('model')} not supported")
        if self.config.get('rank') == 0:
            pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total parameters count", f"{pytorch_total_params:,}")
        return self.model
