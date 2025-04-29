import torch.nn as nn

from .cbl_point_transformer.cbl_point_transformer_module import get_model
from utils import tgn_loss
from utils.utils import LossMap


class PointTransformerModule(nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_ins_cent_model = get_model(
            **config["model_parameter"], 
            c=config["model_parameter"]["input_feat"], 
            k=16 + 1)

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        outputs = {}
        sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0]])
        outputs.update({
            "sem_1": sem_1,
            "offset_1": offset_1,
            "mask_1": mask_1,
            "first_features": first_features,
            "cls_pred": sem_1
        })
        return outputs
    
    def get_loss(self, gt_seg_label_1, sem_1):
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, self.config["loss"]["tooth_class_loss_1"]),
        }

    def step(self, batch_item):
        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:, :3, :].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        output = self.forward([points, seg_label])
        
        loss_meter = LossMap()
        loss_meter.add_loss_by_dict(
            loss_dict=self.get_loss(
                seg_label, 
                output["sem_1"], 
            )
        )

        return loss_meter