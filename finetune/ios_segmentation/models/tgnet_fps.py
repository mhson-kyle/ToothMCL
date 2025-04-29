from utils import tgn_loss
from utils.utils import LossMap
from .grouping_network_module import GroupingNetworkModule

class TGNet_fps(GroupingNetworkModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def get_loss(self, 
                 offset_1, offset_2, 
                 sem_1, sem_2, 
                 mask_1, mask_2, 
                 gt_seg_label_1, gt_seg_label_2, 
                 input_coords, cropped_coords):
        
        half_seg_label = gt_seg_label_1.clone()
        half_seg_label[half_seg_label >= 9] -= 8

        gt_seg_label_2[gt_seg_label_2 >= 0] = 0
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, half_seg_label, 9)
        tooth_class_loss_2 = tgn_loss.tooth_class_loss(sem_2, gt_seg_label_2, 2)

        offset_1_loss, offset_1_dir_loss = tgn_loss.batch_center_offset_loss(offset_1, input_coords, gt_seg_label_1)
        
        chamf_1_loss = tgn_loss.batch_chamfer_distance_loss(offset_1, input_coords, gt_seg_label_1)
        
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, self.config["tr_set"]["loss"]["tooth_class_loss_1"]),
            "tooth_class_loss_2": (tooth_class_loss_2, self.config["tr_set"]["loss"]["tooth_class_loss_2"]),
            "offset_1_loss": (offset_1_loss, self.config["tr_set"]["loss"]["offset_1_loss"]),
            "offset_1_dir_loss": (offset_1_dir_loss, self.config["tr_set"]["loss"]["offset_1_dir_loss"]),
            "chamf_1_loss" : (chamf_1_loss, self.config["tr_set"]["loss"]["chamf_1_loss"])
        }

    def step(self, batch_item):
        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:, :3, :].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        output = self.forward([points, seg_label])
        loss_meter = LossMap()
        
        loss_meter.add_loss_by_dict(
            loss_dict=self.get_loss(
                offset_1=output["offset_1"], 
                offset_2=output["offset_2"], 
                sem_1=output["sem_1"], 
                sem_2=output["sem_2"], 
                mask_1=output["mask_1"], 
                mask_2=output["mask_2"], 
                gt_seg_label_1=seg_label, 
                gt_seg_label_2=output["cluster_gt_seg_label"], 
                input_coords=l0_xyz, 
                cropped_coords=output["cropped_feature_ls"][:, :3, :]
            )
        )
        
        loss_meter.add_loss(
            name="cbl_loss_1", 
            value=output["cbl_loss_1"].sum(), 
            weight=self.config["tr_set"]["loss"]["cbl_loss_1"])

        loss_meter.add_loss(
            name="cbl_loss_2", 
            value=output["cbl_loss_2"].sum(), 
            weight=self.config["tr_set"]["loss"]["cbl_loss_2"])

        return loss_meter