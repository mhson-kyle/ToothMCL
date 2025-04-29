import torch.nn as nn
import numpy as np

from utils.ops_utils import *
from utils.utils import *
from .cbl_point_transformer.cbl_point_transformer_module import get_model

class GroupingNetworkModule(nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        class_num = 9
        # Tooth Grouping Module
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=class_num + 1)
        # Tooth Cropping Module
        self.second_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=2).train().cuda()

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        outputs = {}
        # First Instance Centroid Prediction
        if len(inputs) >= 2 and not test:
            # half_seg_label: 0~8
            half_seg_label = inputs[1].clone()
            half_seg_label[half_seg_label >= 9] -= 8
            
            cbl_loss_1, sem_1, offset_1, mask_1, features_1 = self.first_ins_cent_model([inputs[0], half_seg_label])
            outputs.update({
                "cbl_loss_1": cbl_loss_1,
                "sem_1": sem_1,
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": features_1
            })
        else:
            sem_1, offset_1, mask_1, features_1 = self.first_ins_cent_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,
                "offset_1": offset_1,
                "mask_1": mask_1,
                "first_features": features_1
            })
        
        cluster_centroids = []
        if len(inputs) >= 2:
            for b_idx in range(B):
                b_points_coords = torch_to_numpy(inputs[0][b_idx, :3, :]).T
                b_gt_seg_labels = torch_to_numpy(inputs[1][b_idx, :, :].view(-1))
                
                # b_gt_seg_labels: 0~16
                unique_toothid = np.unique(b_gt_seg_labels)
                temp_centroids = []
                for tooth_id in unique_toothid:
                    # Skip background (Gingiva)
                    if tooth_id == -1:
                        continue
                    temp_centroids.append(np.mean(b_points_coords[b_gt_seg_labels == tooth_id], axis=0))
                    # temp_centroids.append((b_points_coords[b_gt_seg_labels == tooth_id].mean(axis=0))
                cluster_centroids.append(temp_centroids)
        else:
            for b_idx in range(B):
                b_points_coords = torch_to_numpy(inputs[b_idx][b_idx, :3, :]).T
                whole_pd_sem_1 = torch_to_numpy(sem_1)[b_idx, :, :].T
                whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
                whole_offset_1 = torch_to_numpy(offset_1)[b_idx, :, :].T
                b_moved_points = b_points_coords + whole_offset_1
                b_fg_moved_points = b_moved_points[whole_cls_1.reshape(-1) != 0, :]
                fg_points_labels_ls = get_clustering_labels(b_moved_points, whole_cls_1)
                
                unique_toothid = np.unique(fg_points_labels_ls)
                temp_centroids = []
                for tooth_id in unique_toothid:
                    temp_centroids.append(np.mean(b_fg_moved_points[fg_points_labels_ls == tooth_id, :], axis=0))
                cluster_centroids.append(temp_centroids)
        
        xyz = torch_to_numpy(inputs[0][:, :3, :].permute(0, 2, 1))
        nn_crop_indexes = get_nearest_neighbor_idx(
            org_xyz=xyz, 
            sampled_clusters=cluster_centroids, 
            crop_num=self.config["model_parameter"]["crop_sample_size"]
        )
        cropped_feature_ls = get_indexed_features(
            features=inputs[0], 
            cropped_indexes=nn_crop_indexes
        )
        
        if len(inputs) >= 2:
            cluster_gt_seg_label = get_indexed_features(inputs[1], nn_crop_indexes)

        cropped_feature_ls = centering_object(cropped_feature_ls)
        
        # Second Instance Centroid Prediction
        if len(inputs) >= 2 and not test:
            cluster_gt_seg_label[cluster_gt_seg_label >= 0] = 0
            outputs["cluster_gt_seg_label"] = cluster_gt_seg_label
            cbl_loss_2, sem_2, offset_2, mask_2, features_2 = self.second_ins_cent_model([cropped_feature_ls, cluster_gt_seg_label])

            outputs.update({
                "cbl_loss_2": cbl_loss_2,
                "sem_2": sem_2,
                "offset_2": offset_2,
                "mask_2": mask_2,
            })
        else:
            # When inference gt_seg_label is not given
            sem_2, offset_2, mask_2, features_2 = self.second_ins_cent_model([cropped_feature_ls])

            outputs.update({
                "sem_2": sem_2,
                "offset_2": offset_2,
                "mask_2": mask_2,
            })

        outputs["cropped_feature_ls"] = cropped_feature_ls
        outputs["nn_crop_indexes"] =  nn_crop_indexes

        return outputs