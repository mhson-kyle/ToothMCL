import os
from glob import glob

import torch
import numpy as np
from sklearn.neighbors import KDTree

from utils import tgn_loss
from .grouping_network_module import GroupingNetworkModule
from utils.ops_utils import clustering_points
from utils.utils import LossMap, torch_to_numpy, load_json, read_txt_obj_ls, resample_pcd, count_unique_by_row


class TGNet_bdl(GroupingNetworkModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.base_model = GroupingNetworkModule(config["fps_model_info"])
        self.base_model.load_state_dict(torch.load(self.config["fps_model_info"]["load_ckpt_path"]+".h5"))
        self.base_model.cuda()

        self.stl_path_map = {}
        for dir_path in [
            x[0] for x in os.walk(config["boundary_sampling_info"]["orginal_data_obj_path"])
            ][1:]:
            for stl_path in glob(os.path.join(dir_path, "*.obj")):
                self.stl_path_map[os.path.basename(stl_path).split(".")[0]] = stl_path

        self.json_path_map = {}
        for dir_path in [
            x[0] for x in os.walk(config["boundary_sampling_info"]["orginal_data_json_path"])
            ][1:]:
            for json_path in glob(os.path.join(dir_path,"*.json")):
                self.json_path_map[os.path.basename(json_path).split(".")[0]] = json_path

        self.Y_AXIS_MAX = 33.15232091532151
        self.Y_AXIS_MIN = -36.9843781139949
        
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

    def get_points_cluster_labels(self, batch_item):
        """
        Args:
            batch_idx (_type_): _description_
            batch_item (batch_item): _description_

        Returns:
            labels: N
        """

        points = batch_item["feat"].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        gt_seg_label = torch_to_numpy(batch_item["gt_seg_label"])
        with torch.no_grad():
            output = self.base_model([points, seg_label])
        results = {}

        crop_num = output["sem_2"].shape[0]
        org_xyz_cpu = torch_to_numpy(points)[0, :3, :].T

        whole_pd_mask_2 = torch.zeros((points.shape[2], 2)).cuda()
        whole_pd_mask_count_2 = torch.zeros(points.shape[2]).cuda()
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1, 0) # 3072,17
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            whole_pd_mask_2[inside_crop_idx] += pd_mask
            whole_pd_mask_count_2[inside_crop_idx] += 1
        
        whole_pd_mask_2 = torch_to_numpy(whole_pd_mask_2)
        whole_mask_2 = np.argmax(whole_pd_mask_2, axis=1)
        full_masked_points_2 = np.concatenate([org_xyz_cpu, whole_mask_2.reshape(-1, 1)], axis=1)

        results["sem_2"] = {}
        results["sem_2"]["full_masked_points"] = full_masked_points_2
        results["sem_2"]["whole_pd_mask"] = whole_pd_mask_2

        moved_points_cpu = org_xyz_cpu + torch_to_numpy(output["offset_1"])[0, :3,:].T
        fg_moved_points = moved_points_cpu[results["sem_2"]["full_masked_points"][:, 3] == 1, :]

        num_of_clusters = []
        for b_idx in range(1):
            num_of_clusters.append(len(np.unique(gt_seg_label[b_idx, :])) - 1)

        cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = clustering_points(
            [fg_moved_points], 
            method="kmeans", 
            num_of_clusters=num_of_clusters
        )
        
        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels[:] = -1
        points_ins_labels[np.where(results["sem_2"]["full_masked_points"][:, 3])] = fg_points_labels_ls[b_idx]
        
        full_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1, 1)], axis=1)
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points

        results["first_features"] = output["first_features"]
        return results

    def load_json_mesh(self, casename):
        loaded_json = load_json(self.json_path_map[casename])
        labels = np.array(loaded_json['labels']).reshape(-1,1)
        if loaded_json['jaw'] == 'lower':
            labels -= 20
        labels[labels // 10 == 1] %= 10
        labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
        labels[labels < 0] = 0

        vertices = read_txt_obj_ls(self.stl_path_map[casename], ret_mesh=False)[0]
        vertices[:, :3] -= np.mean(vertices[:, :3], axis=0)
        vertices[:, :3] = ((vertices[:, :3] - self.Y_AXIS_MIN) / (self.Y_AXIS_MAX - self.Y_AXIS_MIN)) * 2 - 1
        vertices = vertices.astype("float32")
        labels -= 1
        labels = labels.astype(int)
        return vertices, labels.reshape(-1, 1)

    def get_boundary_sampled_points(self, batch_item):
        casename = os.path.basename(batch_item["mesh_path"][0]).split(".")[0]
        cache_path = os.path.join(self.config["boundary_sampling_info"]["bdl_cache_path"], f"{casename}.npy")

        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            features, gt_seg_label = self.load_json_mesh(casename)
            if (features.shape[0] < self.config["boundary_sampling_info"]["num_of_all_points"]):
                return batch_item["feat"], batch_item["gt_seg_label"]
            results = self.get_points_cluster_labels(batch_item) # N
            points_labels = results["ins"]["full_ins_labeled_points"][:, 3]
            xyz_cpu = torch_to_numpy(batch_item["feat"])[0, :3, :].T # N, 3

            tree = KDTree(xyz_cpu, leaf_size=2)

            if batch_item["aug_obj"][0]:
                auged_features = batch_item["aug_obj"][0].run(features.copy())
            else:
                auged_features = features.copy()

            bd_labels = np.zeros(auged_features.shape[0]) # N
            near_points = tree.query(
                X=auged_features[:,:3], 
                k=40, 
                return_distance=False
            )

            labels_arr = points_labels[near_points]
            label_counts = count_unique_by_row(labels_arr)
            label_ratio = label_counts[:, 0] / 40.
            
            #To change
            bd_labels[label_ratio < self.config["boundary_sampling_info"]["bdl_ratio"]] = 1

            bd_features = features[bd_labels == 1, :]
            bd_auged_features = auged_features[bd_labels == 1, :]
            bd_gt_seg_label = gt_seg_label[bd_labels == 1, :]

            bd_features, bd_auged_features, bd_gt_seg_label = resample_pcd(
                pcd_ls=[
                    bd_features,
                    bd_auged_features, 
                    bd_gt_seg_label
                    ], 
                n=self.config["boundary_sampling_info"]["num_of_bdl_points"], 
                method="uniformly"
            )

            non_bd_features = features[bd_labels == 0, :]
            non_bd_auged_features = auged_features[bd_labels == 0, :]
            non_bd_gt_seg_label = gt_seg_label[bd_labels == 0, :]
            
            non_bd_auged_features, non_bd_gt_seg_label, non_bd_features = resample_pcd(
                pcd_ls=[
                    non_bd_features,
                    non_bd_auged_features, 
                    non_bd_gt_seg_label
                    ], 
                n=self.config["boundary_sampling_info"]["num_of_all_points"] - bd_auged_features.shape[0], 
                method="fps"
            )

            sampled_features = np.concatenate([bd_features, non_bd_features], axis=0)
            sampled_auged_features = np.concatenate([bd_auged_features, non_bd_auged_features], axis=0)
            sampled_gt_seg_label = np.concatenate([bd_gt_seg_label, non_bd_gt_seg_label], axis=0)
            np.save(cache_path, np.concatenate([sampled_features, sampled_gt_seg_label], axis=1))      
        else:
            cached_arr = np.load(cache_path)
            sampled_auged_features, sampled_gt_seg_label = cached_arr[:, :6], cached_arr[:, 6:]
            if batch_item["aug_obj"][0]:
                sampled_auged_features = batch_item["aug_obj"][0].run(sampled_auged_features.copy())
            else:
                sampled_auged_features = sampled_auged_features.copy()
            sampled_auged_features = sampled_auged_features.astype('float32')
            sampled_gt_seg_label = sampled_gt_seg_label.astype(int)

        return torch.from_numpy(sampled_auged_features.T.reshape(1,sampled_auged_features.shape[1], -1)), torch.from_numpy(sampled_gt_seg_label.T.reshape(1, 1, -1))

    def step(self, batch_item):
        points, seg_label = self.get_boundary_sampled_points(batch_item)

        points = points.cuda()
        l0_xyz = points[:, :3, :].cuda()
        seg_label = seg_label.cuda()
        
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

