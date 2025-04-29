import os
import numpy as np
import json

import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


class Teeth3DSDataset(Dataset):
    def __init__(self, 
                 data_dir='/data/kyle/Dental/DATA/IOS', 
                 json_file='./datasets',
                 split='train',
                 jaw='lower',
                 transform=None):
        self.split = split
        self.transform = transform
        
        self.data_dir = os.path.join(data_dir, '3DS/fps_sampled_points')
        self.json_file = os.path.join(json_file, '3DS_split.json')
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        self.json_data = self.json_data[self.split]
        self.data_list = [os.path.join(self.data_dir, data[jaw]) for data in self.json_data]
         
        self.data_dir = os.path.join(data_dir, 'TADPM/fps_sampled_points')
        self.json_file = os.path.join(json_file, 'TADPM_split.json')
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        self.json_data = self.json_data[self.split]
        self.data_list += [os.path.join(data_dir, data[jaw]) for data in self.json_data]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pcd = np.load(self.data_list[idx])

        pcd_feats = pcd.copy()[:, :6].astype("float32")
        seg_label = pcd.copy()[:, 6:].astype("int")
        seg_label -= 1 # -1 means gingiva, 0 means first incisor...
        
        if self.transform:
            pcd_feats = self.transform(pcd_feats)

        pcd_feats = torch.from_numpy(pcd_feats) # (N, 6)
        pcd_feats = pcd_feats.permute(1, 0) # (6, N)

        seg_label = torch.from_numpy(seg_label) # (N, 1)
        seg_label = seg_label.permute(1, 0) # (1, N)

        output = {
            "feat": pcd_feats,
            "gt_seg_label": seg_label,
            "casename": os.path.basename(self.data_list[idx])
        }
        return output


class RandomScaling:
    def __init__(self, scale_range):
        self.scale_range = scale_range
        assert self.scale_range[1] > self.scale_range[0]

    def __call__(self, vert_arr):
        self.scale_val = np.random.uniform(self.scale_range[0], self.scale_range[1])
        vert_arr[:, :3] *= self.scale_val
        return vert_arr


class RandomRotation:
    def __init__(self, angle_range, angle_axis='rand'):
        self.angle_range = angle_range
        self.angle_axis = angle_axis
        
        assert self.angle_range[1] > self.angle_range[0]
        if self.angle_axis == "rand":
            self.angle_axis_val = np.random.rand(3)
            self.angle_axis_val /= np.linalg.norm(self.angle_axis_val)
        elif self.angle_axis == "fixed":
            self.angle_axis_val = np.array([0, 0, 1])
        elif self.angle_axis == "pca":
            pass  # PCA axis is determined during augmentation
        else:
            raise ValueError("Invalid angle_axis parameter")

    def __call__(self, vert_arr):
        self.rot_val = np.random.uniform(self.angle_range[0], self.angle_range[1])
        if self.angle_axis == "pca":
            pca_axis = PCA(n_components=3).fit(vert_arr[:, :3]).components_
            rotation_mat = pca_axis
            flip_rand = np.random.choice([-1, 1], size=3)
            pca_axis = pca_axis * flip_rand[:, np.newaxis]
        else:
            rotation_mat = self.get_rotation_matrix()

        if isinstance(vert_arr, torch.Tensor):
            rotation_mat = torch.from_numpy(rotation_mat).float().cuda()

        vert_arr[:, :3] = np.dot(vert_arr[:, :3], rotation_mat.T)

        if vert_arr.shape[1] == 6:
            vert_arr[:, 3:] = np.dot(vert_arr[:, 3:], rotation_mat.T)

        return vert_arr

    def get_rotation_matrix(self):
        axis = self.angle_axis_val / np.linalg.norm(self.angle_axis_val)
        angle = np.deg2rad(self.rot_val)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        ux, uy, uz = axis

        rotation_mat = np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])
        return rotation_mat


class RandomTranslation:
    def __init__(self, trans_range):
        self.trans_range = trans_range
        assert self.trans_range[1] > self.trans_range[0]

    def __call__(self, vert_arr):
        self.trans_val = np.random.uniform(self.trans_range[0], self.trans_range[1], size=(1, 3))
        vert_arr[:, :3] += self.trans_val
        return vert_arr

if __name__ == "__main__":
    dataset = Teeth3DSDataset(
        data_dir='/data/kyle/Dental/DATA/IOS', 
        json_file='./datasets',
        split='train',
        jaw='lower',
        transform=None
    )
    print(len(dataset))
    # print(dataset[0])