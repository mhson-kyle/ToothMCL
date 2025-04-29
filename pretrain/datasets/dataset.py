import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.decomposition import PCA
import einops

from utils.utils import load_niigz, load_json


class TeethPairedDataset(Dataset):
    def __init__(self, 
                 data_dir='/data/kyle/Dental/DATA/DBT', 
                 json_file='dbt.json', 
                 split='train',
                 data_scale=1,
                 transform=None):

        self.data_dir = data_dir
        self.split = split
        self.data_list = load_json(json_file)[self.split]
        num_samples = int(len(self.data_list) * data_scale)
        np.random.seed(0)
        np.random.shuffle(self.data_list)
        self.data_list = self.data_list[:num_samples]
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        case_dict = self.data_list[idx]
        cbct_image_file = case_dict['cbct_image']
        # cbct_label_file = case_dict['cbct_label']
        ios_lower_file = case_dict['ios_lower']
        ios_upper_file = case_dict['ios_upper']

        self.cbct_image, _ = load_niigz(os.path.join(self.data_dir, cbct_image_file)) # D, H, W
        # self.cbct_label, _ = load_niigz(os.path.join(self.data_dir, cbct_label_file))
        
        self.ios_data_dir = '/ssddata/data/mhson/DATA/IOS/DBT/'
        self.ios_lower_pcd = np.load(os.path.join(self.ios_data_dir, ios_lower_file)) # N, 6
        self.ios_upper_pcd = np.load(os.path.join(self.ios_data_dir, ios_upper_file)) # N, 6
        
        self.ios_lower_pcd = einops.rearrange(self.ios_lower_pcd, 'n c -> c n')
        self.ios_upper_pcd = einops.rearrange(self.ios_upper_pcd, 'n c -> c n')
        self.ios_pcd = np.concatenate([self.ios_lower_pcd, self.ios_upper_pcd], axis=0) # 2N, 6

        self.cbct_image = np.clip(self.cbct_image, 0, 2500) / 2500
        assert np.min(self.cbct_image) >= 0 and np.max(self.cbct_image) <= 1
        
        data = {
            'cbct_image': self.cbct_image,
            'ios_lower_pcd': self.ios_lower_pcd,
            'ios_upper_pcd': self.ios_upper_pcd,
            'casename': os.path.basename(cbct_image_file).split('.')[0].replace('_0000', '')
        }

        if transforms is not None:
            data = self.transform(data)

        return data


class RandomScaling:
    def __init__(self, scale_range):
        self.scale_range = scale_range
        assert self.scale_range[1] > self.scale_range[0]

    def __call__(self, data):
        self.scale_val = np.random.uniform(self.scale_range[0], self.scale_range[1])
        data['ios_lower_pcd'][:, :3] *= self.scale_val
        data['ios_upper_pcd'][:, :3] *= self.scale_val
        return data


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

    def __call__(self, data):
        self.rot_val = np.random.uniform(self.angle_range[0], self.angle_range[1])
        if self.angle_axis == "pca":
            pca_axis = PCA(n_components=3).fit(data['ios_lower_pcd'][:, :3]).components_
            rotation_mat = pca_axis
            flip_rand = np.random.choice([-1, 1], size=3)
            pca_axis = pca_axis * flip_rand[:, np.newaxis]
        else:
            rotation_mat = self.get_rotation_matrix()

        if isinstance(data['ios_lower_pcd'], torch.Tensor):
            rotation_mat = torch.from_numpy(rotation_mat).float().cuda()

        data['ios_lower_pcd'][:, :3] = np.dot(data['ios_lower_pcd'][:, :3], rotation_mat.T)
        data['ios_upper_pcd'][:, :3] = np.dot(data['ios_upper_pcd'][:, :3], rotation_mat.T)
        
        if data['ios_lower_pcd'].shape[1] == 6:
            data['ios_lower_pcd'][:, 3:] = np.dot(data['ios_lower_pcd'][:, 3:], rotation_mat.T)
            data['ios_upper_pcd'][:, 3:] = np.dot(data['ios_upper_pcd'][:, 3:], rotation_mat.T)
        
        return data

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

    def __call__(self, data):
        self.trans_val = np.random.uniform(self.trans_range[0], self.trans_range[1], size=(1, 3))
        data['ios_lower_pcd'][:, :3] += self.trans_val
        data['ios_upper_pcd'][:, :3] += self.trans_val
        return data


class RandomCrop:
    """
    Randomly crop a 3D image (depth, height, width) to the given output size.
    """
    def __init__(self, output_size: Tuple[int, int, int]) -> None:
        """
        Args:
            output_size (tuple): Desired output size (depth, height, width).
        """
        if not isinstance(output_size, tuple) or len(output_size) != 3:
            raise ValueError("output_size should be a tuple of three integers (depth, height, width)")
        
        self.output_size = output_size
    def __call__(self, data):
        image = data['cbct_image']
        # label = data['cbct_label']

        # pad the data if necessary
        if image.shape[0] <= self.output_size[0] or \
            image.shape[1] <= self.output_size[1] or \
                image.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], 
                      h1:h1 + self.output_size[1], 
                      d1:d1 + self.output_size[2]]
        # label = label[w1:w1 + self.output_size[0],
        #               h1:h1 + self.output_size[1],
        #               d1:d1 + self.output_size[2]]
        
        data['cbct_image'] = image
        # data['cbct_label'] = label
        
        return data


class ToTensor(object):
    """
    Convert ndarrays in data to Tensors.
    """
    def __call__(self, data):
        data['cbct_image'] = torch.from_numpy(data['cbct_image']).unsqueeze(0).float()
        data['ios_lower_pcd'] = torch.from_numpy(data['ios_lower_pcd']).float()
        data['ios_upper_pcd'] = torch.from_numpy(data['ios_upper_pcd']).float()
        return data



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import Sampler
    
    dataset = TeethPairedDataset(
        data_dir='/ssddata1/data/mhson/Dental/DATA/nnUNet/nnUNet_raw/Dataset123_DBT', 
        json_file='./datasets/paired_datalist.json', 
        split='train',
        transform=transforms.Compose([
            RandomScaling(scale_range=(0.8, 1.2)),
            RandomRotation(angle_range=(-10, 10)),
            RandomTranslation(trans_range=(-10, 10)),
            RandomCrop(output_size=(256, 256, 64)),
            ToTensor()
            ])
        )
    train_sampler = None
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=16,
        sampler=train_sampler,
        pin_memory=True
        )
    print(f'Train Dataloader: {len(dataloader)}')
    sample_batch = next(iter(dataloader))
    # data = dataset[1400]
    print(sample_batch.keys())
    print(sample_batch['cbct_image'].shape)
    print(sample_batch['ios_lower_pcd'].shape)
    print(sample_batch['ios_upper_pcd'].shape)
    print(sample_batch['casename'])
    
    