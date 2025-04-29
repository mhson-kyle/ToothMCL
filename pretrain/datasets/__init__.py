from torchvision import transforms
from torch.utils.data import DataLoader

from .dataset import *


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config.get('data_dir')
        self.json_file = self.config.get('json_file')

    def get_train_dataloader(self):
        train_ds = TeethPairedDataset(
            data_dir=self.data_dir, 
            json_file=self.json_file, 
            split='train',
            transform=transforms.Compose([
                RandomScaling(scale_range=(0.8, 1.2)),
                RandomRotation(angle_range=(-10, 10)),
                RandomTranslation(trans_range=(-10, 10)),
                RandomCrop(output_size=(
                    self.config.get('roi_x', 256), 
                    self.config.get('roi_y', 256), 
                    self.config.get('roi_z', 64))
                ),
                ToTensor()
            ]),
            data_scale=self.config.get('data_scale', 1.0),
        )
        
        # train_sampler = Sampler(train_ds) if self.config.get('distributed') else None
        self.train_dataloader = DataLoader(
            dataset=train_ds, 
            batch_size=self.config.get('train_batch_size', 1),
            shuffle=True,
            num_workers=self.config.get('num_workers', 16),
            sampler=None,
            pin_memory=True
        )
        if self.config.get('rank') == 0:
            print(f'Train Dataloader: {len(self.train_dataloader)}')
        return self.train_dataloader

    def get_val_dataloader(self):
        val_ds = TeethPairedDataset(
            data_dir=self.data_dir, 
            json_file=self.json_file, 
            split='val',
            transform=transforms.Compose([
                RandomCrop(output_size=(
                    self.config.get('roi_x'), 
                    self.config.get('roi_y'), 
                    self.config.get('roi_z'))
                ),
                ToTensor()
            ])
        )   

        self.val_dataloader = DataLoader(
            dataset=val_ds,
            batch_size=self.config.get('val_batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('num_workers', 16),
            pin_memory=True
        )
        return self.val_dataloader


if __name__ == "__main__":
    configs = {
        'data_dir': '/ssddata1/data/mhson/Dental/DATA/nnUNet/nnUNet_raw/Dataset123_DBT',
        'json_file': './datasets/paired_datalist.json',
        'roi_x': 256,
        'roi_y': 256,
        'roi_z': 64,
        'train_batch_size': 1,
        'val_batch_size': 1,
        'num_workers': 16,
        'distributed': False,
        'rank': 0
    }
    dm = DatasetManager(configs)
    train_dl = dm.get_train_dataloader()
    sample_batch = next(iter(train_dl))
    print(sample_batch['cbct_image'].shape)