import torch
from .datasets import Teeth3DSDataset, RandomScaling, RandomRotation, RandomTranslation
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

def collate_fn(batch):
    output = {}

    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])
    
    for output_key in output.keys():
        if output_key in ["feat", "gt_seg_label", "uniform_feat", "uniform_gt_seg_label"]:
            output[output_key] = torch.stack(output[output_key])
    return output

class DatasetManager:
    def __init__(self, config):
        self.config = config

    def get_dataloader(self):
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        return train_dataloader, val_dataloader

    def get_train_dataloader(self):
        train_dataset = Teeth3DSDataset(
            data_dir=self.config.get('data_dir'), 
            json_file=self.config.get('dataset_json'),
            split='train',
            transform=Compose([
                RandomScaling([0.85, 1.15]), 
                RandomRotation([-30, 30], 'rand'), 
                RandomTranslation([-0.2, 0.2])
            ]),
            jaw=self.config.get('jaw')
        )
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            shuffle=True, 
            batch_size=self.config.get('train_batch_size'), 
            collate_fn=collate_fn
        )
        
        print('Training Set: ', len(train_dataset))
        return train_dataloader
    
    def get_val_dataloader(self):
        val_dataset = Teeth3DSDataset(
            data_dir=self.config.get('data_dir'),
            json_file=self.config.get('dataset_json'),
            split='validation',
            jaw=self.config.get('jaw')
        )
        
        val_dataloader = DataLoader(
            dataset=val_dataset, 
            shuffle=False, 
            batch_size=self.config.get('val_batch_size'), 
            collate_fn=collate_fn
        )
        
        print('Validation Set: ', len(val_dataset))
        return val_dataloader
    

if __name__ == '__main__':
    config = {
        "DATA": {
            "data_dir": "/data/kyle/Dental/DATA/IOS/",
            "json_file": "./datasets/"
        },
        "TRAIN": {
            "train_batch_size": 2,
            "val_batch_size": 2
        }
    }
    dataset_manager = DatasetManager(config)
    train_dataloader, val_dataloader = dataset_manager.get_dataloader()
