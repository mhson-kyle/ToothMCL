import argparse
import logging
import os
import resource
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm
import wandb

from models import ModelManager
from datasets import DatasetManager
from utils.config import Config, seed
from utils.lr_scheduler import WarmupCosineSchedule
from utils.utils import AverageMeter, distributed_all_gather

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


class Trainer:
    def __init__(self, model, train_dataloader, config, device):
        self.config = config
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader

        # Training parameters
        self.start_epoch = 0
        self.max_epochs = self.config.get('max_epochs')
        self.save_freq = self.config.get('save_freq')
        self.log_freq = self.config.get('log_freq')
        self.patience = self.config.get('patience', 5)
        self.warmup_epochs = self.config.get('warmup_epochs', 5)       
        self.distributed = self.config.get('distributed', False)
        self.rank = self.config.get('rank', 0)
        self.world_size = self.config.get('ngpus_per_node', 1)
 
        self.scaler = GradScaler()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()
        self.amp = self.config.get('amp', True)

        self.early_stop_counter = 0

        if self.config.get('resume', None) != None:
            self.load_checkpoint(self.config.get('resume'))

        self.wandb_on = self.config.get('wandb_on', False)
        if self.wandb_on and self.rank == 0:
            self.wandb_logger = wandb.init(
                entity=self.config.get('entity'),
                project=self.config.get('project'),
                name=self.config.get('run_name'),
                mode='online',
                config=config,
            )

    def log_batch_info(self, epoch, iter_num, loss):
        loss_info = ", ".join([f"{k}: {v:.4f}" for k, v in loss.items()])
        logging.info(
            f"Epoch [{epoch}/{self.max_epochs}], "
            f"Batch [{iter_num}/{len(self.train_dataloader)}], "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
            f"{loss_info}"
        )

    def log_epoch_info(self, epoch, loss_dict, wandb_on=False):
        # Log using logging.info
        loss_info = ", ".join([f"Train {k}: {v:.4f}" for k, v in loss_dict.items()])
        lr = self.optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch [{epoch + 1}/{self.max_epochs}], "
            f"LR: {lr:.6f}, "
            f"{loss_info}, "
        )

        if wandb_on:
            loss_dict['lr'] = lr
            if self.rank == 0:
                self.wandb_logger.log(loss_dict, step=epoch)

    def get_lr_scheduler(self):
        warmup_steps = int(self.warmup_epochs * len(self.train_dataloader))
        t_total = self.max_epochs * len(self.train_dataloader)
        
        if self.config.get('scheduler') == "warmup_cosine":
            scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=warmup_steps, t_total=t_total)
        elif self.config.get('scheduler') == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)
        elif self.config.get('scheduler') == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        else:
            scheduler = None
        return scheduler

    def get_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        base_lr = float(self.config.get('base_lr'))
        weight_decay = self.config.get('weight_decay', 1e-5)
        momentum = self.config.get('momentum', 0.9)

        if optimizer_name == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=base_lr, amsgrad=True)
        elif optimizer_name == 'adam':
            optimizer = Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
        return optimizer
    
    def save_checkpoint(self, epoch, filename='latest.pt'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss if hasattr(self, 'best_val_loss') else None,
        }
        save_path = os.path.join(self.config.get('ckpt_dir'), f'{filename}')
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt)

        if "state_dict" in checkpoint.keys():
            model_state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint.keys():
            model_state_dict = checkpoint["model_state_dict"]
        elif "network_weights" in checkpoint.keys():
            model_state_dict = checkpoint["network_weights"]
        elif "net" in checkpoint.keys():
            model_state_dict = checkpoint["net"]
        else:
            model_state_dict = checkpoint

        if "module." in list(model_state_dict.keys())[0]:
            # print("Tag 'module.' found in state dict - fixing!")
            for key in list(model_state_dict.keys()):
                model_state_dict[key.replace("module.", "")] = model_state_dict.pop(key)

        if "backbone." in list(model_state_dict.keys())[0]:
            # print("Tag 'backbone.' found in state dict - fixing!")
            for key in list(model_state_dict.keys()):
                model_state_dict[key.replace("backbone.", "")] = model_state_dict.pop(key)

        if "swin_vit" in list(model_state_dict.keys())[0]:
            # print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(model_state_dict.keys()):
                model_state_dict[key.replace("swin_vit", "swinViT")] = model_state_dict.pop(key)

        current_model_dict = self.model.model_state_dict()
        new_model_state_dict = {
            k: model_state_dict[k] if (k in model_state_dict.keys()) and (model_state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
            for k in current_model_dict.keys()}

        self.model.load_state_dict(new_model_state_dict, strict=True)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logging.info(f"Loaded checkpoint: {ckpt}")

    def train(self):
        self.model.train()
        train_loss = AverageMeter()
        ######## Start of Training ########
        self.step = 0
        for epoch in tqdm(range(self.start_epoch, self.max_epochs), desc="Epochs"):
                train_loss.reset()
                ###### Start of epoch ########
                if self.distributed:
                    # self.train_dataloader.sampler.set_epoch(epoch)
                    torch.distributed.barrier()
                for idx, batch in enumerate(self.train_dataloader):
                    self.optimizer.zero_grad()
                    t1 = time()
                    img = batch['cbct_image'].cuda(self.device, non_blocking=True) # B C H W D (B, 1, 256, 256, 32)
                    # label = batch['label'].cuda(self.device) # B C H W D (B, 1, 256, 256, 32)
                    pcd_lower = batch['ios_lower_pcd'].cuda(self.device, non_blocking=True) # B, C, N (B, 6, 24000)
                    pcd_upper = batch['ios_upper_pcd'].cuda(self.device, non_blocking=True) # B, C, N (B, 6, 24000)
                    b, c, h, w, d = img.shape
                    
                    # Forward pass
                    # _, _, loss = self.model(img, label, pcd_lower, pcd_upper)
                    with autocast(enabled=self.amp):
                        loss, features = self.model(img, img, pcd_lower, pcd_upper)
                        image_loss, pcd_loss, image_pcd_loss, total_loss = loss
                    if self.amp:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        self.optimizer.step()

                    if self.distributed:
                        # Convert loss list to tensors
                        loss_tensors = [torch.tensor(v.item(), device=self.device) for v in loss]
                        loss_list = distributed_all_gather(loss_tensors, out_numpy=True, is_valid=idx < len(self.train_dataloader))
                        
                        # Convert gathered losses to dictionary
                        gathered_losses = {
                            "total": np.mean([l[0] for l in loss_list]),
                            "image": np.mean([l[1] for l in loss_list]),
                            "pcd": np.mean([l[2] for l in loss_list]),
                            "image_pcd": np.mean([l[3] for l in loss_list])
                        }
                        
                        train_loss.update(gathered_losses, n=b * self.world_size)
                    else:
                        train_loss.update({
                            "total": total_loss.item(),
                            "image": image_loss.item(),
                            "pcd": pcd_loss.item(),
                            "image_pcd": image_pcd_loss.item()
                        }, n=b)
                    if self.rank == 0:
                        self.wandb_logger.log(train_loss.meters['value'], step=self.step)
                    if (idx % self.log_freq == 0) and (not self.distributed or self.rank == 0):
                        self.log_batch_info(epoch, idx, train_loss.meters['value'])
                        
                    if self.scheduler is not None and self.step % self.config.get('scheduler_step', 100) == 0:
                        self.scheduler.step()
                        
                    self.step += 1
                    del img, pcd_lower, pcd_upper, loss, features
                    torch.cuda.empty_cache()
                    ######## End of batch ########
                ######## End of epoch ########

                if (not self.distributed or self.rank == 0):
                    self.log_epoch_info(epoch, train_loss.meters['avg'], wandb_on=False)
                    
                if (epoch % self.save_freq == 0) and (not self.distributed or self.rank == 0):
                    self.save_checkpoint(epoch, "latest_model.pt")
        
        ######### End of Training ########
        if self.distributed:
            if dist.get_rank() == 0:
                self.save_checkpoint(epoch, "latest_model.pt")
            dist.destroy_process_group()
        else:
            self.save_checkpoint(epoch, "latest_model.pt")
            

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--config", default="configs/pretrain_single_config.yaml", help="Path to config file")
    parser.add_argument("--data_scale", default=1, type=float, help="Path to config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Config(args.config).config
    config['data_scale'] = args.data_scale
    seed(config.get('seed', 1234))
    
    if config.get('distributed', False):
        # Set multiprocessing start method
        torch.multiprocessing.set_start_method("fork", force=True)
        
        # Get number of GPUs
        config['ngpus_per_node'] = torch.cuda.device_count()
        print(f"Found {config['ngpus_per_node']} GPUs")
        
        # Spawn processes
        mp.spawn(main_worker, nprocs=config['ngpus_per_node'], args=(config,))
    else:
        main_worker(device=0, config=config)
    return
        
def main_worker(device, config):
    if config.get('distributed'):
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(config['ngpus_per_node'])
        os.environ['RANK'] = str(device)
        os.environ['LOCAL_RANK'] = str(device)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=config['ngpus_per_node'],
            rank=device
        )
        config['rank'] = device
        # Set device
        torch.cuda.set_device(device)
    else:
        config['rank'] = 0

    # Model
    model_manager = ModelManager(config, device)
    model = model_manager.get_model()
    
    if config.get('distributed'):
        # Convert BatchNorm to SyncBatchNorm only in distributed mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(device)
        # Use find_unused_parameters=True to handle unused parameters
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True
        )
    else:
        model.cuda(device)

    # Data loaders
    dataset_manager = DatasetManager(config)
    train_dataloader = dataset_manager.get_train_dataloader()

    # Train
    train_manager = Trainer(model, train_dataloader, config, device)
    train_manager.train()

    if config.get('distributed'):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()