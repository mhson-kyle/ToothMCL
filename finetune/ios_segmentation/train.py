import os
import logging
import argparse

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from external_libs.scheduler import build_scheduler_from_cfg
from utils.utils import LossMeter
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.config import Config
from utils.utils import seed
from models import get_model
from datasets import DatasetManager


class Trainer:
    def __init__(self, config=None, model=None, train_dataloader=None, val_dataloader=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.model = model
        
        self.model.cuda()
        # self.step_count = 0
        self.log_freq = self.config.get('log_freq', 100)
        self.start_epoch = 0
        self.max_epochs = self.config.get('max_epochs', 500)
        
        self.best_val_loss = float('inf')
        self.prev_val_loss = float('inf')
        self.patience = self.config.get('patience', 5)
        
        self.set_optimizer()
        self.set_scheduler()
        
        if self.config.get('resume', None) is not None:
            self.load_checkpoint()
        
        self.wandb_on = self.config.get('wandb_on', False)
        if self.wandb_on:
            wandb.init(
                entity=self.config["entity"],
                project=self.config["project"],
                name=self.config["run_name"],
                config=self.config,
            )

    def log_batch_info(self, epoch, iter_num, loss_dict):
        loss_info = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        lr = self.optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch [{epoch}/{self.max_epochs}], "
            f"Batch [{iter_num}/{len(self.train_dataloader)}], "
            f"LR: {lr:.6f}, "
            f"{loss_info}"
        )

    def log_epoch_info(self, epoch, loss_dict, wandb_on=False, image=None, pred=None, gt=None):
        loss_info = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        lr = self.optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch [{epoch + 1}/{self.max_epochs}], "
            f"LR: {lr:.6f}, "
            f"{loss_info}, "
        )

        if wandb_on:
            loss_dict['lr'] = lr
            wandb.log(loss_dict, step=epoch)
                
    def train(self):
        total_loss_meter = LossMeter()
        step_loss_meter =  LossMeter()
        self.step_count = 0
        for self.epoch in tqdm(range(self.start_epoch, self.max_epochs)):
            self.model.train()
            pre_step = self.step_count
            for batch_idx, batch_item in enumerate(self.train_dataloader):
                loss = self.model.step(batch_item)
                
                loss_sum = loss.get_sum()
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()
                self.scheduler.step()
                # if ((batch_idx + 1) % self.config["scheduler_step"] == 0) or (self.step_count == pre_step and batch_idx == len(self.train_dataloader) - 1):
                #     self.step_count += 1
                #     self.scheduler.step(self.step_count)
                
                torch.cuda.empty_cache()
                
                total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
                step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
                
                if batch_idx % self.log_freq == 0:
                    self.log_batch_info(
                        epoch=self.epoch,
                        iter_num=batch_idx,
                        loss_dict=loss.get_loss_dict_for_print("step")
                    )
                ## End of Batch
            
            # Logging
            self.log_epoch_info(
                epoch=self.epoch,
                loss_dict=total_loss_meter.get_avg_results(),
                wandb_on=self.wandb_on,
            )
            self.save_checkpoint("latest.pt")
            ## End of Epoch
            
            if self.test():
                # break
                pass
        ## End of Training

        self.save_checkpoint("latest.pt")

    def test(self):
        total_loss_meter = LossMeter()
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_item in enumerate(self.val_dataloader):
                loss = self.model.step(batch_item)
                total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        self.log_epoch_info(
            epoch=self.epoch,
            loss_dict=avg_total_loss,
            wandb_on=self.wandb_on,
        )
        
        # Save best model
        if self.best_val_loss > avg_total_loss["total_val"]:
            self.best_val_loss = avg_total_loss["total_val"]
            self.save_checkpoint("best.pt")
        
        # Early Stopping
        if self.prev_val_loss > avg_total_loss["total_val"]:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            logging.info(
                f'No improvement in validation for {self.early_stop_counter} epochs.'
                )
            if self.early_stop_counter >= self.patience:
                logging.info(
                    'Early stopping triggered. Stopping training.'
                    )
                return True
        self.prev_val_loss = avg_total_loss["total_val"]
        return False
        
    def set_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        self.base_lr = self.config.get('base_lr', 1.0e-3)
        self.weight_decay = self.config.get('weight_decay', 1.0e-4)
        self.momentum = self.config.get('momentum', 0.9)
        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(), 
                lr=self.base_lr, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), 
                lr=self.base_lr, 
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=self.base_lr, 
                weight_decay=self.weight_decay
            )

    def set_scheduler(self):
        warmup_steps = self.config.get('warmup_epochs', 10) * len(self.train_dataloader)
        max_steps = self.max_epochs * len(self.train_dataloader)
        if self.config.get('scheduler') == "exp":
            self.scheduler = ExponentialLR(
                optimizer=self.optimizer, 
                gamma=self.config["tr_set"]["scheduler"]["step_decay"]
            )
        elif self.config.get('scheduler') == "warmup_cosine":
            self.scheduler = LinearWarmupCosineAnnealingLR(
                self.optimizer, warmup_epochs=warmup_steps, max_epochs=max_steps
            )
        elif self.config.get('scheduler') == "cosine":
            scheduler_config = {}
            scheduler_config["sched"] = "cosine"
            scheduler_config["full_steps"] = self.config.get('full_steps', 40)
            scheduler_config["lr"] = self.config.get('base_lr', 1.0e-1)
            scheduler_config["min_lr"] = self.config.get('min_lr', 1.0e-5)
            scheduler_config["scheduler_step"] = self.config.get('scheduler_step', 1)
            self.scheduler = build_scheduler_from_cfg(scheduler_config, self.optimizer)

    def load_checkpoint(self):
        ckpt = torch.load(self.config.get('resume'))
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.model.load_state_dict(ckpt['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    def save_checkpoint(self, savename):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss if hasattr(self, 'best_val_loss') else None
        }
        
        savename = os.path.join(self.config["ckpt_dir"], savename)
        torch.save(checkpoint, savename)
        logging.info(f"Save checkpoint: {savename}")
        
def parse_args():
    parser = argparse.ArgumentParser(description='Inference models')
    parser.add_argument('--config', default="configs/pointtransformer_lower.yaml", type=str, help = "train config file path.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Config(args.config).config
    seed(config.get('seed', 42))

    # Model
    model = get_model(config)

    # Dataset Manager
    dataset_manager = DatasetManager(config)
    train_dataloader, val_dataloader = dataset_manager.get_dataloader()

    # Trainer
    trainner = Trainer(
        config=config, 
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    trainner.train()

if __name__ == '__main__':
    main()
    