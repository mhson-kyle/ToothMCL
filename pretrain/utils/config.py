import os
import yaml
import numpy as np
import random
import logging
from datetime import datetime

import torch
from munch import unmunchify


class Config:
    def __init__(self, config_file: str, test: bool = False):
        self.config_file = config_file
        self.load_config()
        self.validate_args()
        if not test:
            self.setup_run()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def validate_args(self):
        # Validate root_dir
        data_dir = self.config.get('data_dir', '')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    def setup_run(self):
        self.create_dir()
        self.set_logger()
        self.save_config()

    def create_dir(self):
        current_datetime = datetime.now().strftime('%y%m%d_%H%M%S')
        self.run_name = self.config.get('run_name') + '_' + current_datetime
        print(f'Run Name: {self.run_name}')
        
        self.config['run_dir'] = os.path.join('runs', self.run_name)
        self.config['ckpt_dir'] = os.path.join('runs', self.run_name, 'ckpts')
        self.config['log_dir'] = os.path.join('runs', self.run_name, 'logs')

        os.makedirs(self.config.get('run_dir'), exist_ok=True)
        os.makedirs(self.config.get('ckpt_dir'), exist_ok=True)
        os.makedirs(self.config.get('log_dir'), exist_ok=True)

    def set_logger(self):
        logging_level = logging.getLevelName(self.config.get('loglevel').upper())

        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=os.path.join(self.config.get('log_dir'), 'training_log.txt'),
            filemode='w',
            encoding='utf-8'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(console_handler)

    def save_config(self):
        with open(os.path.join(self.config['run_dir'], 'config.yaml'), 'w') as file:
            yaml.safe_dump(unmunchify(self.config), file, default_flow_style=False, sort_keys=False, width=1000)
        
def seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
