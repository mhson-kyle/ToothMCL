# ToothMCL: Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation

## Getting Started

To get started with this project, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/mhson-kyle/ToothMCL.git
cd ToothMCL
```

### Requirements
Before Training the model, make sure you have the following requirements installed:

```bash
pip install -r requirements.txt
```

### Pretraining
1. Prepare your dataset in the required format
2. Adjust the configuration files to suit your training needs
3. Run the following command to train the model:

```bash
cd pretrain
CUDA_VISIBLE_DEVICES=0 python train.py --config path/to/config.yaml
```