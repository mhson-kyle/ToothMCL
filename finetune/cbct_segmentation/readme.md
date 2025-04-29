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

### Finetuning
#### CBCT
```bash
cd finetune/cbct_segmentation
sh scripts/train.sh
```