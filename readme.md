# ToothMCL: Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation

This is the official repository of "[Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation]()."
> **Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation** <br>
> [Moo Hyun (Kyle) Son](mailto:mhson@cse.ust.hk)<sup>1</sup>, [Juyoung (Justin) Bae](mailto:jbaeaa@cse.ust.hk)<sup>1</sup>, [Zelin Qiu](mailto:zqiuak@cse.ust.hk)<sup>1</sup>, [Jiale Peng]()<sup>3</sup>[Kai Xin Li]()<sup>2</sup>, [Yifan Lin]()<sup>3</sup>, [Hao Chen](https://www.cse.ust.hk/~haochen/)<sup>1</sup> <br>
> <sup>1</sup>The Hong Kong University of Science and Technology (HKUST)&nbsp;&nbsp;<sup>2</sup>Delun Dental Hospital&nbsp;&nbsp;<sup>3</sup>The University of Hong Kong (HKU)<br>
>
> **Abstract.** Digital dentistry represents a transformative shift in modern dental practice. The foundational step in this transformation is the accurate digital representation of the patient's dentition, which is obtained from segmented Cone-Beam Computed Tomography (CBCT) and Intraoral Scans (IOS). Despite the growing interest in digital dental technologies, existing segmentation methodologies frequently lack rigorous validation and demonstrate limited performance and clinical applicability. To the best of our knowledge, this is the first work to introduce a multimodal pretraining framework for tooth segmentation. We present ToothMCL, a Tooth Multimodal Contrastive Learning for pretraining that integrates volumetric (CBCT) and surface-based (IOS) modalities. By capturing modality-invariant representations through multimodal contrastive learning, our approach effectively models fine-grained anatomical features, enabling precise multi-class segmentation and accurate identification of Fédération Dentaire Internationale (FDI) tooth numbering. Along with the framework, we curated \ourdataset{}, the largest paired CBCT and IOS dataset to date, comprising 3,867 patients. We then evaluated ToothMCL on a comprehensive collection of independent datasets, representing the largest and most diverse evaluation to date. Our method achieves state-of-the-art performance in both internal and external testing, with an increase of 12\% for CBCT segmentation and 8\% for IOS segmentation in the Dice Similarity Coefficient (DSC). Furthermore, ToothMCL consistently surpasses existing approaches in tooth groups and demonstrates robust generalizability across varying imaging conditions and clinical scenarios. Our findings underscore the transformative potential of large-scale multimodal pretraining in digital dentistry and highlight the critical importance of effectively leveraging paired multimodal data. Our approach lays the foundation for enhanced clinical workflows, including caries detection, orthodontic simulation, and dental prosthesis design.

 [Paper]() ·  [Code](https://github.com/mhson-kyle/ToothMCL)

<!-- <p align="center"><img src = assets/overview.png width="85%" height="85%"></p> -->

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
### Datasets and Preprocessing
- **CBCT**: 
Resample to 0.3 mm isotropic voxel size
```bash
cd preprocessing
python voxel_resample.py --input_dir path/to/input_dir --output_dir path/to/output_dir
```
- Crop to ROI.
```bash
cd preprocessing
python crop_roi.py --images_dir path/to/images_dir --labels_dir path/to/labels_dir --save_dir path/to/save_dir
```
- **IOS**:
Farthest point sampling to 24000 points per jaw
```bash
cd preprocessing
CUDA_VISIBLE_DEVICES=0 python fps.py --obj_dir path/to/obj_dir --json_dir path/to/json_dir --save_dir path/to/save_dir
```

### Pretraining
1. Prepare your dataset in the required format
2. Adjust the configuration files to suit your training needs
3. Run the following command to train the model:

```bash
cd pretrain
CUDA_VISIBLE_DEVICES=0 python train.py --config path/to/config.yaml
```

### Finetuning
#### CBCT
```bash
cd finetune/cbct_segmentation
sh scripts/train.sh
```
#### IOS
```bash
cd finetune/ios_segmentation
CUDA_VISIBLE_DEVICES=0 python train.py --config path/to/config.yaml
```


<!-- ## Results

Below shows the predicted perfusion parameter using the Progressive Knowledge Distillation. -->


## Acknowledgements
We sincerely thank those who have open-sourced their works including, but not limited to, the repositories below:
- [monai](https://github.com/Project-MONAI/research-contributions)
- [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer)
- [SwinUNETR](https://github.com/microsoft/SimMIM)
- [ToothGroupNetwork](https://github.com/limhoyeon/ToothGroupNetwork)

## Citation
If you find this repository useful, please consider citing:

``` bibtex
@inproceedings{Son2025ToothMCL,
  title={Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation},
  author={Son, Moo Hyun and Bae, Juyoung and Qiu, Zelin and Li, Kai Xin and Lin, Yifan and Chen, Hao},
  journal={Under Review},
  year={2025},
}
```