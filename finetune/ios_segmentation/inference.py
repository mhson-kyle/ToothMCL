import os
from glob import glob
import argparse

from tqdm import tqdm

from inference_pipelines import make_inference_pipeline
from utils.predict_utils import ScanSegmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Inference models')
    parser.add_argument('--input_dir', default="/ssddata/data/mhson/DATA/IOS/DBT_pre/", type=str, help = "input directory path that contain obj files.")
    parser.add_argument('--save_dir', type=str, default="/ssddata/data/mhson/DATA/IOS/DBT_pre/", help = "result save directory.")
    parser.add_argument('--model_name', type=str, default="pointtransformer", help = "model name. list: tsegnet | tgnet | pointnet | pointnetpp | dgcnn | pointtransformer")
    parser.add_argument('--checkpoint_path', default="ckpts/tgnet_fps" ,type=str,help = "checkpoint path.")
    parser.add_argument('--jaw', default="upper", type=str, help = "jaw type. list: upper | lower")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    obj_files = sorted(glob(os.path.join(args.input_dir, f"*{args.jaw}.obj")))

    pred_obj = ScanSegmentation(make_inference_pipeline(args.model_name, args.checkpoint_path))
    os.makedirs(args.save_dir, exist_ok=True)
    for obj_file in tqdm(obj_files):
        base_name = os.path.dirname(obj_file).split("/")[-1]
        print(base_name)
        # break
        pred_obj.process(obj_file, os.path.join(args.save_path, base_name, f"{args.jaw}.json"))

if __name__ == "__main__":
    main()