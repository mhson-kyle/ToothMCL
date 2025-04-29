import os
from glob import glob
from tqdm import tqdm

import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
import argparse
import SimpleITK as sitk

import SimpleITK as sitk


def resample_image(
        image_path: str,
        target_spacing=(0.30, 0.30, 0.30),
        interpolator=sitk.sitkLinear):
    """
    Resample a NIfTI volume to the desired voxel spacing and write it back
    to disk with SimpleITK, preserving orientation.

    Args
    ----
    image_path : str
        Path to the input .nii or .nii.gz.
    output_path : str
        Where the resampled image will be written.
    target_spacing : 3-tuple[float]
        Desired voxel spacing in millimetres (z, y, x order).
    interpolator : sitk.InterpolatorEnum
        sitk.sitkLinear for intensity images,
        sitk.sitkNearestNeighbor for label maps.
    """
    img = sitk.ReadImage(image_path)                  # load
    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()
    print(f"Original image shape: {orig_size}")
    print(f"Original image spacing: {orig_spacing}")
    # new size that fits the requested spacing
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())        # preserve origin
    resampler.SetOutputDirection(img.GetDirection())  # preserve orientation
    resampler.SetTransform(sitk.Transform())          # identity
    resampler.SetInterpolator(interpolator)

    resampled_img = resampler.Execute(img)            # resample
    return resampled_img  # handy if you want to keep it in RAM

def parse_args():
    parser = argparse.ArgumentParser(description="Resample NIfTI images to a target voxel spacing.")
    parser.add_argument("--images_dir", default="images", help="Path(s) to the input NIfTI image file(s) ('.nii.gz').")
    parser.add_argument("--save_dir", default="resampled_images", help="Directory to save the resampled images.")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="Target voxel spacing (x, y, z).")
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    save_dir = args.save_dir
    target_spacing = tuple(args.target_spacing)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_files = glob(os.path.join(args.images_dir, "*.nii.gz"))
    
    for image_path in image_files:
        try:
            new_img = resample_image(image_path, target_spacing)
            print(f"Resampled image shape: {new_img.GetSize()}")
            print(f"Resampled image spacing: {new_img.GetSpacing()}")
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            if ext == ".gz":
                name, ext2 = os.path.splitext(name)
                ext = ext2 + ext
            output_path = os.path.join(save_dir, f"{name}_resampled{ext}")
            sitk.WriteImage(new_img, output_path)
            print(f"Resampled image saved to: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()