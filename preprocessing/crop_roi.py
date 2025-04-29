import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import os
import argparse
import multiprocessing

def process_image(args):
    image_file, labels_dir = args
    try:
        casename = os.path.basename(image_file).split('.')[0]
        label_file = os.path.join(labels_dir, f'{casename}.nii.gz')
        
        image = sitk.ReadImage(image_file)
        label = sitk.ReadImage(label_file)
        
        image_data = sitk.GetArrayFromImage(image)
        label_data = sitk.GetArrayFromImage(label)
        
        label_array = label_data
        image_array = image_data

        # Create a mask for labels >= 1
        roi_mask = label_array >= 1

        if not np.any(roi_mask):
            # If there are no labels with values >=1, return zeros
            roi_label = np.zeros_like(label_array)
            roi_image = np.zeros_like(image_array)
        else:
            # Get the indices where the ROI mask is True
            indices = np.nonzero(roi_mask)
            z_min, y_min, x_min = np.min(indices, axis=1)
            z_max, y_max, x_max = np.max(indices, axis=1)

            # Crop the label and image to the ROI
            # Add padding
            padding_z = 5
            padding_xy = 10
            z_min_padded = max(z_min - padding_z, 0)
            z_max_padded = min(z_max + padding_z + 1, label_array.shape[0])
            
            y_min_padded = max(y_min - padding_xy, 0)
            y_max_padded = min(y_max + padding_xy + 1, label_array.shape[1])
            
            x_min_padded = max(x_min - padding_xy, 0)
            x_max_padded = min(x_max + padding_xy + 1, label_array.shape[2])

            roi_label = label_array[z_min_padded:z_max_padded, y_min_padded:y_max_padded, x_min_padded:x_max_padded]
            roi_image = image_array[z_min_padded:z_max_padded, y_min_padded:y_max_padded, x_min_padded:x_max_padded]

        # Proceed with saving the cropped images and labels
        cropped_label = roi_label        
        cropped_label_nifti = sitk.GetImageFromArray(cropped_label)
        cropped_label_nifti.SetSpacing(label.GetSpacing())
        cropped_label_nifti.SetOrigin(label.GetOrigin())
        cropped_label_nifti.SetDirection(label.GetDirection())

    except Exception as e:
        print(f"Error processing {image_file}: {e}")

def crop_images_roi(images_dir, labels_dir, num_processes=None):
    image_files = sorted(glob(os.path.join(images_dir, '*.nii.gz')))
    args_list = [(image_file, labels_dir) for image_file in image_files]
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, args_list), total=len(args_list)):
            pass

def parse_args():
    parser = argparse.ArgumentParser(description='Crop ROI')
    parser.add_argument('--images_dir', type=str, default='/data/kyle/Dental/DATA/DBT/DBT_Batch1/CBCT/imagesTs/')
    parser.add_argument('--labels_dir', type=str, default='/data/kyle/Dental/DATA/DBT/DBT_Batch1/CBCT/nnunet_predBinaryLabelsTs/')
    parser.add_argument('--save_dir', type=str, default='/data/kyle/Dental/DATA/DBT/DBT_Batch1/CBCT/nnunet_predBinaryLabelsTs/')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use. Default is number of CPU cores.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    crop_images_roi(args.images_dir, args.labels_dir, num_processes=args.num_processes)

if __name__ == '__main__':
    main()