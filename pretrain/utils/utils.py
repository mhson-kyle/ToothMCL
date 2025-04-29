import os
import json

import numpy as np
import torch
import SimpleITK as sitk


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        # Handle case where tensor_list is a list of tensors
        if isinstance(tensor_list, list) and len(tensor_list) > 0:
            if isinstance(tensor_list[0], list):
                # If tensor_list[0] is also a list, get the first tensor from it
                device = tensor_list[0][0].device
            else:
                device = tensor_list[0].device
        else:
            # Default to current device if we can't determine from tensor_list
            device = torch.cuda.current_device()
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=device)
    
    if not no_barrier:
        torch.distributed.barrier()
    
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        
        # Handle nested lists of tensors
        if isinstance(tensor_list, list) and len(tensor_list) > 0 and isinstance(tensor_list[0], list):
            # Flatten the nested list
            flat_tensors = [t for sublist in tensor_list for t in sublist]
            for tensor in flat_tensors:
                gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gather_list, tensor)
                if valid_batch_size is not None:
                    gather_list = gather_list[:valid_batch_size]
                elif is_valid is not None:
                    gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
                if out_numpy:
                    gather_list = [t.cpu().numpy() for t in gather_list]
                tensor_list_out.append(gather_list)
        else:
            # Handle regular list of tensors
            for tensor in tensor_list:
                gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gather_list, tensor)
                if valid_batch_size is not None:
                    gather_list = gather_list[:valid_batch_size]
                elif is_valid is not None:
                    gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
                if out_numpy:
                    gather_list = [t.cpu().numpy() for t in gather_list]
                tensor_list_out.append(gather_list)
    
    return tensor_list_out

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)
    
def load_niigz(file_path):
    image_file = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(image_file)
    return image, image_file

class AverageMeter(object):
    """Computes and stores the average and current value for multiple loss types."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.meters = {'value': {}, 'sum': {}, 'count': {}, 'avg': {}}

    def update(self, loss_dict, n=1):
        for loss_type, value in loss_dict.items():
            if loss_type not in self.meters['value']:
                self.meters['value'][loss_type] = 0.0
                self.meters['sum'][loss_type] = 0.0
                self.meters['count'][loss_type] = 0
                self.meters['avg'][loss_type] = 0.0

            self.meters['value'][loss_type] = value
            self.meters['sum'][loss_type] += value * n
            self.meters['count'][loss_type] += n
            self.meters['avg'][loss_type] = (
                self.meters['sum'][loss_type] / self.meters['count'][loss_type]
            )


def distributed_all_gather(tensor_list, 
                           valid_batch_size=None, 
                           out_numpy=False, 
                           world_size=None, 
                           no_barrier=False, 
                           is_valid=None):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out