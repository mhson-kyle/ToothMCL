import os
import argparse
from glob import glob
from tqdm import tqdm
import json

import torch
import numpy as np
import open3d as o3d
from external_libs.pointops.functions import pointops


def load_obj(obj_file):
    import trimesh
    tri_mesh_loaded_mesh = trimesh.load_mesh(obj_file, process=False)
    vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
    tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    # mesh = o3d.io.read_triangle_mesh(obj_file)
    # mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals) 
    features = np.concatenate([vertices, vertex_normals], axis=1)
    # print(features.shape)
    return features, mesh

def load_json(json_file):
    with open(json_file, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def torch_to_numpy(cuda_arr):
    if cuda_arr.is_cuda:
        return cuda_arr.cpu().detach().numpy()
    else:
        return cuda_arr.detach().numpy()

def resample_pcd(pcd_ls, n, method):
    """Drop or duplicate points so that pcd has exactly n points"""
    if method=="uniformly":
        idx = np.random.permutation(pcd_ls[0].shape[0])
    elif method == "fps":
        idx = fps(pcd_ls[0][:, :3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    return pcd_resampled_ls

def fps(xyz, npoint):
    if xyz.shape[0] <= npoint:
        raise "new fps error"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(xyz, 
                                    torch.tensor([xyz.shape[0]]).cuda().type(torch.int), 
                                    torch.tensor([npoint]).cuda().type(torch.int)) 
    return torch_to_numpy(idx).reshape(-1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_dir', default="/data/kyle/Dental/DATA/IOS/3DS/obj/", type=str, help="data path in which original .obj data are saved")
    parser.add_argument('--json_dir', default='/data/kyle/Dental/DATA/IOS/3DS/json/', type=str, help="data path in which original .json data are saved")
    parser.add_argument('--save_dir', default="/data/kyle/Dental/DATA/IOS/3DS/fps_sampled_points", type=str, help="data path in which processed data will be saved")
    args = parser.parse_args()
    return args

def main():
    Y_AXIS_MAX = 33.15232091532151
    Y_AXIS_MIN = -36.9843781139949
    
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    obj_files = sorted(glob(os.path.join(args.obj_dir, "*.obj")))
    
    if args.json_dir is not None:
        for obj_file in tqdm(obj_files):
            casename = os.path.basename(obj_file).split(".")[0] # XXXXX_lower, TADPM_F0001_lower
            # print(casename)
            json_file = os.path.join(args.json_dir, f'{casename}.json')
            assert casename == os.path.basename(json_file).split(".")[0]
            
            torch.cuda.empty_cache()
            features, mesh = load_obj(obj_file)
            loaded_json = load_json(json_file)
            labels = np.array(loaded_json['labels']).reshape(-1, 1)
            if loaded_json['jaw'] == 'lower':
                labels -= 20

            # Convert label to 0-16 range
            labels[labels // 10 == 1] %= 10
            labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
            labels[labels < 0] = 0
            try:
                features = np.concatenate([features, labels], axis=1)
            except:
                print(f"Error: {casename}")
                continue
            features[:, :3] = ((features[:, :3] - Y_AXIS_MIN) / (Y_AXIS_MAX - Y_AXIS_MIN)) * 2 - 1 # Vertices normalization
            num_vertices = features.shape[0]
            if num_vertices > 24000:
                sampled_points = resample_pcd([features], 24000, "fps")[0]
            else:
                sampled_points = features
                
            np.save(os.path.join(args.save_dir, f"{casename}_sampled_points"), sampled_points)
    else:
        for obj_file in tqdm(obj_files):
            torch.cuda.empty_cache() 
            casename = os.path.basename(obj_file).split(".")[0]
            
            features, mesh = load_obj(obj_file)
            features[:, :3] = ((features[:, :3] - Y_AXIS_MIN) / (Y_AXIS_MAX - Y_AXIS_MIN)) * 2 - 1 # Vertices normalization
            num_vertices = features.shape[0]
            if num_vertices > 24000:
                sampled_points = resample_pcd([features], 24000, "fps")[0]
            np.save(os.path.join(args.save_dir, f"{casename}_sampled_points"), sampled_points)
        
        
if __name__ == '__main__':
    main()