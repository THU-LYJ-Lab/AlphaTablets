import os
import json
import numpy as np

from recon.third_party.omnidata.omnidata_tools.torch.demo_normal_custom_func import demo_normal_custom_func
from recon.third_party.metric3d.depth_custom_func import depth_custom_func


def metric3d_depth(img_dir, input_dir, depth_dir, dataset_type):
    """Initialize the dense depth of each image using single-image depth prediction
    """
    os.makedirs(depth_dir, exist_ok=True)
    if dataset_type == 'scannet':
        intr = np.loadtxt(f'{input_dir}/intrinsic/intrinsic_color.txt')
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    elif dataset_type == 'replica':
        if os.path.exists(os.path.join(input_dir, 'cam_params.json')):
            cam_param_path = os.path.join(input_dir, 'cam_params.json')
        elif os.path.exists(os.path.join(input_dir, '../cam_params.json')):
            cam_param_path = os.path.join(input_dir, '../cam_params.json')
        else:
            raise FileNotFoundError('cam_params.json not found')
        with open(cam_param_path, 'r') as f:
            j = json.load(f)
            fx, fy, cx, cy = j["camera"]["fx"], j["camera"]["fy"], j["camera"]["cx"], j["camera"]["cy"]
    else:
        raise NotImplementedError(f'Unknown dataset type {dataset_type} exiting')
    indir = img_dir
    outdir = depth_dir
    depth_custom_func(fx, fy, cx, cy, indir, outdir)


def omnidata_normal(img_dir, input_dir, normal_dir):
    """Initialize the dense normal of each image using single-image normal prediction
    """
    os.makedirs(normal_dir, exist_ok=True)
    demo_normal_custom_func(img_dir, normal_dir)
