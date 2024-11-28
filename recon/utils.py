import json
import os
import cv2
import glob
import numpy as np


def load_images(img_dir):
    imgs = []
    image_names = sorted(os.listdir(img_dir))
    for name in image_names:
        img = cv2.imread(os.path.join(img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs, image_names


def load_sfm(sfm_dir):
    """Load converted sfm world-to-cam depths and poses
    """
    depth_dir = os.path.join(sfm_dir, "colmap_outputs_converted/depths")
    pose_dir = os.path.join(sfm_dir, "colmap_outputs_converted/poses")
    depth_names = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    pose_names = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    sfm_depths, sfm_poses = [], []
    for dn, pn in zip(depth_names, pose_names):
        assert os.path.basename(dn)[:-4] == os.path.basename(pn)[:-4]
        depth = np.load(dn)
        pose = np.loadtxt(pn)
        sfm_depths.append(depth)
        sfm_poses.append(pose)
    return sfm_depths, sfm_poses


def load_sfm_pose(sfm_dir):
    """Load sfm poses and then convert into proj mat
    """
    pose_dir = os.path.join(sfm_dir, "colmap_outputs_converted/poses")
    intr_dir = os.path.join(sfm_dir, "colmap_outputs_converted/intrinsics")
    pose_names = sorted(glob.glob(os.path.join(pose_dir, "*.txt")), key=lambda x: int(x.split('/')[-1][:-4]))
    intr_names = sorted(glob.glob(os.path.join(intr_dir, "*.txt")), key=lambda x: int(x.split('/')[-1][:-4]))
    K = np.loadtxt(intr_names[0])
    KH = np.eye(4)
    KH[:3,:3] = K
    sfm_poses, sfm_projmats = [], []
    for pn in pose_names:
        # world-to-cam
        pose = np.loadtxt(pn)
        pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
        c2w = np.linalg.inv(pose)
        c2w[0:3, 1:3] *= -1
        pose = np.linalg.inv(c2w)
        sfm_poses.append(pose)
        # projmat
        projmat = KH @ pose
        sfm_projmats.append(projmat)
    return K, sfm_poses, sfm_projmats


def load_scannet_pose(scannet_dir):
    """Load scannet poses and then convert into proj mat
    """
    pose_dir = os.path.join(scannet_dir, "pose")
    intr_dir = os.path.join(scannet_dir, "intrinsic")
    pose_names = sorted(glob.glob(os.path.join(pose_dir, "*.txt")), key=lambda x: int(x.split('/')[-1][:-4]))
    intr_name = os.path.join(intr_dir, "intrinsic_color.txt")
    KH = np.loadtxt(intr_name)
    scannet_poses, scannet_projmats = [], []
    for pn in pose_names:
        p = np.loadtxt(pn)
        R = p[:3, :3]
        R = np.matmul(R, np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]))
        p[:3, :3] = R
        p = np.linalg.inv(p)
        scannet_poses.append(p)
        # projmat
        projmat = KH @ p
        scannet_projmats.append(projmat)
    return KH[:3, :3], scannet_poses, scannet_projmats


def load_replica_pose(replica_dir):
    """Load replica poses and then convert into proj mat
    """
    if os.path.exists(os.path.join(replica_dir, 'cam_params.json')):
        cam_param_path = os.path.join(replica_dir, 'cam_params.json')
    elif os.path.exists(os.path.join(replica_dir, '../cam_params.json')):
        cam_param_path = os.path.join(replica_dir, '../cam_params.json')
    else:
        raise FileNotFoundError('cam_params.json not found')
    with open(cam_param_path, 'r') as f:
        j = json.load(f)
        intrinsics = np.array([
            j["camera"]["fx"], 0, j["camera"]["cx"],
            0, j["camera"]["fy"], j["camera"]["cy"],
            0, 0, 1
        ], dtype=np.float32).reshape(3, 3)

    KH = np.eye(4)
    KH[:3, :3] = intrinsics

    extrinsics = np.loadtxt(os.path.join(replica_dir, 'traj.txt')).reshape(-1, 4, 4)
    poses = []
    projmats = []
    for extrinsic in extrinsics:
        p = extrinsic
        R = p[:3, :3]
        R = np.matmul(R, np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]))
        p[:3, :3] = R
        p = np.linalg.inv(p)
        poses.append(p)
        # projmat
        projmat = KH @ p
        projmats.append(projmat)

    return intrinsics, poses, projmats


def load_dense_depths(depth_dir):
    """Load initialized single-image dense depth predictions
    """
    depth_names = sorted(os.listdir(depth_dir))
    depths = []
    for dn in depth_names:
        depth = np.load(os.path.join(depth_dir, dn))
        depths.append(depth)
    return depths, depth_names


def depth2disp(depth):
    """Convert depth map to disparity
    """
    disp = 1.0 / (depth + 1e-8)
    return disp


def disp2depth(disp):
    """Convert disparity map to depth
    """
    depth = 1.0 / (disp + 1e-8)
    return depth
