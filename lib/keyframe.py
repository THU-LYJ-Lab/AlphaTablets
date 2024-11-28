import numpy as np
import glob
import os
from datetime import datetime


def get_keyframes(folder, min_angle=15, min_distance=0.2, window_size=9,
                  min_mean=0.2, max_mean=10):
    txt_list = sorted(glob.glob(f'{folder}/pose/*.txt'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    if len(txt_list) != 0:
        last_pose = np.loadtxt(txt_list[0])
        image_skip = np.ceil(len(txt_list) / 2000)
        txt_list = txt_list[::int(image_skip)]
        pose_num = len(txt_list)
    else:
        extrs = np.loadtxt(os.path.join(folder, 'traj.txt')).reshape(-1, 4, 4)
        last_pose = extrs[0]
        image_skip = np.ceil(len(extrs) / 2000)
        extrs = extrs[::int(image_skip)]
        pose_num = len(extrs)
 
    count = 1
    all_ids = []

    if len(txt_list) != 0:
        depth_list = [ pname.replace('pose', 'aligned_dense_depths').replace('.txt', '.npy') for pname in txt_list ]
    else:
        depth_list = sorted(glob.glob(f'{folder}/aligned_dense_depths/*.npy'))
    for i, j in zip(txt_list, depth_list):
        if int(i.split('/')[-1].split('.')[0]) != int(j.split('/')[-1].split('.')[0]):
            print(i, j)
            raise ValueError('pose and depth not match')
    depth_list = depth_list[::int(image_skip)]

    depth_list = np.array([ np.load(i).mean() for i in depth_list ])
    id_list = np.linspace(0, len(depth_list)-1, pose_num).astype(int)[::int(image_skip)]
    id_list = id_list[np.logical_and(depth_list > min_mean, depth_list < max_mean)]
    depth_list = depth_list[np.logical_and(depth_list > min_mean, depth_list < max_mean)]

    from scipy.signal import medfilt
    filtered_depth_list = medfilt(depth_list, kernel_size=29)

    # filtered out the depth_list
    id_list = id_list[
        np.logical_and(
            depth_list > filtered_depth_list * 0.85,
            depth_list < filtered_depth_list * 1.15
        )
    ]

    depth_list = depth_list[
        np.logical_and(
            depth_list > filtered_depth_list * 0.85,
            depth_list < filtered_depth_list * 1.15
        )
    ]

    print(str(datetime.now()) + ': \033[92mI', 'filtered out depth_list', len(id_list), '/', pose_num, '\033[0m')

    ids = [id_list[0]]

    for idx_pos, idx in enumerate(id_list[1:]):
        if len(txt_list) != 0:
            i = txt_list[idx]
            with open(i, 'r') as f:
                cam_pose = np.loadtxt(i)
        else:
            cam_pose = extrs[idx]
        angle = np.arccos(
            ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                [0, 0, 1])).sum())
        dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
        if angle > (min_angle / 180) * np.pi or dis > min_distance:
            ids.append(idx)
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            count += 1
            if count == window_size:
                ids = [i * int(image_skip) for i in ids]
                all_ids.append(ids)
                ids = []
                count = 0

    if len(ids) > 2:
        ids = [i * int(image_skip) for i in ids]
        all_ids.append(ids)
    else:
        ids = [i * int(image_skip) for i in ids]
        all_ids[-1].extend(ids)

    return all_ids, int(image_skip)


if __name__ == '__main__':
    folder = '/data0/bys/Hex/PixelPlane/data/scene0709_01'
    keyframes, image_skip = get_keyframes(folder)
    print(keyframes, image_skip)
