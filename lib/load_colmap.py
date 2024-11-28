from datetime import datetime
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

from recon.utils import load_sfm_pose, load_scannet_pose

from geom.plane_utils import calc_plane

import torch
import torch.nn.functional as F


def dilate(img):
    kernel = torch.ones(3, 3).cuda().double()
    img = torch.tensor(img.astype('int32')).cuda().double()
    
    while True:
        mask = img > 0
        if torch.logical_not(mask).sum() == 0:
            break
        mask = mask.double()

        mask_dilated = F.conv2d(mask[None, None], kernel[None, None], padding=1).squeeze(0).squeeze(0)
        
        img_dilated = F.conv2d(img[None, None], kernel[None, None], padding=1).squeeze(0).squeeze(0)
        img_dilated = torch.where(mask_dilated == 0, img_dilated, img_dilated / mask_dilated)

        img = torch.where(mask == 0, img_dilated, img)

    img = img.cpu().numpy()

    return img


def parse_sp(sp, intr, pose, depth, normal, img_sp, edge_thres=0.95, crop=20):
    """Parse the superpixel segmentation into planes
    """
    sp = torch.from_numpy(sp).cuda()
    depth = torch.from_numpy(depth).cuda()
    normal = (torch.from_numpy(normal).cuda() - 0.5) * 2
    normal = normal.permute(1, 2, 0)
    img_sp = torch.from_numpy(img_sp).cuda()
    sp = sp.int()
    sp_ids = torch.unique(sp)
    planes = []
    mean_colors = []

    # mask crop to -1
    sp[:crop] = -1
    sp[-crop:] = -1
    sp[:, :crop] = -1
    sp[:, -crop:] = -1

    new_sp = torch.ones_like(sp).cuda() * -1

    sp_id_cnt = 0

    for sp_id in tqdm(sp_ids):
        sp_mask = sp == sp_id

        if sp_mask.sum() <= 15:
            continue

        new_sp[sp_mask] = sp_id_cnt
        mean_color = torch.mean(img_sp[sp_mask], dim=0)

        sp_dis = depth[sp_mask]
        sp_dis = torch.median(sp_dis)
        if sp_dis < 0:
            continue
        sp_normal = normal[sp_mask]

        sp_coeff = torch.einsum('ij,kj->ik', sp_normal, sp_normal)
        if sp_coeff.min() < edge_thres:
            is_edge_plane = True
        else:
            is_edge_plane = False

        sp_normal = torch.mean(sp_normal, dim=0)
        sp_normal = sp_normal / torch.norm(sp_normal)
        sp_normal[1] = -sp_normal[1]
        sp_normal[2] = -sp_normal[2]

        x_accum = torch.sum(sp_mask, dim=0)
        y_accum = torch.sum(sp_mask, dim=1)
        x_range_uv = [torch.min(torch.nonzero(x_accum)).item(), torch.max(torch.nonzero(x_accum)).item()]
        y_range_uv = [torch.min(torch.nonzero(y_accum)).item(), torch.max(torch.nonzero(y_accum)).item()]
        
        sp_dis = sp_dis.cpu().numpy()
        sp_depth = depth[y_range_uv[0]:y_range_uv[1]+1, x_range_uv[0]:x_range_uv[1]+1].cpu().numpy()
        sp_mask = sp_mask[y_range_uv[0]:y_range_uv[1]+1, x_range_uv[0]:x_range_uv[1]+1].cpu().numpy()

        ret = calc_plane(intr, pose, sp_depth, sp_dis, sp_mask, x_range_uv, y_range_uv, plane_normal=sp_normal.cpu().numpy())
        if ret is None:
            continue
        plane, plane_up, resol, new_x_range_uv, new_y_range_uv = ret
        
        plane = np.concatenate([plane, plane_up, resol, new_x_range_uv, new_y_range_uv, [is_edge_plane]])
        planes.append(plane)
        mean_colors.append(mean_color.cpu().numpy())

        sp_id_cnt += 1

    return planes, new_sp.cpu().numpy(), mean_colors


def get_projection_matrix(fovy: float, aspect_wh: float, near: float, far: float):
    proj_mtx = np.zeros((4, 4), dtype=np.float32)
    proj_mtx[0, 0] = 1.0 / (np.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[1, 1] = -1.0 / np.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[2, 2] = -(far + near) / (far - near)
    proj_mtx[2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(c2w, proj_mtx):
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = np.zeros((c2w.shape[0], 4, 4))
    w2c[:, :3, :3] = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    w2c[:, :3, 3:] = np.transpose(-c2w[:, :3, :3], (0, 2, 1)) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0

    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c

    return mvp_mtx


def load_colmap_data(args, kf_list=[], image_skip=1, load_plane=True, scannet_pose=True):
    imgs = []
    image_dir = os.path.join(args.input_dir, "images")
    start = kf_list[0]
    end = kf_list[-1] + 1 if kf_list[-1] != -1 else len(glob.glob(os.path.join(image_dir, "*")))
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")), key=lambda x: int(x.split('/')[-1][:-4]))

    image_names_sp = [image_names[i] for i in kf_list]
    image_names = image_names[start:end:image_skip]

    print(str(datetime.now()) + ': \033[92mI', 'loading images ...', '\033[0m')
    for name in tqdm(image_names):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img.astype(np.float32) / 255.)

    imgs_sp = []
    for name in image_names_sp:
        img = cv2.imread(name)
        # convert to ycbcr using cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgs_sp.append(img.astype(np.float32) / 255.)

    imgs_sp = np.stack(imgs_sp, 0)

    if scannet_pose:
        print(str(datetime.now()) + ': \033[92mI', 'loading scannet poses ...', '\033[0m')
        intr_scannet, sfm_poses_scannet, sfm_camprojs_scannet = load_scannet_pose(args.input_dir)
        intr = intr_scannet
        sfm_poses, sfm_camprojs = sfm_poses_scannet, sfm_camprojs_scannet
    else:
        print(str(datetime.now()) + ': \033[92mI', 'loading colmap poses ...', '\033[0m')
        intr, sfm_poses, sfm_camprojs = load_sfm_pose(os.path.join(args.input_dir, "sfm"))


    sfm_poses_sp = np.array(sfm_poses)[kf_list]
    sfm_poses = np.array(sfm_poses)[start:end:image_skip]
    sfm_camprojs = np.array(sfm_camprojs)[start:end:image_skip]

    normal_dir = os.path.join(args.input_dir, "omnidata_normal")
    normal_names = sorted(glob.glob(os.path.join(normal_dir, "*")), key=lambda x: int(x.split('/')[-1][:-4]))
    depth_names = [ iname.replace('omnidata_normal', 'aligned_dense_depths') for iname in normal_names ]
    depth_names = [depth_names[i] for i in kf_list]

    print(str(datetime.now()) + ': \033[92mI', 'loading depths ...', '\033[0m')
    depths = []
    for name in tqdm(depth_names):
        depth = np.load(name)
        depths.append(depth)

    depths = np.stack(depths, 0)
    aligned_depth_dir = os.path.join(args.input_dir, "aligned_dense_depths")
    all_depth_names = sorted(glob.glob(os.path.join(aligned_depth_dir, "*")), key=lambda x: int(x.split('/')[-1][:-4]))[start:end:image_skip]
    all_depths = []

    for name, pose in tqdm(zip(all_depth_names, sfm_poses)):
        depth = np.load(name)
        all_depths.append(depth)

    if args.init.normal_model_type == 'omnidata':
        normal_dir = os.path.join(args.input_dir, "omnidata_normal")
    else:
        raise NotImplementedError(f'Unknown normal model type {args.init.normal_model_type} exiting')
    
    normal_names = sorted(glob.glob(os.path.join(normal_dir, "*")), key=lambda x: int(x.split('/')[-1][:-4]))
    normal_names = [normal_names[i] for i in kf_list]

    print(str(datetime.now()) + ': \033[92mI', 'loading normals ...', '\033[0m')
    normals = []
    for name in tqdm(normal_names):
        normal = np.load(name)[:3]
        normals.append(normal)

    normals = np.stack(normals, 0)

    all_normal_names = sorted(glob.glob(os.path.join(normal_dir, "*")), key=lambda x: int(x.split('/')[-1][:-4]))[start:end:image_skip]
    all_normals = []

    for name, pose in tqdm(zip(all_normal_names, sfm_poses)):
        normal = np.load(name)
        normal = normal[:3]

        normal = torch.from_numpy(normal).cuda()
        normal = (normal - 0.5) * 2
        normal = normal / torch.norm(normal, dim=0, keepdim=True)
        normal = normal.permute(1, 2, 0)
        normal[:, :, 1] = -normal[:, :, 1]
        normal[:, :, 2] = -normal[:, :, 2]
        normal = torch.einsum('ijk,kl->ijl', normal, torch.from_numpy(np.linalg.inv(pose[:3, :3]).T).cuda().float())
        normal = F.interpolate(normal.permute(2, 0, 1).unsqueeze(0),
                               (int(imgs[0].shape[0]/64)*8, int(imgs[0].shape[1]/64)*8),
                               mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        all_normals.append(normal.cpu().numpy())

    sp_dir = os.path.join(args.input_dir, "sp")
    sp_names = [os.path.join(sp_dir, os.path.split(name)[-1].replace('.jpg', '.npy').replace('.png', '.npy')) for name in image_names_sp]

    if load_plane:
        print(str(datetime.now()) + ': \033[92mI', 'loading superpixels ...', '\033[0m')
        planes_all = []
        new_sp_all = []
        mean_colors_all = []
        plane_index_start = 0
        for plane_idx, (sp_name, depth, normal, pose, img_sp) in tqdm(enumerate(zip(sp_names, depths, normals, sfm_poses_sp, imgs_sp))):
            sp = np.load(sp_name)
            planes, new_sp, mean_colors = parse_sp(sp, intr, pose, depth, normal, img_sp)
            planes = np.array(planes)
            planes_idx = np.ones((planes.shape[0], 1)) * plane_idx
            planes = np.concatenate([planes, planes_idx], 1)
            planes_all.extend(planes)
            new_sp_all.append(new_sp + plane_index_start)
            plane_index_start += planes.shape[0]
            mean_colors = np.array(mean_colors)
            mean_colors_all.extend(mean_colors)

        planes_all = np.array(planes_all)
        mean_colors_all = np.array(mean_colors_all)

    else:
        planes_all = None
        new_sp_all = None
        mean_colors_all = None

    # proj matrixs
    frames_proj = []
    frames_c2w = []
    frames_center = []
    for i in range(len(sfm_camprojs)):
        fovy = 2 * np.arctan(0.5 * imgs[0].shape[0] / intr[1, 1])
        proj = get_projection_matrix(
            fovy, imgs[0].shape[1] / imgs[0].shape[0], 0.1, 1000.0
        )
        proj = np.array(proj)
        frames_proj.append(proj)
        # sfm_poses is w2c
        c2w = np.linalg.inv(sfm_poses[i])
        frames_c2w.append(c2w)
        frames_center.append(c2w[:3, 3])

    frames_proj = np.stack(frames_proj, 0)
    frames_c2w = np.stack(frames_c2w, 0)
    frames_center = np.stack(frames_center, 0)

    mvp_mtxs = get_mvp_matrix(
        frames_c2w, frames_proj,
    )

    index_init = ((np.array(kf_list) - kf_list[0]) / image_skip).astype(np.int32)

    return imgs, intr, sfm_poses, sfm_camprojs, frames_center, all_depths, all_normals, planes_all, \
        mvp_mtxs, index_init, new_sp_all, mean_colors_all
