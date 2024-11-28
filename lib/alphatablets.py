from datetime import datetime
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from geom.plane_utils import simple_distribute_planes_2D, cluster_planes

import nvdiffrast.torch as dr

from tqdm import tqdm, trange
from collections import Counter

from scipy.spatial import Delaunay


def transform_pos(mtx, pos):
    verts_homo = torch.cat(
        [pos, torch.ones([pos.shape[0], 1]).to(pos)], dim=-1
    )
    return torch.matmul(verts_homo, mtx.permute(0, 2, 1))


def init_textures(sp, plane_sets, plane_params, alpha_init, alpha_init_empty):
    print(str(datetime.now()) + f': \033[92mI', 'Init tablet texture ...', '\033[0m')
    imgs, sps, mvp_mtxs = sp
    texs = []

    for plane_set, plane_param in tqdm(zip(plane_sets, plane_params)):
        ori_img_idx = int(plane_param[15])

        mvp_mtx = mvp_mtxs[ori_img_idx]
        cur_sp = sps[ori_img_idx]
        cur_img = imgs[ori_img_idx]
        w, h = plane_param[13:15] - plane_param[11:13]

        plane_normal = plane_param[:3]
        plane_center = plane_param[3:6]
        plane_up = plane_param[6:9]
        reso = plane_param[9:11]
        plane_right = np.cross(plane_normal, plane_up)

        w_grid, h_grid = np.meshgrid(
            np.linspace(plane_param[11], plane_param[13], round(w)) / reso[0],
            np.linspace(plane_param[12], plane_param[14], round(h)) / reso[1]
        )

        pos = w_grid[..., None] * plane_right[None, None, ...] + h_grid[..., None] * plane_up[None, None, ...] + plane_center[None, None, ...]

        with torch.no_grad():
            pos_clip = transform_pos(torch.from_numpy(mvp_mtx[None, ...]), torch.from_numpy(pos).reshape(-1, 3))
            pos_clip = pos_clip / pos_clip[..., 3:4]
            pos_clip = pos_clip.squeeze(0)[:, :2].reshape(round(h), round(w), 2).cuda()
            mask = np.zeros_like(cur_sp)
            for plane_id in plane_set:
                mask += (cur_sp == plane_id)
            masked_img = cur_img * mask[..., None]
            masked_img += (1 - mask[..., None]) * 0.5
            masked_img = torch.from_numpy(masked_img).cuda()
            masked_img = masked_img.permute(2, 0, 1)
            masked_img = masked_img[None, ...]

            interp_tex = F.grid_sample(masked_img, pos_clip[None, ...], mode='bilinear', align_corners=False)[0].permute(1,2,0)
            interp_alpha = F.grid_sample(torch.from_numpy(mask[..., None].transpose(2, 0, 1)[None, ...]).cuda().double()*(alpha_init-alpha_init_empty)+alpha_init_empty+1e-6,
                                         pos_clip[None, ...], mode='bilinear', align_corners=False)[0].permute(1,2,0)

            interp_tex = torch.cat([interp_tex, interp_alpha], dim=-1)[None, ...].cpu().numpy()
            texs.append(interp_tex)

    return texs



def proj_textures(new_pos, new_lookat, new_up,
                  new_x_min, new_x_max, new_y_min, new_y_max,
                  new_H, new_W, v_vt,
                  old_plane_params, old_center, old_n, old_up, old_reso, old_pp, pdb=False):
    # calculate view matrix
    new_right = -np.cross(new_lookat, new_up)
    R = np.array([new_right, new_up, -new_lookat])
    T = np.array([-np.dot(new_right, new_pos), -np.dot(new_up, new_pos), np.dot(new_lookat, new_pos)])
    view_matrix = np.vstack((R, np.array([0, 0, 0])))
    view_matrix = np.column_stack((view_matrix, np.append(T, 1)))

    # scale max(new_H, new_W) to 2048
    if max(new_H, new_W) > 2048:
        new_H /= (max(new_H, new_W) / 2048)
        new_W /= (max(new_H, new_W) / 2048)
        new_H, new_W = int(new_H/8)*8, int(new_W/8)*8

    with torch.no_grad():
        # construct_pseudo_mesh
        t_pos_idx = torch.zeros((old_plane_params.shape[0] * 2, 3), dtype=torch.int32, device=old_plane_params.device)
        t_pos_idx[::2, 0] = torch.arange(0, old_plane_params.shape[0] * 4, 4, device=old_plane_params.device)
        t_pos_idx[::2, 1] = torch.arange(1, old_plane_params.shape[0] * 4 + 1, 4, device=old_plane_params.device)
        t_pos_idx[::2, 2] = torch.arange(2, old_plane_params.shape[0] * 4 + 2, 4, device=old_plane_params.device)

        t_pos_idx[1::2, 0] = torch.arange(0, old_plane_params.shape[0] * 4, 4, device=old_plane_params.device)
        t_pos_idx[1::2, 1] = torch.arange(3, old_plane_params.shape[0] * 4 + 3, 4, device=old_plane_params.device)
        t_pos_idx[1::2, 2] = torch.arange(2, old_plane_params.shape[0] * 4 + 2, 4, device=old_plane_params.device)

        v_pos = torch.zeros((old_plane_params.shape[0] * 4, 3), dtype=torch.float32, device=old_plane_params.device)

        plane_right = torch.cross(old_n, old_up, dim=-1)
        plane_xy_min = old_plane_params[:, 11:13]
        plane_xy_max = old_plane_params[:, 13:15]

        v_pos[::4] = old_center + plane_xy_min[:, 0:1] * plane_right / old_reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * old_up / old_reso[:, 1:2]
        v_pos[1::4] = old_center + plane_xy_min[:, 0:1] * plane_right / old_reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * old_up / old_reso[:, 1:2]
        v_pos[2::4] = old_center + plane_xy_max[:, 0:1] * plane_right / old_reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * old_up / old_reso[:, 1:2]
        v_pos[3::4] = old_center + plane_xy_max[:, 0:1] * plane_right / old_reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * old_up / old_reso[:, 1:2]
        
        view_matrix = torch.from_numpy(view_matrix[np.newaxis, ...]).float().to(old_plane_params.device)

        v_pos_new = transform_pos(view_matrix, v_pos)
        v_pos_new = v_pos_new / v_pos_new[:, :, 3:4]

        # scale z-axis from [min, max] to [-1, 1]
        zmin = v_pos_new[:, :, 2].min()
        zmax = v_pos_new[:, :, 2].max()
        v_pos_new[:, :, 2] = -(v_pos_new[:, :, 2] - (zmax + zmin) / 2) / (zmax - zmin) * 2
        # scale x-axis and y-axis from [min, max] to [-1, 1]
        x_min = new_x_min
        x_max = new_x_max
        y_min = new_y_min
        y_max = new_y_max
        v_pos_new[:, :, 0] = (v_pos_new[:, :, 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
        v_pos_new[:, :, 1] = (v_pos_new[:, :, 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2

        if pdb:
            import pdb; pdb.set_trace()
            
        # rasterize
        colors = []
        alphas = []
        with dr.DepthPeeler(old_pp.glctx, v_pos_new, t_pos_idx, (new_H, new_W)) as peeler:
            weight_thres_checker = torch.ones((1, new_H, new_W), dtype=torch.float32, device=old_plane_params.device)
            while True:
                rast_out, rast_out_db = peeler.rasterize_next_layer()
                if (rast_out[..., 3] == 0).all():
                    break

                texc, texd = dr.interpolate(v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                color = dr.texture(old_pp.tex_color[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=old_pp.max_mip_level)
                color = torch.sigmoid(color)

                alpha = dr.texture(old_pp.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=old_pp.max_mip_level)
                alpha = torch.sigmoid(alpha)

                aa_coloralpha = dr.antialias(color * alpha, rast_out, v_pos_new, t_pos_idx)
                aa_alpha = dr.antialias(alpha, rast_out, v_pos_new, t_pos_idx)
                color = aa_coloralpha / (aa_alpha + 1e-6)

                with torch.no_grad():
                    plane_mask = (rast_out[..., 3:4] != 0).float().squeeze(-1)
                    
                cur_alpha = alpha[..., 0]
                cur_alpha = cur_alpha * plane_mask
                cur_alpha = torch.clamp(cur_alpha, min=1e-6)
                
                colors.append(color[..., :3])
                alphas.append(cur_alpha)

                weight_thres_checker *= (1 - cur_alpha)

        if len(colors) == 0:
            print(str(datetime.now()) + ': \033[93mWarning:', 'no valid tablet found in the current view', '\033[0m')
            return np.zeros((1, new_H, new_W, 4)).astype(np.float32)

        colors = torch.stack(colors, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # alpha blending
        # 0.4, 0.6, 0.7 -> 0.4, (1-0.4) * 0.6, (1-0.4) * (1-0.6) * 0.7
        weights = torch.cumprod(1 - alphas, dim=0)
        weights = torch.cat([torch.ones_like(weights[:1]), weights], dim=0)
        weights = weights * torch.cat([alphas, torch.ones_like(alphas[:1])], dim=0)
        weights = weights[:-1]

        color = torch.sum(weights.unsqueeze(-1) * colors, dim=0)
        alpha = torch.sum(weights, dim=0)
        color /= alpha.unsqueeze(-1)

        return torch.cat([color, alpha.unsqueeze(-1)], dim=-1).cpu().numpy()


class AlphaTablets(nn.Module):
    def __init__(self, plane_params=None, sp=None, cam_centers=None,
                 mean_colors=None, HW=None, pp=None,
                 alpha_init=None, alpha_init_empty=None, ckpt_paths=None,
                 inside_mask_alpha_thres=None,
                 merge_cfgs=None) -> None:
        super(AlphaTablets, self).__init__()
        assert merge_cfgs is not None
        self.merge_cfgs = merge_cfgs

        if ckpt_paths is not None:
            self.alpha_init = alpha_init
            self.alpha_init_empty = alpha_init_empty
            self.inside_mask_alpha_thres = inside_mask_alpha_thres
            self.max_mip_level = 0
            self.construct_planes_from_ckpts(ckpt_paths)
            self.glctx = dr.RasterizeCudaContext()

        elif pp is not None:
            super(AlphaTablets, self).__init__()

            self.cnt = 0
            self.ori_plane_params = pp.proj_ori_planes()

            if pp.cam_ray_mode:
                self.ori_cam_centers = pp.ori_cam_centers.detach()
                self.cam_ray_mode = True
            else:
                self.cam_ray_mode = False

            self.alpha_init = pp.alpha_init
            self.alpha_init_empty = pp.alpha_init_empty
            self.inside_mask_alpha_thres = pp.inside_mask_alpha_thres
            self.max_mip_level = 0

            self.merge_planes(thres=self.merge_cfgs['normal_thres'], init_plane_sets=pp.plane_sets, prev_pp=pp)
            self.construct_planes()

            self.glctx = dr.RasterizeCudaContext()

            self.HW = pp.HW
            self.cnt = pp.cnt

            del pp

        elif plane_params is not None and HW is not None:
            self.ori_plane_params = plane_params
            mean_colors[..., 0] *= merge_cfgs['Y_decay']
            self.ori_plane_params = np.concatenate(
                [self.ori_plane_params, mean_colors], axis=-1
            )

            if cam_centers is not None:
                self.ori_cam_centers = cam_centers
                self.ori_cam_centers = torch.from_numpy(self.ori_cam_centers).float().cuda()
                self.cam_ray_mode = True
            else:
                self.cam_ray_mode = False

            # determine the density bias shift
            self.alpha_init = alpha_init
            self.alpha_init_empty = alpha_init_empty
            self.inside_mask_alpha_thres = inside_mask_alpha_thres

            self.max_mip_level = 0

            self.merge_planes(thres=self.merge_cfgs['normal_thres_init'], per_view_merge=True)
            if sp is not None:
                self.projed_textures = init_textures(sp, self.plane_sets, self.plane_params, self.alpha_init, self.alpha_init_empty)
            self.construct_planes()

            self.glctx = dr.RasterizeCudaContext()

            self.HW = HW
            self.cnt = 0

        else:
            raise NotImplementedError



    def save_ckpt(self, path):
        tex_colors, tex_alphas = [], []
        tex_W, tex_H = self.tex_color.shape[:2]
        for i in range(self.plane_params.shape[0]):
            w_min, h_min = round(self.v_vt[i*4][1].item()*tex_W), round(self.v_vt[i*4][0].item()*tex_H)
            w_max, h_max = round(self.v_vt[i*4+2][1].item()*tex_W), round(self.v_vt[i*4+2][0].item()*tex_H)
            tex_colors.append(self.tex_color[w_min:w_max, h_min:h_max])
            tex_alphas.append(self.tex_alpha[w_min:w_max, h_min:h_max])

        self.plane_params[:, :3] = self.plane_n
        self.plane_params[:, 3:6] = self.plane_center
        plane_n = self.plane_n / torch.norm(self.plane_n, dim=-1, keepdim=True)
        self.plane_params[:, 6:9] = self.calc_plane_up(plane_n)

        torch.save({
            'ori_plane_params': self.proj_ori_planes(),
            'plane_params': self.plane_params,
            'plane_sets': self.plane_sets,
            'plane_n': plane_n,
            'plane_dist': self.plane_dis,
            'plane_up': self.plane_params[:, 6:9],
            'cam_centers': self.ori_cam_centers,
            'ray_dir': self.ray_dir,
            'cam_ray_mode': self.cam_ray_mode,
            'tex_colors': tex_colors,
            'tex_alphas': tex_alphas,
            'HW': self.HW,
            'cnt': self.cnt
        }, path)


    def save_tex(self, path):
        tex_color = torch.cat([
            torch.sigmoid(self.tex_color),
            torch.sigmoid(self.tex_alpha)
        ], dim=-1).detach().cpu().numpy()
        tex_color[:, :, :3] = tex_color[:, :, :3][:, :, ::-1]
        cv2.imwrite(path, (tex_color[::-1] * 255).astype(np.uint8))

    
    def load_tex(self, path):
        tex_color = np.clip(cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255, 1e-4, 1-1e-4)[::-1].copy()
        self.tex_color.data = torch.from_numpy(tex_color[..., :3][..., ::-1].copy()).cuda().float()
        self.tex_alpha.data = torch.from_numpy(tex_color[..., 3:4]).cuda().float()

        self.tex_color.data = torch.log(self.tex_color.data / (1 - self.tex_color.data) + 1e-6)
        self.tex_alpha.data = torch.log(self.tex_alpha.data / (1 - self.tex_alpha.data) + 1e-6)


    def copy(self, bbox, shift):
        # bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        # shift: [x_shift, y_shift, z_shift]

        copy_idxs = []
        for i, plane_center in enumerate(self.plane_center):
            if bbox[0] <= plane_center[0] <= bbox[1] and bbox[2] <= plane_center[1] <= bbox[3] and bbox[4] <= plane_center[2] <= bbox[5]:
                copy_idxs.append(i)
                # print(i)

        if len(copy_idxs) == 0:
            return
        
        self.plane_params = torch.cat([self.plane_params, self.plane_params[copy_idxs]], dim=0)
        self.plane_params[-len(copy_idxs):, 3:6] += torch.from_numpy(np.array(shift)).cuda()

        self.plane_n = nn.Parameter(torch.cat([self.plane_n, self.plane_n[copy_idxs]], dim=0), requires_grad=True)
        self.prev_plane_n = torch.cat([self.prev_plane_n, self.prev_plane_n[copy_idxs]], dim=0)
        self.prev_plane_up = torch.cat([self.prev_plane_up, self.prev_plane_up[copy_idxs]], dim=0)
        self.cam_centers = torch.cat([self.cam_centers, self.cam_centers[copy_idxs] + torch.from_numpy(np.array(shift)).cuda()], dim=0)
        self.ray_dir = torch.cat([self.ray_dir, self.ray_dir[copy_idxs]], dim=0)
        self.plane_dis = nn.Parameter(torch.cat([self.plane_dis, self.plane_dis[copy_idxs]], dim=0), requires_grad=True)
        self.ori_plane_dis = torch.cat([self.ori_plane_dis, self.ori_plane_dis[copy_idxs]], dim=0)
        self.ori_reso = torch.cat([self.ori_reso, self.ori_reso[copy_idxs]], dim=0)
        self.plane_center = torch.cat([self.plane_center, self.plane_center[copy_idxs]], dim=0)
        self.plane_center += torch.from_numpy(np.array(shift)).cuda()

        copy_idxs_for_vt = []
        for i in copy_idxs:
            copy_idxs_for_vt.extend([i*4, i*4+1, i*4+2, i*4+3])
        self.v_vt = nn.Parameter(torch.cat([self.v_vt, self.v_vt[copy_idxs_for_vt]], dim=0), requires_grad=False)

    
    def proj_ori_planes(self):
        # self.ori_plane_params: [N, 6]
        # self.plane_sets: [M], lists of the original plane idxs that a plane plane_params created from
        # self.plane_params: [M, 6]

        # Initialize a tensor to hold the projected plane parameters
        projected_params = self.ori_plane_params.copy()

        new_plane_normals = self.plane_n / torch.norm(self.plane_n, dim=-1, keepdim=True)
        new_plane_centers = self.plane_center
        new_plane_ups = self.calc_plane_up(new_plane_normals)

        new_plane_normals = new_plane_normals.detach().cpu().numpy()
        new_plane_centers = new_plane_centers.detach().cpu().numpy()
        new_plane_ups = new_plane_ups.detach().cpu().numpy()

        # Loop through each new plane
        for i, plane_set in enumerate(self.plane_sets):
            # Extract the parameters of the new plane
            new_plane_normal = new_plane_normals[i]
            new_plane_center = new_plane_centers[i]
            new_plane_up = new_plane_ups[i]

            center_delta = new_plane_center - self.plane_params_backup[i, 3:6]

            # Loop through the original planes that form the new plane
            for ori_plane_idx in plane_set:
                # Extract the parameters of the original plane
                ori_plane_center = self.ori_plane_params[ori_plane_idx, 3:6] + center_delta

                # Project the original plane parameters onto the new plane parameters
                projected_normal = new_plane_normal
                projected_center = ori_plane_center + np.dot(new_plane_center - ori_plane_center, new_plane_normal) * new_plane_normal

                # Update the projected_params tensor
                projected_params[ori_plane_idx, :3] = projected_normal
                projected_params[ori_plane_idx, 3:6] = projected_center
                projected_params[ori_plane_idx, 6:9] = new_plane_up

        return projected_params


    def merge_planes(self, thres, per_view_merge=False, merge_edge_planes=True,
                     init_plane_sets=None, prev_pp=None):
        assert init_plane_sets is None or init_plane_sets and prev_pp
        if init_plane_sets:
            id2prevsp = {}
            for i, plane_set in enumerate(prev_pp.plane_sets):
                for plane_idx in plane_set:
                    id2prevsp[plane_idx] = i
            old_n = prev_pp.plane_n
            old_n = old_n / torch.norm(old_n, dim=-1, keepdim=True)
            old_up = prev_pp.calc_plane_up(old_n)
            old_center = prev_pp.plane_center
            old_reso = prev_pp.reso
            self.projed_textures = []

        self.plane_params = []

        if not per_view_merge:
            plane_sets = cluster_planes(self.ori_plane_params, thres=thres, merge_edge_planes=merge_edge_planes,
                                        init_plane_sets=init_plane_sets,
                                        color_thres_1=self.merge_cfgs['color_thres1'],
                                        color_thres_2=self.merge_cfgs['color_thres2'],
                                        dis_thres_1=self.merge_cfgs['dist_thres1'],
                                        dis_thres_2=self.merge_cfgs['dist_thres2'],)
        else:
            assert init_plane_sets is None
            plane_sets = []
            for view_idx in np.unique(self.ori_plane_params[:, 16]):
                idx_map = np.where(self.ori_plane_params[:, 16] == view_idx)[0]
                cur_plane_sets = cluster_planes(self.ori_plane_params[self.ori_plane_params[:, 16] == view_idx], thres=thres, merge_edge_planes=merge_edge_planes,
                                                color_thres_1=114514.,
                                                color_thres_2=self.merge_cfgs['color_thres2'],
                                                dis_thres_1=self.merge_cfgs['dist_thres1'],
                                                dis_thres_2=self.merge_cfgs['dist_thres2'],)
                for plane_set in cur_plane_sets:
                    plane_sets.append(idx_map[plane_set])

        print(str(datetime.now()) + f': \033[94mTablets merged from', self.ori_plane_params.shape[0], 'to', len(plane_sets), '\033[0m')

        for plane_idxs in plane_sets:
            plane_normals = self.ori_plane_params[plane_idxs, :3].copy()
            plane_centers = self.ori_plane_params[plane_idxs, 3:6].copy()
            plane_ups = self.ori_plane_params[plane_idxs, 6:9].copy()
            plane_rights = np.cross(plane_normals, plane_ups)
            reso = self.ori_plane_params[plane_idxs, 9:11].copy()
            plane_xy_min = self.ori_plane_params[plane_idxs, 11:13].copy()
            plane_xy_max = self.ori_plane_params[plane_idxs, 13:15].copy()
            
            new_normal = np.mean(plane_normals, axis=0)
            new_center = np.mean(plane_centers, axis=0)
            new_plane_up = np.mean(plane_ups, axis=0)
            new_normal = new_normal / np.linalg.norm(new_normal)
            new_plane_up = new_plane_up - np.sum(new_normal * new_plane_up) * new_normal
            new_plane_up = new_plane_up / np.linalg.norm(new_plane_up)

            new_reso = np.mean(self.ori_plane_params[plane_idxs, 9:11], axis=0)

            corners = np.concatenate([
                plane_centers + plane_xy_min[:, 0:1] / reso[:, 0:1] * plane_rights + plane_xy_min[:, 1:2] / reso[:, 1:2] * plane_ups - new_center[None, :],
                plane_centers + plane_xy_min[:, 0:1] / reso[:, 0:1] * plane_rights + plane_xy_max[:, 1:2] / reso[:, 1:2] * plane_ups - new_center[None, :],
                plane_centers + plane_xy_max[:, 0:1] / reso[:, 0:1] * plane_rights + plane_xy_max[:, 1:2] / reso[:, 1:2] * plane_ups - new_center[None, :],
                plane_centers + plane_xy_max[:, 0:1] / reso[:, 0:1] * plane_rights + plane_xy_min[:, 1:2] / reso[:, 1:2] * plane_ups - new_center[None, :],
            ])

            # Project corners onto the new plane's right and up vectors
            new_plane_right = np.cross(new_normal, new_plane_up)
            corners_projected_right = np.dot(corners, new_plane_right)
            corners_projected_up = np.dot(corners, new_plane_up)

            # Calculate the new plane_xy_min and plane_xy_max
            new_plane_xy_min = np.array([
                np.min(corners_projected_right) * new_reso[0],
                np.min(corners_projected_up) * new_reso[1]
            ])
            new_plane_xy_max = np.array([
                np.max(corners_projected_right) * new_reso[0],
                np.max(corners_projected_up) * new_reso[1]
            ])

            if init_plane_sets:
                prev_plane_idxs_to_proj = set([id2prevsp[plane_idx] for plane_idx in plane_idxs])
                prev_plane_idxs_to_proj = list(prev_plane_idxs_to_proj)

                v_vt_idxs = np.concatenate([np.arange(i*4, i*4+4) for i in prev_plane_idxs_to_proj])

                proj_texture = proj_textures(
                    new_center, -new_normal, new_plane_up,
                    np.min(corners_projected_right), np.max(corners_projected_right),
                    np.min(corners_projected_up), np.max(corners_projected_up),
                    int((new_plane_xy_max[1] - new_plane_xy_min[1])/8+1)*8, int((new_plane_xy_max[0] - new_plane_xy_min[0])/8+1)*8,
                    prev_pp.v_vt[v_vt_idxs], prev_pp.plane_params[prev_plane_idxs_to_proj],
                    old_center[prev_plane_idxs_to_proj],
                    old_n[prev_plane_idxs_to_proj], old_up[prev_plane_idxs_to_proj], 
                    old_reso[prev_plane_idxs_to_proj], prev_pp
                )

                try:
                    alpha_mask = proj_texture[0, ..., 3] > 0.03
                    # find the 4 corners of the non-zero alpha region
                    y, x = np.nonzero(alpha_mask)
                    ymin, ymax = np.min(y), np.max(y)
                    xmin, xmax = np.min(x), np.max(x)
                    old_ymin, old_ymax = 0, proj_texture.shape[1]
                    old_xmin, old_xmax = 0, proj_texture.shape[2]
                    right_max, up_max = new_plane_xy_max
                    right_min, up_min = new_plane_xy_min
                    new_right_max = right_max - (old_xmax - xmax) / (old_xmax - old_xmin) * (right_max - right_min)
                    new_right_min = right_min + (xmin - old_xmin) / (old_xmax - old_xmin) * (right_max - right_min)
                    new_up_max = up_max - (old_ymax - ymax) / (old_ymax - old_ymin) * (up_max - up_min)
                    new_up_min = up_min + (ymin - old_ymin) / (old_ymax - old_ymin) * (up_max - up_min)

                    new_plane_xy_min = np.array([new_right_min, new_up_min])
                    new_plane_xy_max = np.array([new_right_max, new_up_max])
                    proj_texture = proj_texture[:, ymin:ymax, xmin:xmax]

                except:
                    print(str(datetime.now()) + ': \033[93mWarning:', 'no valid alpha region found in the projected texture', '\033[0m')
                    pass

                self.projed_textures.append(proj_texture)

            new_plane = np.concatenate([
                new_normal,
                new_center,
                new_plane_up,
                new_reso,
                new_plane_xy_min,
                new_plane_xy_max,
                [np.bincount(self.ori_plane_params[plane_idxs, 16].astype('int64')).argmax()]
            ])

            self.plane_params.append(new_plane)

        self.plane_params = np.stack(self.plane_params, axis=0)
        self.plane_sets = plane_sets


    def construct_planes_from_ckpts(self, ckpt_paths):
        all_texs = []
        all_ori_plane_params = []
        all_plane_params = []
        all_plane_sets = []
        all_plane_n = []
        all_plane_dist = []
        all_ray_dir = []
        all_prev_plane_n = []
        all_prev_plane_up = []
        all_cam_centers = []
        cam_ray_mode, HW = None, None

        view_cnt = 0
        plane_cnt = 0

        with torch.no_grad():
            for ckpt_path in ckpt_paths:
                ckpt = torch.load(ckpt_path)
                ori_plane_params = ckpt['ori_plane_params']
                ori_plane_params[:, 16] += view_cnt
                all_ori_plane_params.append(ori_plane_params)
                plane_params = ckpt['plane_params']
                plane_params[:, 15] += view_cnt
                
                all_plane_params.append(plane_params)
                view_cnt += len(ckpt['cam_centers'])

                plane_sets = ckpt['plane_sets']
                plane_sets = [
                    [i + plane_cnt for i in plane_set] for plane_set in plane_sets
                ]
                all_plane_sets.extend(plane_sets)
                plane_cnt += len(ckpt['ori_plane_params'])

                all_plane_n.append(ckpt['plane_n'])
                all_prev_plane_n.append(ckpt['plane_n'])
                all_prev_plane_up.append(ckpt['plane_up'])
                all_cam_centers.append(ckpt['cam_centers'])
                all_plane_dist.append(ckpt['plane_dist'])
                all_ray_dir.append(ckpt['ray_dir'])
                tex = [ torch.cat([color, alpha], dim=-1) for color, alpha in zip(ckpt['tex_colors'], ckpt['tex_alphas']) ]
                all_texs.extend(tex)

                assert cam_ray_mode is None or cam_ray_mode == ckpt['cam_ray_mode']
                cam_ray_mode = ckpt['cam_ray_mode']

                assert HW is None or HW == ckpt['HW']
                HW = ckpt['HW']

            self.ori_plane_params = np.concatenate(all_ori_plane_params, axis=0)
            self.plane_params = torch.cat(all_plane_params, dim=0)
            self.plane_sets = all_plane_sets
            self.plane_n = torch.cat(all_plane_n, dim=0)
            self.prev_plane_n = torch.cat(all_prev_plane_n, dim=0)
            self.prev_plane_up = torch.cat(all_prev_plane_up, dim=0)
            self.ori_cam_centers = torch.cat(all_cam_centers, dim=0)
            self.plane_dis = torch.cat(all_plane_dist, dim=0)
            self.ray_dir = torch.cat(all_ray_dir, dim=0)
            self.cam_ray_mode = cam_ray_mode
            self.HW = HW
            self.cnt = 0

            uv_ranges = self.plane_params[:, 13:15] - self.plane_params[:, 11:13]
            self.uv_ranges = uv_ranges
            plane_leftup, now_W, H = simple_distribute_planes_2D(self.uv_ranges)
            self.plane_leftup = plane_leftup

            now_W = int(now_W / 4) * 4 + 4
            H = int(H / 4) * 4 + 4

            self.tex_color = torch.full((now_W, H, 3), fill_value=0.5, dtype=torch.float32, device=self.uv_ranges.device)
            self.tex_alpha = torch.full((now_W, H, 1), fill_value=0.5, dtype=torch.float32, device=self.uv_ranges.device)
            self.tex_alpha.fill_(math.log(self.alpha_init_empty+1e-6 / (1-self.alpha_init_empty-1e-6)))
            self.v_vt = torch.zeros((self.plane_params.shape[0] * 4, 2), dtype=torch.float32, device=self.uv_ranges.device)
            # idx % 4 == 0, => self.plane_leftup
            self.v_vt[::4] = self.plane_leftup
            self.v_vt[2::4] = self.plane_leftup + self.uv_ranges
            self.v_vt[1::4] = self.plane_leftup + self.uv_ranges * torch.tensor([0, 1], dtype=torch.float32, device=self.uv_ranges.device)
            self.v_vt[3::4] = self.plane_leftup + self.uv_ranges * torch.tensor([1, 0], dtype=torch.float32, device=self.uv_ranges.device)

            for i in range(self.plane_params.shape[0]):
                w_min, h_min = round(self.v_vt[i*4][0].item()), round(self.v_vt[i*4][1].item())
                w_max, h_max = round(self.v_vt[i*4+2][0].item()), round(self.v_vt[i*4+2][1].item())
                if w_max - w_min > 0 and h_max - h_min > 0 and all_texs[i].shape[0] > 0 and all_texs[i].shape[1] > 0:
                    new_image = F.interpolate(all_texs[i][None].permute(0, 3, 1, 2), (w_max-w_min, h_max-h_min), mode='bilinear')[0].permute(1, 2, 0)
                    self.tex_color[w_min:w_max, h_min:h_max] = new_image[..., 0:3]
                    alpha = torch.sigmoid(new_image[..., 3:4]) * (self.alpha_init - self.alpha_init_empty) + self.alpha_init_empty + 1e-6
                    self.tex_alpha[w_min:w_max, h_min:h_max] = torch.log(alpha / (1 - alpha))

            self.v_vt = torch.cat([self.v_vt[..., 1:2], self.v_vt[..., 0:1]], dim=-1)
            self.v_vt[:, 0] /= H
            self.v_vt[:, 1] /= now_W

        self.tex_color = nn.Parameter(self.tex_color, requires_grad=True)
        self.tex_alpha = nn.Parameter(self.tex_alpha, requires_grad=True)
        self.v_vt = nn.Parameter(self.v_vt, requires_grad=False)

        self.plane_params_backup = self.plane_params.clone().cpu().numpy()

        self.ori_reso = self.plane_params[:, 9:11]

        self.plane_n = nn.Parameter(self.plane_n, requires_grad=True)

        if self.cam_ray_mode:
            self.cam_centers = self.ori_cam_centers[self.plane_params[:, 15].long()]
            self.ori_plane_dis = self.plane_dis.clone()
            self.plane_dis = nn.Parameter(self.plane_dis, requires_grad=True)
            self.plane_center = self.cam_centers + self.plane_dis * self.ray_dir
        else:
            raise NotImplementedError
            self.plane_center = nn.Parameter(self.plane_center, requires_grad=True)

        self.reso = self.ori_reso
    

    def construct_planes(self):
        uv_ranges = self.plane_params[:, 13:15] - self.plane_params[:, 11:13]
        self.uv_ranges = torch.from_numpy(uv_ranges).cuda()
        plane_leftup, now_W, H = simple_distribute_planes_2D(self.uv_ranges)
        self.plane_leftup = plane_leftup

        now_W = int(now_W / 4) * 4 + 4
        H = int(H / 4) * 4 + 4

        self.tex_color = torch.full((now_W, H, 3), fill_value=0.5, dtype=torch.float32, device=self.uv_ranges.device)
        self.tex_alpha = torch.full((now_W, H, 1), fill_value=0.5, dtype=torch.float32, device=self.uv_ranges.device)
        self.tex_alpha.fill_(math.log(self.alpha_init_empty + 1e-6 / (1-self.alpha_init_empty - 1e-6)))
        self.v_vt = torch.zeros((self.plane_params.shape[0] * 4, 2), dtype=torch.float32, device=self.uv_ranges.device)
        # idx % 4 == 0, => self.plane_leftup
        self.v_vt[::4] = self.plane_leftup
        self.v_vt[2::4] = self.plane_leftup + self.uv_ranges
        self.v_vt[1::4] = self.plane_leftup + self.uv_ranges * torch.tensor([0, 1], dtype=torch.float32, device=self.uv_ranges.device)
        self.v_vt[3::4] = self.plane_leftup + self.uv_ranges * torch.tensor([1, 0], dtype=torch.float32, device=self.uv_ranges.device)
        
        if hasattr(self, 'projed_textures'):
            for i in range(self.plane_params.shape[0]):
                w_min, h_min = round(self.v_vt[i*4][0].item()), round(self.v_vt[i*4][1].item())
                w_max, h_max = round(self.v_vt[i*4+2][0].item()), round(self.v_vt[i*4+2][1].item())
                if w_max - w_min > 0 and h_max - h_min > 0:
                    new_image = F.interpolate(torch.from_numpy(self.projed_textures[i]).cuda().permute(0, 3, 1, 2), (h_max-h_min, w_max-w_min), mode='bilinear')[0].permute(2, 1, 0)
                    new_image = torch.clamp(new_image, 1e-6, 1-1e-6)
                    self.tex_color[w_min:w_max, h_min:h_max] = torch.log(new_image[..., 0:3] / (1 - new_image[..., 0:3]))
                    self.tex_alpha[w_min:w_max, h_min:h_max] = torch.log(new_image[..., 3:4] / (1 - new_image[..., 3:4]))

        self.v_vt = torch.cat([self.v_vt[..., 1:2], self.v_vt[..., 0:1]], dim=-1)
        self.v_vt[:, 0] /= H
        self.v_vt[:, 1] /= now_W

        self.tex_color = nn.Parameter(self.tex_color, requires_grad=True)
        self.tex_alpha = nn.Parameter(self.tex_alpha, requires_grad=True)
        self.v_vt = nn.Parameter(self.v_vt, requires_grad=False)

        self.plane_params_backup = self.plane_params.copy()
        self.plane_params = torch.from_numpy(self.plane_params).float().cuda()

        plane_n = self.plane_params[:, :3]
        plane_center = self.plane_params[:, 3:6]
        self.prev_plane_n = plane_n
        self.prev_plane_up = self.plane_params[:, 6:9]

        self.ori_reso = self.plane_params[:, 9:11]

        self.plane_n = nn.Parameter(plane_n, requires_grad=True)

        if self.cam_ray_mode:
            self.cam_centers = self.ori_cam_centers[self.plane_params[:, 15].long()]
            plane_dis = torch.norm(plane_center - self.cam_centers, dim=-1, keepdim=True)
            self.ray_dir = (plane_center - self.cam_centers) / plane_dis
            self.ori_plane_dis = plane_dis.clone()
            self.plane_dis = nn.Parameter(plane_dis, requires_grad=True)
            self.plane_center = self.cam_centers + self.plane_dis * self.ray_dir
        else:
            self.plane_center = nn.Parameter(plane_center, requires_grad=True)

        self.reso = self.ori_reso


    def calc_plane_up(self, plane_n):
        with torch.no_grad():
            u = torch.cross(self.prev_plane_n, plane_n, dim=-1)
            u = u / torch.norm(u, dim=1, keepdim=True)

            cos_theta = torch.sum(self.prev_plane_n * plane_n, dim=1)
            sin_theta = torch.norm(torch.cross(self.prev_plane_n, plane_n, dim=-1), dim=1)

            K = torch.cat([
                torch.cat([torch.zeros(self.plane_params.shape[0], 1, device=self.plane_params.device), -u[:, 2:3], u[:, 1:2]], dim=1),
                torch.cat([u[:, 2:3], torch.zeros(self.plane_params.shape[0], 1, device=self.plane_params.device), -u[:, 0:1]], dim=1),
                torch.cat([-u[:, 1:2], u[:, 0:1], torch.zeros(self.plane_params.shape[0], 1, device=self.plane_params.device)], dim=1),
            ], dim=1).reshape(self.plane_params.shape[0], 3, 3)


            R = cos_theta[:, None, None] * torch.eye(3, device=self.plane_params.device).unsqueeze(0).repeat(self.plane_params.shape[0], 1, 1) \
                + sin_theta[:, None, None] * K \
                + (1 - cos_theta[:, None, None]) * torch.einsum('bi,bj->bij', u, u)
            
            plane_up = torch.einsum('bij,bj->bi', R, self.prev_plane_up)
            self.prev_plane_n = plane_n
            self.prev_plane_up = plane_up

            return plane_up


    def export_mesh(self):
        t_pos_idx = torch.zeros((self.plane_params.shape[0] * 2, 3), dtype=torch.int32, device=self.plane_params.device)
        t_pos_idx[::2, 0] = torch.arange(0, self.plane_params.shape[0] * 4, 4, device=self.plane_params.device)
        t_pos_idx[::2, 1] = torch.arange(1, self.plane_params.shape[0] * 4 + 1, 4, device=self.plane_params.device)
        t_pos_idx[::2, 2] = torch.arange(2, self.plane_params.shape[0] * 4 + 2, 4, device=self.plane_params.device)

        t_pos_idx[1::2, 0] = torch.arange(0, self.plane_params.shape[0] * 4, 4, device=self.plane_params.device)
        t_pos_idx[1::2, 1] = torch.arange(3, self.plane_params.shape[0] * 4 + 3, 4, device=self.plane_params.device)
        t_pos_idx[1::2, 2] = torch.arange(2, self.plane_params.shape[0] * 4 + 2, 4, device=self.plane_params.device)

        v_pos = torch.zeros((self.plane_params.shape[0] * 4, 3), dtype=torch.float32, device=self.plane_params.device)

        # WARNING: the plane normal is not normalized due to training
        plane_n = self.plane_n / torch.norm(self.plane_n, dim=-1, keepdim=True)
        plane_up = self.calc_plane_up(plane_n)
        plane_right = torch.cross(plane_n, plane_up, dim=-1)
        plane_xy_min = self.plane_params[:, 11:13]
        plane_xy_max = self.plane_params[:, 13:15]

        v_pos[::4] = self.plane_center + plane_xy_min[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[1::4] = self.plane_center + plane_xy_min[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[2::4] = self.plane_center + plane_xy_max[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[3::4] = self.plane_center + plane_xy_max[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * plane_up / self.reso[:, 1:2]
        
        self.cnt += 1

        # export mesh
        import trimesh
        mesh = trimesh.Trimesh(vertices=v_pos.detach().cpu().numpy(), faces=t_pos_idx.detach().cpu().numpy())
        mesh.export('mesh_0.obj')

    
    def construct_pseudo_mesh_fast(self, mvp_mtxs):
        t_pos_idx = torch.zeros((self.plane_params.shape[0] * 2, 3), dtype=torch.int32, device=self.plane_params.device)
        t_pos_idx[::2, 0] = torch.arange(0, self.plane_params.shape[0] * 4, 4, device=self.plane_params.device)
        t_pos_idx[::2, 1] = torch.arange(1, self.plane_params.shape[0] * 4 + 1, 4, device=self.plane_params.device)
        t_pos_idx[::2, 2] = torch.arange(2, self.plane_params.shape[0] * 4 + 2, 4, device=self.plane_params.device)

        t_pos_idx[1::2, 0] = torch.arange(0, self.plane_params.shape[0] * 4, 4, device=self.plane_params.device)
        t_pos_idx[1::2, 1] = torch.arange(3, self.plane_params.shape[0] * 4 + 3, 4, device=self.plane_params.device)
        t_pos_idx[1::2, 2] = torch.arange(2, self.plane_params.shape[0] * 4 + 2, 4, device=self.plane_params.device)

        v_pos = torch.zeros((self.plane_params.shape[0] * 4, 3), dtype=torch.float32, device=self.plane_params.device)

        # WARNING: the plane normal is not normalized due to training
        plane_n = self.plane_n / torch.norm(self.plane_n, dim=-1, keepdim=True)
        plane_up = self.calc_plane_up(plane_n)

        plane_right = torch.cross(plane_n, plane_up, dim=-1)
        plane_xy_min = self.plane_params[:, 11:13]
        plane_xy_max = self.plane_params[:, 13:15]

        v_pos[::4] = self.plane_center + plane_xy_min[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[1::4] = self.plane_center + plane_xy_min[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[2::4] = self.plane_center + plane_xy_max[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_max[:, 1:2] * plane_up / self.reso[:, 1:2]
        v_pos[3::4] = self.plane_center + plane_xy_max[:, 0:1] * plane_right / self.reso[:, 0:1] \
                + plane_xy_min[:, 1:2] * plane_up / self.reso[:, 1:2]
        
        self.cnt += 1
        
        v_pos_clip = transform_pos(mvp_mtxs, v_pos)
        plane_view_idx = self.plane_params[:, 16:17]

        return v_pos_clip, t_pos_idx, plane_view_idx, v_pos


    def density_degradation(self, factor=0.8):
        with torch.no_grad():
            self.tex[..., 3] *= factor


    def export_ply(self, mvp_mtxs, alpha_thr=0.5, name=None, render_reso=None):
        print(str(datetime.now()) + ': \033[92mI', 'exporting ply ...', '\033[0m')
        import random
        def generate_random_color():
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        def create_color_pool(pool_size):
            color_pool = [(0, 0, 0)]
            while len(color_pool) < pool_size*2+1:
                new_color = generate_random_color()
                color_pool.append(new_color)
                color_pool.append(new_color)
            color_pool = np.array(color_pool, dtype=np.uint8)
            return color_pool

        color_pool = create_color_pool(self.plane_params.shape[0])
        point_with_color = []

        if not render_reso:
            render_reso = (int(self.HW[0]/64)*8, int(self.HW[1]/64)*8)

        for i in range(len(mvp_mtxs)):
            mvp_mtx = mvp_mtxs[i:i+1]

            with torch.no_grad():
                # get points by rasterize
                v_pos_clip, t_pos_idx, plane_idx, v_pos = self.construct_pseudo_mesh_fast(mvp_mtx)
                with dr.DepthPeeler(self.glctx, v_pos_clip, t_pos_idx, render_reso) as peeler:
                    counter = 0
                    while True:
                        rast_out, rast_out_db = peeler.rasterize_next_layer()
                        if (rast_out[..., 3] == 0).all():
                            break

                        texc, texd = dr.interpolate(self.v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                        alpha = dr.texture(self.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)[0]
                        alpha = torch.sigmoid(alpha[..., 0])

                        point_3d_pos = dr.interpolate(v_pos, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')[0][0]

                        rast_out = rast_out[0]
                        mask = rast_out[..., 3] > 0
                        cur_alpha = alpha[mask]
                        cur_point_3d_pos = point_3d_pos[mask]
                        tri_id = rast_out[..., 3][mask]

                        new_mask = cur_alpha > alpha_thr
                        cur_alpha = cur_alpha[new_mask]
                        cur_point_3d_pos = cur_point_3d_pos[new_mask]
                        tri_id = tri_id[new_mask]
                        cur_color = color_pool[tri_id.detach().cpu().numpy().astype(np.int32)]
                        cur_color = torch.from_numpy(cur_color).float().cuda()

                        # mask -> 0
                        masked_alpha = alpha.clone()
                        masked_alpha[~mask] = 0
                        counter += 1
                        cur_point_with_color = torch.cat([cur_point_3d_pos, cur_color], dim=-1)
                        point_with_color.append(cur_point_with_color.detach().cpu().numpy())

        point_with_color = np.concatenate(point_with_color, axis=0)

        # export ply
        import trimesh
        mesh = trimesh.Trimesh(vertices=point_with_color[:, :3], vertex_colors=point_with_color[:, 3:]/255)
        if name is None:
            mesh.export('mesh_{}.ply'.format(self.cnt))
        else:
            mesh.export(name)


    def weightCheck(self, all_mvp_mtxs, render_reso=None, pixel_thres=10, weight_thres=0.3):
        assert all_mvp_mtxs.shape[1] == 4 and all_mvp_mtxs.shape[2] == 4

        if self.cam_ray_mode:
            self.plane_center = self.cam_centers + self.plane_dis * self.ray_dir
            self.reso = self.ori_reso * (self.ori_plane_dis / self.plane_dis)
        else:
            self.reso = self.ori_reso

        if not render_reso:
            render_reso = (int(self.HW[0]/64)*8, int(self.HW[1]/64)*8)

        counter = Counter()
        plane2points = {}

        with torch.no_grad():
            all_pids = []
            all_coords = []
            for i in trange(len(all_mvp_mtxs)):
                mvp_mtx = all_mvp_mtxs[i:i+1]
                alphas = []
                tri_ids = []
                tex_coords = []

                # get points by rasterize
                v_pos_clip, t_pos_idx, plane_idx, _ = self.construct_pseudo_mesh_fast(mvp_mtx)
                with dr.DepthPeeler(self.glctx, v_pos_clip, t_pos_idx, render_reso) as peeler:
                    while True:
                        rast_out, rast_out_db = peeler.rasterize_next_layer()
                        if (rast_out[..., 3] == 0).all():
                            break

                        texc, texd = dr.interpolate(self.v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                        alpha = dr.texture(self.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)[0]
                        alpha = torch.sigmoid(alpha[..., 0])

                        rast_out = rast_out[0]
                        plane_mask = (rast_out[..., 3:4] != 0).float().squeeze(-1)

                        cur_alpha = alpha * plane_mask
                        cur_alpha = torch.clamp(cur_alpha, min=1e-6)

                        tri_id = rast_out[..., 3]

                        alphas.append(cur_alpha)
                        tri_ids.append(tri_id)
                        tex_coords.append(texc[0])

                if len(alphas) == 0:
                    continue
                alphas = torch.stack(alphas, dim=0)

                weights = torch.cumprod(1 - alphas, dim=0)
                weights = torch.cat([torch.ones_like(weights[:1]), weights], dim=0)
                weights = weights * torch.cat([alphas, torch.ones_like(alphas[:1])], dim=0)
                weights = weights[:-1]

                for ii in range(weights.shape[0]):
                    weight = weights[ii]
                    tri_id = tri_ids[ii]
                    texc = tex_coords[ii]

                    tri_id[weight < weight_thres] = 0
                    texc = texc[tri_id != 0]
                    tri_id = tri_id[tri_id != 0]
                    tri_id = torch.round((tri_id+0.1)/2).long()-1

                    all_pids.append(tri_id)
                    all_coords.append(texc)

                    new_count = Counter(tri_id.cpu().numpy())
                    for index, count in new_count.items():
                        counter[index] = max(counter[index], count)

            all_pids = torch.cat(all_pids, dim=0)
            all_coords = torch.cat(all_coords, dim=0)

            for pid in torch.unique(all_pids):
                pid = pid.item()
                plane2points[pid] = all_coords[all_pids == pid]
                ymax, ymin = plane2points[pid][:, 0].max(), plane2points[pid][:, 0].min()
                xmax, xmin = plane2points[pid][:, 1].max(), plane2points[pid][:, 1].min()
                old_ymax, old_ymin = self.v_vt[pid*4:pid*4+4, 0].max(), self.v_vt[pid*4:pid*4+4, 0].min()
                old_xmax, old_xmin = self.v_vt[pid*4:pid*4+4, 1].max(), self.v_vt[pid*4:pid*4+4, 1].min()
                right_min, up_min = self.plane_params[pid, 11:13]
                right_max, up_max = self.plane_params[pid, 13:15]
                new_right_max = right_max - (old_xmax - xmax) / (old_xmax - old_xmin) * (right_max - right_min)
                new_right_min = right_min + (xmin - old_xmin) / (old_xmax - old_xmin) * (right_max - right_min)
                new_up_max = up_max - (old_ymax - ymax) / (old_ymax - old_ymin) * (up_max - up_min)
                new_up_min = up_min + (ymin - old_ymin) / (old_ymax - old_ymin) * (up_max - up_min)

                self.plane_params[pid, 11] = new_right_min
                self.plane_params[pid, 12] = new_up_min
                self.plane_params[pid, 13] = new_right_max
                self.plane_params[pid, 14] = new_up_max
                self.v_vt[pid*4, 0] = ymin
                self.v_vt[pid*4+3, 0] = ymin
                self.v_vt[pid*4+1:pid*4+3, 0] = ymax
                self.v_vt[pid*4:pid*4+2, 1] = xmin
                self.v_vt[pid*4+2:pid*4+4, 1] = xmax

        retain_plane_idxs = []
        for index, count in counter.items():
            if count > pixel_thres:
                retain_plane_idxs.append(index)

        retain_plane_idxs = sorted(retain_plane_idxs)
        self.plane_sets = [self.plane_sets[ii] for ii in retain_plane_idxs]
        retain_ori_plane_idxs = []
        for plane_set in self.plane_sets:
            retain_ori_plane_idxs.extend(plane_set)
        retain_ori_plane_idxs = sorted(retain_ori_plane_idxs)

        retain_plane_vt_idxs = []
        for plane_idx in retain_plane_idxs:
            retain_plane_vt_idxs.extend([plane_idx*4, plane_idx*4+1, plane_idx*4+2, plane_idx*4+3])

        ori_plane_idx_map = {idx: i for i, idx in enumerate(retain_ori_plane_idxs)}

        self.plane_sets = [
            [ori_plane_idx_map[plane_idx] for plane_idx in plane_set]
            for plane_set in self.plane_sets
        ]
        
        self.ori_plane_params = self.ori_plane_params[retain_ori_plane_idxs]
        self.ori_reso = self.ori_reso[retain_plane_idxs]
        self.ori_plane_dis = self.ori_plane_dis[retain_plane_idxs]
        if self.cam_ray_mode:
            self.plane_center = self.plane_center[retain_plane_idxs]
            self.plane_dis = nn.Parameter(self.plane_dis[retain_plane_idxs], requires_grad=True)
        else:
            self.plane_center = nn.Parameter(self.plane_center[retain_plane_idxs], requires_grad=True)
        self.reso = self.reso[retain_plane_idxs]
        self.plane_params = self.plane_params[retain_plane_idxs]
        self.plane_n = nn.Parameter(self.plane_n[retain_plane_idxs], requires_grad=True)
        self.prev_plane_n = self.prev_plane_n[retain_plane_idxs]
        self.prev_plane_up = self.prev_plane_up[retain_plane_idxs]
        self.cam_centers = self.cam_centers[retain_plane_idxs]
        self.ray_dir = self.ray_dir[retain_plane_idxs]
        self.plane_params_backup = self.plane_params_backup[retain_plane_idxs]
        self.v_vt = nn.Parameter(self.v_vt[retain_plane_vt_idxs], requires_grad=False)

        print(str(datetime.now()) + ': \033[94mTablet num after weight check:', len(retain_plane_idxs), '\033[0m')


    def export_mesh_with_weight_check(self, all_mvp_mtxs, render_reso=None, pixel_thres=10, weight_thres=0.3, connect_thres=60, name=None):
        assert all_mvp_mtxs.shape[1] == 4 and all_mvp_mtxs.shape[2] == 4

        if self.cam_ray_mode:
            self.plane_center = self.cam_centers + self.plane_dis * self.ray_dir
            self.reso = self.ori_reso * (self.ori_plane_dis / self.plane_dis)
        else:
            self.reso = self.ori_reso

        if not render_reso:
            render_reso = (int(self.HW[0]/64)*8, int(self.HW[1]/64)*8)

        counter = Counter()
        plane2points = {}

        with torch.no_grad():
            all_pids = []
            all_coords = []
            for i in trange(len(all_mvp_mtxs)):
                mvp_mtx = all_mvp_mtxs[i:i+1]
                alphas = []
                tri_ids = []
                tex_coords = []

                # get points by rasterize
                v_pos_clip, t_pos_idx, plane_idx, v_pos = self.construct_pseudo_mesh_fast(mvp_mtx)
                with dr.DepthPeeler(self.glctx, v_pos_clip, t_pos_idx, render_reso) as peeler:
                    while True:
                        rast_out, rast_out_db = peeler.rasterize_next_layer()
                        if (rast_out[..., 3] == 0).all():
                            break

                        texc, texd = dr.interpolate(self.v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                        alpha = dr.texture(self.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)[0]
                        alpha = torch.sigmoid(alpha[..., 0])

                        rast_out = rast_out[0]
                        plane_mask = (rast_out[..., 3:4] != 0).float().squeeze(-1)

                        cur_alpha = alpha * plane_mask
                        cur_alpha = torch.clamp(cur_alpha, min=1e-6)

                        tri_id = rast_out[..., 3]

                        alphas.append(cur_alpha)
                        tri_ids.append(tri_id)
                        tex_coords.append(texc[0])

                if len(alphas) == 0:
                    continue
                alphas = torch.stack(alphas, dim=0)

                weights = torch.cumprod(1 - alphas, dim=0)
                weights = torch.cat([torch.ones_like(weights[:1]), weights], dim=0)
                weights = weights * torch.cat([alphas, torch.ones_like(alphas[:1])], dim=0)
                weights = weights[:-1]

                for ii in range(weights.shape[0]):
                    weight = weights[ii]
                    tri_id = tri_ids[ii]
                    texc = tex_coords[ii]

                    tri_id[weight < weight_thres] = 0
                    texc = texc[tri_id != 0]
                    tri_id = tri_id[tri_id != 0]
                    tri_id = torch.round((tri_id+0.1)/2).long()-1

                    all_pids.append(tri_id)
                    all_coords.append(texc)

                    new_count = Counter(tri_id.cpu().numpy())
                    for index, count in new_count.items():
                        counter[index] = max(counter[index], count)

            all_pids = torch.cat(all_pids, dim=0)
            all_coords = torch.cat(all_coords, dim=0)

            vs = []
            vts = []
            fs = []
            pixel_nums = []
            instance_ids = []
            now_v_index_start = 0
            now_instance_id_start = 0

            class DisjointSetPlain:
                def __init__(self, size):
                    self.parent = [i for i in range(size)]

                def find(self, x):
                    if self.parent[x] != x:
                        self.parent[x] = self.find(self.parent[x])
                    return self.parent[x]

                def union(self, x, y):
                    root_x = self.find(x)
                    root_y = self.find(y)
                    self.parent[root_x] = root_y

            print(str(datetime.now()) + ': \033[92mI', 'Pruning Tablets ...', '\033[0m')
            for pid in tqdm(torch.unique(all_pids)):
                pid = pid.item()
                if counter[pid] <= pixel_thres:
                    continue
                cur_plane_vs = []
                cur_plane_vts = []
                plane2points[pid] = all_coords[all_pids == pid]

                tex_W, tex_H = self.tex_color.shape[:2]
                w_min, h_min = round(self.v_vt[pid*4][1].item()*tex_W), round(self.v_vt[pid*4][0].item()*tex_H)
                w_max, h_max = round(self.v_vt[pid*4+2][1].item()*tex_W), round(self.v_vt[pid*4+2][0].item()*tex_H)
                points = plane2points[pid].cpu().numpy()
                points_vt = points.copy()
                points = np.round(points * np.array([tex_H-1, tex_W-1])).astype(np.int32)
                points = points - np.array([h_min, w_min])

                # print(f'try delaunay points', points.shape)
                if len(points) > 50000:
                    # print('random sample to 50000 points')
                    idx = np.random.choice(len(points), 50000, replace=False)
                    points = points[idx]
                    points_vt = points_vt[idx]
                try:
                    tri = Delaunay(points)
                except:
                    tri = Delaunay(points, qhull_options='QJ')

                p1 = points[tri.simplices][:, 0]
                p2 = points[tri.simplices][:, 1]
                p3 = points[tri.simplices][:, 2]
                dis1 = np.linalg.norm(p1 - p2, axis=-1)
                dis2 = np.linalg.norm(p2 - p3, axis=-1)
                dis3 = np.linalg.norm(p3 - p1, axis=-1)
                mask = np.logical_and(np.logical_and(dis1 < connect_thres, dis2 < connect_thres), dis3 < connect_thres)
                
                faces = tri.simplices[mask]

                wmap = np.zeros((h_max-h_min+1, w_max-w_min+1))
                for simplex in faces:
                    cv2.fillConvexPoly(wmap, points[simplex][:, ::-1], 1)
                # dilate wmap
                wmap = cv2.dilate(wmap, np.ones((3, 3), np.uint8), iterations=1)

                ds = DisjointSetPlain(len(points))
                for face in faces:
                    ds.union(face[0], face[1])
                    ds.union(face[0], face[2])

                root2idx = {}
                for i in range(len(points)):
                    root = ds.find(i)
                    if root not in root2idx:
                        root2idx[root] = []
                    root2idx[root].append(i)

                vertex_sets = []
                for root in root2idx:
                    if len(root2idx[root]) > 10:
                        vertex_sets.append(set(root2idx[root]))

                f_num = len(fs)

                for vertex_set in vertex_sets:
                    edge = { i:[] for i in vertex_set }
                    boundary_verts = []
                    for face in faces:
                        if face[0] in vertex_set:
                            edge[face[0]].append(face[1])
                            edge[face[0]].append(face[2])
                            edge[face[1]].append(face[0])
                            edge[face[1]].append(face[2])
                            edge[face[2]].append(face[0])
                            edge[face[2]].append(face[1])

                    for v1 in vertex_set:
                        if len(np.unique(edge[v1])) * 2 != len(edge[v1]):
                            boundary_verts.append(v1)

                    cur_plane_vs.append(points[boundary_verts])
                    cur_plane_vts.append(points_vt[boundary_verts])
                    idxdict = {}
                    for idx, v in enumerate(boundary_verts):
                        idxdict[v] = idx
                    
                    boundary_conn = {}
                    for v in boundary_verts:
                        for v2 in np.unique(edge[v]):
                            if v2 in boundary_verts:
                                if v not in boundary_conn:
                                    boundary_conn[v] = set()
                                boundary_conn[v].add(v2)

                    # print(f'try delaunay points', points.shape)
                    try:
                        new_tri = Delaunay(points[boundary_verts])
                    except:
                        new_tri = Delaunay(points[boundary_verts], qhull_options='QJ')

                    for face in new_tri.simplices:
                        # check 3 midpoints whether is 1 in wmap
                        midpoint1 = np.round((points[boundary_verts[face[0]]] + points[boundary_verts[face[1]]]) / 2).astype('int32')
                        midpoint2 = np.round((points[boundary_verts[face[1]]] + points[boundary_verts[face[2]]]) / 2).astype('int32')
                        midpoint3 = np.round((points[boundary_verts[face[2]]] + points[boundary_verts[face[0]]]) / 2).astype('int32')

                        if wmap[midpoint1[0], midpoint1[1]] == 1 and wmap[midpoint2[0], midpoint2[1]] == 1 and wmap[midpoint3[0], midpoint3[1]] == 1:
                            fs.append([idxdict[boundary_verts[face[0]]] + now_v_index_start,
                                       idxdict[boundary_verts[face[1]]] + now_v_index_start,
                                       idxdict[boundary_verts[face[2]]] + now_v_index_start])

                    now_v_index_start += len(boundary_verts)

                if len(cur_plane_vs) == 0:
                    continue
                cur_plane_vs = np.concatenate(cur_plane_vs, axis=0)
                cur_plane_vs = torch.from_numpy(cur_plane_vs).float().cuda()
                cur_plane_vts = np.concatenate(cur_plane_vts, axis=0)
                cur_plane_vts = torch.from_numpy(cur_plane_vts).float().cuda()

                right_min, up_min = self.plane_params[pid, 11:13]
                right_max, up_max = self.plane_params[pid, 13:15]

                new_rights = right_max - (w_max-w_min+1 - cur_plane_vs[:, 1:2]) / (w_max-w_min+1) * (right_max - right_min)
                new_ups = up_max - (h_max-h_min+1 - cur_plane_vs[:, 0:1]) / (h_max-h_min+1) * (up_max - up_min)

                plane_right = torch.cross(self.prev_plane_n[pid:pid+1], self.prev_plane_up[pid:pid+1], dim=-1)

                v_pos = self.plane_center[pid:pid+1] + new_rights * plane_right / self.reso[pid:pid+1, 0:1] \
                    + new_ups * self.prev_plane_up[pid:pid+1] / self.reso[pid:pid+1, 1:2]
                
                vs.append(v_pos)
                vts.append(cur_plane_vts)
                pixel_nums.append(np.ones(len(fs) - f_num) * counter[pid])
                instance_ids.append(np.ones(len(fs) - f_num) * now_instance_id_start)

                now_instance_id_start += 1

            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(torch.cat(vs, dim=0).detach().cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(fs)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(torch.cat(vts, dim=0).detach().cpu().numpy())

            with open(name, 'w') as f:
                # write obj
                f.write('mtllib test.mtl\n')
                for v in mesh.vertices:
                    f.write('v ' + ' '.join(map(str, v)) + '\n')
                for vt in mesh.triangle_uvs:
                    f.write('vt ' + ' '.join([str(vt[0]), str(1-vt[1])]) + '\n')
                f.write('usemtl test\n')
                for face in mesh.triangles:
                    f.write('f ' + ' '.join([str(v+1) + '/' + str(v+1) for v in face]) + '\n')

            with open(os.path.dirname(name) + '/test.mtl', 'w') as f:
                f.write('newmtl test\n')
                f.write('Ka 1.000 1.000 1.000\n')
                f.write('Kd 1.000 1.000 1.000\n')
                f.write('Ks 0.000 0.000 0.000\n')
                f.write('d 1.0\n')
                f.write('illum 1\n')
                f.write('map_Kd texture.png\n')

            texture = torch.sigmoid(self.tex_color) / (torch.sigmoid(self.tex_alpha) + 1e-6)
            texture = texture.detach().cpu().numpy()
            texture = np.clip(texture, 0, 1)
            texture = (texture * 255).astype(np.uint8)
            cv2.imwrite(os.path.dirname(name) + '/texture.png', texture[..., ::-1])

            pixel_nums = np.concatenate(pixel_nums, axis=0)
            with open(name[:-4] + '_pixel_num.txt', 'w') as f:
                for num in pixel_nums:
                    f.write(str(num) + '\n')

            instance_ids = np.concatenate(instance_ids, axis=0)
            with open(name[:-4] + '_instance_id.txt', 'w') as f:
                for num in instance_ids:
                    f.write(str(num) + '\n')


    def forward(self, mvp_mtx, optimize_geo=True, optimize_tex=True, \
                render_normal_reso=None, return_alpha=False, max_rasterize_layers=15):
        '''Volume rendering
        @mvp_mtx: [N, 4, 4]
        '''
        assert mvp_mtx.shape[1] == 4 and mvp_mtx.shape[2] == 4

        self.cnt += 1

        if self.cam_ray_mode:
            self.plane_center = self.cam_centers + self.plane_dis * self.ray_dir
            self.reso = self.ori_reso * (self.ori_plane_dis / self.plane_dis)
        else:
            self.reso = self.ori_reso

        v_pos_clip, t_pos_idx, _, _ = self.construct_pseudo_mesh_fast(mvp_mtx)

        if not optimize_geo:
            v_pos_clip = v_pos_clip.detach()
            t_pos_idx = t_pos_idx.detach()

        colors = []
        alphas = []
        normals = []
        pos_3ds = []
        depths = []

        weight_thres_checker = torch.ones((mvp_mtx.shape[0], *self.HW), dtype=torch.float32, device=self.plane_params.device)
        layer_cnt = 0
        plane_n = self.plane_n / torch.norm(self.plane_n, dim=-1, keepdim=True)
        plane_n = torch.cat([torch.zeros_like(plane_n[:1, :], device=plane_n.device), plane_n], dim=0)
        with dr.DepthPeeler(self.glctx, v_pos_clip, t_pos_idx, self.HW) as peeler:
            while True:
                layer_cnt += 1
                rast_out, rast_out_db = peeler.rasterize_next_layer()
                if (rast_out[..., 3] == 0).all() or (weight_thres_checker < 1e-3).all():
                    break
                if layer_cnt > max_rasterize_layers:
                    break

                if optimize_tex:
                    texc, texd = dr.interpolate(self.v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                else:
                    v_vt = self.v_vt.detach()
                    texc, texd = dr.interpolate(v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                
                color = dr.texture(self.tex_color[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
                color = torch.sigmoid(color)

                alpha = dr.texture(self.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=self.max_mip_level)
                alpha = torch.sigmoid(alpha)

                depth = rast_out[..., 2:3]
                depth = -1 / ((depth - 1) / 2 * 1/0.1 + (depth + 1) / 2 * 1/1000)

                aa_coloralpha = dr.antialias(color * alpha, rast_out, v_pos_clip, t_pos_idx)
                aa_alpha = dr.antialias(alpha, rast_out, v_pos_clip, t_pos_idx)
                color = aa_coloralpha / (aa_alpha + 1e-6)

                aa_depthalpha = dr.antialias(depth * alpha, rast_out, v_pos_clip, t_pos_idx)
                depth = aa_depthalpha / (aa_alpha + 1e-6)

                pos_3d, _ = dr.interpolate(v_pos_clip, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')

                with torch.no_grad():
                    plane_mask = (rast_out[..., 3:4] != 0).float().squeeze(-1)
                    if layer_cnt == 1:
                        # Now inside mask only works for the first layer
                        inside_mask = F.conv2d((rast_out[..., 3:4]!=0).float().permute(0, 3, 1, 2), torch.ones(1, 1, 5, 5).to(rast_out.device), padding=2)
                        inside_mask = (inside_mask.permute(0, 2, 3, 1) == 25)
                        plane_inside_mask = F.conv2d(rast_out[..., 3:4].permute(0, 3, 1, 2), torch.ones(1, 1, 5, 5).to(rast_out.device), padding=2)
                        plane_inside_mask = (plane_inside_mask.permute(0, 2, 3, 1) == rast_out[..., 3:4] * 25)
                        plane_inside_mask = torch.logical_and(plane_inside_mask, alpha > self.inside_mask_alpha_thres)

                cur_alpha = alpha[..., 0]
                cur_alpha = cur_alpha * plane_mask
                cur_alpha = torch.clamp(cur_alpha, min=1e-6)

                depth = depth[..., 0]
                depth = depth * plane_mask

                colors.append(color[..., :3])
                alphas.append(cur_alpha)
                pos_3ds.append(pos_3d)
                depths.append(depth)

                weight_thres_checker *= (1 - cur_alpha)

        alphas_lowres = []
        if render_normal_reso is None:
            render_normal_reso = (int(self.HW[0]/64)*8, int(self.HW[1]/64)*8)
        with dr.DepthPeeler(self.glctx, v_pos_clip, t_pos_idx, render_normal_reso) as peeler:
            for _ in range(layer_cnt-1):
                rast_out, rast_out_db = peeler.rasterize_next_layer()

                texc, texd = dr.interpolate(self.v_vt, rast_out, t_pos_idx, rast_db=rast_out_db, diff_attrs='all')
                alpha = dr.texture(self.tex_alpha[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=2)
                alpha = torch.sigmoid(alpha)

                cur_normal = plane_n[torch.round((rast_out[..., 3]+0.5)/2).long().view(-1)]
                cur_normal = cur_normal.view(*rast_out.shape[:3], 3)
                
                aa_normalalpha = dr.antialias(cur_normal * alpha, rast_out, v_pos_clip, t_pos_idx)
                aa_alpha = dr.antialias(alpha, rast_out, v_pos_clip, t_pos_idx)
                cur_normal = aa_normalalpha / (aa_alpha + 1e-6)
                
                with torch.no_grad():
                    plane_mask = (rast_out[..., 3:4] != 0).float().squeeze(-1)

                cur_alpha = alpha[..., 0]
                cur_alpha = cur_alpha * plane_mask
                cur_alpha = torch.clamp(cur_alpha, min=1e-6)
                
                alphas_lowres.append(cur_alpha)
                normals.append(cur_normal)

        if len(colors) == 0:
            print(str(datetime.now()) + ': \033[93mWarning:', 'No tablet rendered', '\033[0m')
            return None
        
        colors = torch.stack(colors, dim=0)
        alphas = torch.stack(alphas, dim=0)
        normals = torch.stack(normals, dim=0)
        pos_3ds = torch.stack(pos_3ds, dim=0)
        alphas_lowres = torch.stack(alphas_lowres, dim=0)
        depths = torch.stack(depths, dim=0)

        weight_thres_checker = weight_thres_checker.clamp(min=1e-8, max=1-1e-8)
        alpha_loss = torch.mean(weight_thres_checker)

        # alpha blending
        # 0.4, 0.6, 0.7 -> 0.4, (1-0.4) * 0.6, (1-0.4) * (1-0.6) * 0.7
        weights = torch.cumprod(1 - alphas, dim=0)
        weights = torch.cat([torch.ones_like(weights[:1]), weights], dim=0)
        weights = weights * torch.cat([alphas, torch.ones_like(alphas[:1])], dim=0)
        weights = weights[:-1]

        weights_lowres = torch.cumprod(1 - alphas_lowres, dim=0)
        weights_lowres = torch.cat([torch.ones_like(weights_lowres[:1]), weights_lowres], dim=0)
        weights_lowres = weights_lowres * torch.cat([alphas_lowres, torch.ones_like(alphas_lowres[:1])], dim=0)
        weights_lowres = weights_lowres[:-1]

        # dis difference
        dis = torch.norm(pos_3ds[1:] - pos_3ds[:-1], dim=-1)
        distort_loss = torch.mean(dis * weights[:1] * weights[1:])

        color = torch.sum(weights.unsqueeze(-1) * colors, dim=0)
        normal = torch.sum(weights_lowres.unsqueeze(-1) * normals, dim=0)
        depth = torch.sum(weights * depths, dim=0)
        alpha = torch.sum(weights, dim=0)

        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        depth /= (alpha + 1e-6)

        if return_alpha:
            return color, alpha

        return color, normal, depth, alpha_loss, distort_loss, inside_mask, plane_inside_mask
