import numpy as np
import torch

from sklearn.neighbors import KDTree
from tqdm import trange


class DisjointSet:
    def __init__(self, size, params, mean_colors):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size
        self.accum_normal = [_ for _ in params[:, :3].copy()]
        self.accum_center = [_ for _ in params[:, 3:].copy()]
        self.accum_color = [_ for _ in mean_colors.copy()]
        self.accum_num = [1] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y, normal_thres=0.9, dis_thres=0.1, color_thres=1.):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y and np.abs(
            np.sum(self.accum_normal[root_x] * self.accum_normal[root_y]) /
            np.linalg.norm(self.accum_normal[root_x]) /
            np.linalg.norm(self.accum_normal[root_y])
        ) > normal_thres and np.linalg.norm(
            (self.accum_center[root_x] / self.accum_num[root_x] -
             self.accum_center[root_y] / self.accum_num[root_y])
            * self.accum_normal[root_x] / np.linalg.norm(self.accum_normal[root_x])
        ) < dis_thres and np.linalg.norm(
            self.accum_color[root_x] / self.accum_num[root_x] -
            self.accum_color[root_y] / self.accum_num[root_y]
        ) < color_thres:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
                self.accum_normal[root_y] += self.accum_normal[root_x]
                self.accum_center[root_y] += self.accum_center[root_x]
                self.accum_color[root_y] += self.accum_color[root_x]
                self.accum_num[root_y] += self.accum_num[root_x]
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.accum_normal[root_x] += self.accum_normal[root_y]
                self.accum_center[root_x] += self.accum_center[root_y]
                self.accum_color[root_x] += self.accum_color[root_y]
                self.accum_num[root_x] += self.accum_num[root_y]
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
                self.accum_normal[root_y] += self.accum_normal[root_x]
                self.accum_center[root_y] += self.accum_center[root_x]
                self.accum_color[root_y] += self.accum_color[root_x]
                self.accum_num[root_y] += self.accum_num[root_x]


def calc_plane(K, pose, depth, dis, mask, x_range_uv, y_range_uv, plane_normal=None):
    """Calculate the plane parameters from the camera pose and the pixel plane parameters.
    Args:
        K: camera intrinsic matrix, [3, 3]
        pose: camera pose, world-to-cam, [4, 4]
        dis: pixel plane distance to camera, scalar
        x_range_uv: pixel x range, [2]
        y_range_uv: pixel y range, [2]
        plane_normal: plane normal vector, [3], optional
    Returns:
        plane: plane parameters, [4]
        plane_up_vec: plane up vector, [3]
        resol: pixel plane resolution, [2]
        new_x_range_uv: new pixel x range, [2]
        new_y_range_uv: new pixel y range, [2]
    """

    # create a pixel grid in the image plane
    u = np.linspace(x_range_uv[0], x_range_uv[1], int(x_range_uv[1]-x_range_uv[0]+1))
    v = np.linspace(y_range_uv[0], y_range_uv[1], int(y_range_uv[1]-y_range_uv[0]+1))
    U, V = np.meshgrid(u, v)

    # back project the pixel grid to 3D space
    X = depth * (U + 0.5 - K[0, 2]) / K[0, 0]
    Y = -depth * (V + 0.5 - K[1, 2]) / K[1, 1]
    Z = -depth * np.ones(U.shape)

    # transform the points from camera coordinate to world coordinate
    points = np.stack((X, Y, Z, np.ones(U.shape)), axis=-1)
    points_world = np.matmul(points, np.linalg.inv(pose).T)
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]

    # use PCA to fit a plane to these points
    points_world_flat = points_world[mask]
    mean = np.mean(points_world_flat, axis=0)
    points_world_zero_centroid = points_world_flat - mean

    if plane_normal is not None:
        plane_normal = np.matmul(plane_normal, np.linalg.inv(pose[:3, :3]).T)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

    else:
        if len(points_world_zero_centroid) < 10000:
            _, _, v = np.linalg.svd(points_world_zero_centroid)
        else:
            import random
            _, _, v = np.linalg.svd(random.sample(list(points_world_zero_centroid), 10000))

        # plane parameters
        plane_normal = v[-1, :] / np.linalg.norm(v[-1, :])
        if np.abs(plane_normal).max() != plane_normal.max():
            plane_normal = -plane_normal

    plane = np.concatenate((plane_normal, mean))

    resol = np.array([K[0,0]/dis, K[1,1]/dis])

    # calculate plane_up_vector according to the plane_normal
    pose_up_vec = np.array([0, 0, 1])
    if np.abs(np.sum(plane_normal * pose_up_vec)) > 0.8:
        pose_up_vec = np.array([0, 1, 0])
    plane_up_vec = pose_up_vec - np.sum(plane_normal * pose_up_vec) * plane_normal
    plane_up_vec = plane_up_vec / np.linalg.norm(plane_up_vec)

    U, V = U[mask], V[mask]

    plane_right = np.cross(plane_normal, plane_up_vec)

    new_x_argmin =np.matmul(points_world_flat, plane_right).argmin()
    new_x_argmax =np.matmul(points_world_flat, plane_right).argmax()
    new_y_argmin =np.matmul(points_world_flat, plane_up_vec).argmin()
    new_y_argmax =np.matmul(points_world_flat, plane_up_vec).argmax()

    new_x_width = np.sqrt((U[new_x_argmin] - U[new_x_argmax])**2 + (V[new_x_argmin] - V[new_x_argmax])**2)
    new_y_width = np.sqrt((U[new_y_argmin] - U[new_y_argmax])**2 + (V[new_y_argmin] - V[new_y_argmax])**2)

    new_x_range_uv = [-new_x_width/2 - 2, new_x_width/2 + 2]
    new_y_range_uv = [-new_y_width/2 - 2, new_y_width/2 + 2]

    y_range_3d = np.matmul(points_world_flat, plane_up_vec).max() - np.matmul(points_world_flat, plane_up_vec).min()
    x_range_3d = np.matmul(points_world_flat, plane_right).max() - np.matmul(points_world_flat, plane_right).min()

    resol = np.array([
        new_x_width / (x_range_3d + 1e-6),
        new_y_width / (y_range_3d + 1e-6)
    ])

    if resol[0] < K[0,0]/dis / 10 or resol[1] < K[1,1]/dis / 10:
        return None

    xy_min = np.array([new_x_range_uv[0], new_y_range_uv[0]])
    xy_max = np.array([new_x_range_uv[1], new_y_range_uv[1]])

    return plane, plane_up_vec, resol, xy_min, xy_max


def ray_plane_intersect(plane, ray_origin, ray_direction):
    """Calculate the intersection of a ray and a plane.
    Args:
        plane: plane parameters, [4]
        ray_origin: ray origin, [3]
        ray_direction: ray direction, [3]
    
    Returns:
        intersection: intersection point, [3]
    """

    # calculate intersection
    t = -(plane[3] + np.dot(plane[:3], ray_origin)) / np.dot(plane[:3], ray_direction)
    intersection = ray_origin + t * ray_direction

    return intersection


def points_xyz_to_plane_uv(points, plane, resol, plane_up):
    """Project 3D points to the pixel plane.
    Args:
        points: 3D points, [N, 3]
        plane: plane parameters, [4]
        resol: pixel plane resolution, [2]
        plane_up: plane up vector, [3]
    Returns:
        uv: pixel plane coordinates, [N, 2]
    """

    # plane normal vector
    plane_normal = np.asarray(plane[:3])
    # projection points of 'points' on the plane
    points_proj = points - np.outer( np.sum(points*plane_normal, axis=1)+plane[3] , plane_normal) / np.linalg.norm(plane_normal)
    mean_proj = np.mean(points_proj, axis=0)
    uvw_right = np.cross(plane_normal, plane_up)
    uvw_up = plane_up
    # calculate the uv coordinates
    uv = np.c_[np.sum((points_proj-mean_proj) * uvw_right, axis=-1), np.sum((points_proj-mean_proj) * uvw_up, axis=-1)]
    uv = uv * resol
    return uv


def points_xyz_to_plane_uv_torch(points, plane, resol, plane_up):
    """Project 3D points to the pixel plane.
    Args:
        points: 3D points, [N, 3]
        plane: plane parameters, [4]
        resol: pixel plane resolution, [2]
        plane_up: plane up vector, [3]
    Returns:
        uv: pixel plane coordinates, [N, 2]
    """

    # plane normal vector
    plane_normal = plane[:3]
    # projection points of 'points' on the plane
    points_proj = points - torch.outer( torch.sum(points*plane_normal, dim=1)+plane[3] , plane_normal) / torch.norm(plane_normal)
    mean_proj = torch.mean(points_proj, dim=0)
    uvw_right = torch.cross(plane_normal, plane_up)
    uvw_up = plane_up
    # calculate the uv coordinates
    uv = torch.cat([torch.sum((points_proj-mean_proj) * uvw_right, dim=-1, keepdim=True), torch.sum((points_proj-mean_proj) * uvw_up, dim=-1, keepdim=True)], dim=-1)
    uv = uv * resol
    return uv


def simple_distribute_planes_2D(uv_ranges, gap=2, min_H=8192):
    """Distribute pixel planes in 2D space.
    Args:
        uv_ranges: pixel ranges of each plane, [N, 2]
    Returns:
        plane_leftup: left-up pixel of each plane, [N, 2]
    """

    # calculate the left-up pixel of each plane
    plane_leftup = torch.zeros_like(uv_ranges)
    
    H = (int(torch.max(uv_ranges[:, 1]).item()) + 2) // gap * gap + gap
    H = max(H, min_H)

    # sort the planes by the height
    _, sort_idx = torch.sort(uv_ranges[:, 1], descending=True)

    # distribute the planes
    idx = 0
    now_H = 0
    now_W = 0
    prev_W = 0
    while idx < len(sort_idx):
        now_leftup = torch.tensor((prev_W+1, now_H+1))
        plane_leftup[sort_idx[idx]] = now_leftup
        now_W = max(now_W, uv_ranges[sort_idx[idx], 0] + 1 + prev_W)
        now_H += uv_ranges[sort_idx[idx], 1] + 1

        if idx + 1 < len(sort_idx) and now_H + uv_ranges[sort_idx[idx+1], 1] + 1 > H:
            prev_W = now_W
            now_H = 0

        idx += 1

    W = (int(now_W.item()) + 2) // gap * gap + gap

    return plane_leftup, W, H


def cluster_planes(planes, K=2, thres=0.999,
                   color_thres_1=0.3, color_thres_2=0.2,
                   dis_thres_1=0.5, dis_thres_2=0.1,
                   merge_edge_planes=False,
                   init_plane_sets=None):
    # planes[:, :3]: plane normal vector
    # planes[:, 3:6]: plane center

    # create disjoint set
    ds = DisjointSet(len(planes), planes[:, :6], planes[:, 17:20])

    if init_plane_sets is not None:
        for plane_set in init_plane_sets:
            for i in range(len(plane_set)-1):
                ds.union(plane_set[i], plane_set[i+1], normal_thres=0.99, dis_thres=1, color_thres=114514)
        
    # construct kd-tree for plane center
    print('clustering tablets ...')
    tree = KDTree(planes[:, 3:6])

    # find the nearest K neighbors for each plane
    _, ind = tree.query(planes[:, 3:6], k=K+1)
    
    # calculate the angle between each plane and its neighbors
    neighbor_normals = planes[ind[:, 1:], :3]
    plane_normals = planes[:, :3]
    cos = np.sum(neighbor_normals * plane_normals[:, None, :], axis=-1)

    # merge planes that have cos > thres
    for i in trange(len(planes)):
        if not merge_edge_planes and planes[i, 15] == True:
            continue

        for j in range(K):
            if not merge_edge_planes and planes[ind[i, j+1], 15] == True:
                continue
            if cos[i, j] > thres:
                ds.union(i, ind[i, j+1], normal_thres=0.99, dis_thres=dis_thres_1, color_thres=color_thres_1)

    # merge planes that have cos > thres
    for i in trange(len(planes)):
        if not merge_edge_planes and planes[i, 15] == True:
            continue

        for j in range(K):
            if not merge_edge_planes and planes[ind[i, j+1], 15] == True:
                continue
            if cos[i, j] > thres:
                ds.union(i, ind[i, j+1], normal_thres=0.9, dis_thres=dis_thres_2, color_thres=color_thres_2)

    root2idx = {}
    for i in range(len(planes)):
        root = ds.find(i)
        if root not in root2idx:
            root2idx[root] = []
        root2idx[root].append(i)

    plane_sets = []
    for root in root2idx:
        plane_sets.append(root2idx[root])

    return plane_sets

