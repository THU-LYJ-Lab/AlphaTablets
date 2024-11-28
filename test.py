import os
import pickle
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

def compute_sc(gt_in, pred_in):
    # to be consistent with skimage sklearn input arrangment

    assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

    acc, pred, gt  = match_seg(pred_in, gt_in) # n_gt * n_pred

    bestmatch_gt2pred = acc.max(axis=1)
    bestmatch_pred2gt = acc.max(axis=0)

    pred_id, pred_cnt = np.unique(pred, return_counts=True)
    gt_id, gt_cnt = np.unique(gt, return_counts=True)

    cnt_pred, sum_pred = 0, 0
    for i, _ in enumerate(pred_id):
        cnt_pred += bestmatch_pred2gt[i] * pred_cnt[i]
        sum_pred += pred_cnt[i]

    cnt_gt, sum_gt = 0, 0
    for i, _ in enumerate(gt_id):
        cnt_gt += bestmatch_gt2pred[i] * gt_cnt[i]
        sum_gt += gt_cnt[i]

    sc = (cnt_pred / sum_pred + cnt_gt / sum_gt) / 2

    return sc

def match_seg(pred_in, gt_in):
    assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

    pred, gt = compact_segm(pred_in), compact_segm(gt_in)
    n_gt = gt.max() + 1
    n_pred = pred.max() + 1

    # this will offer the overlap between gt and pred
    # if gt == 1, we will later have conf[1, j] = gt(1) + pred(j) * n_gt
    # essential, we encode conf_mat[i, j] to overlap, and when we decode it we let row as gt, and col for pred
    # then assume we have 13 gt label, 6 pred label --> gt 1 will correspond to 14, 1+2*13 ... 1 + 6*13
    overlap =  gt + n_gt * pred
    freq, bin_val = np.histogram(overlap, np.arange(0, n_gt * n_pred+1)) # hist given bins [1, 2, 3] --> return [1, 2), [2, 3)
    conf_mat = freq.reshape([ n_gt, n_pred], order='F') # column first reshape, like matlab

    acc = np.zeros([n_gt, n_pred])
    for i in range(n_gt):
        for j in range(n_pred):
            gt_i = conf_mat[i].sum()
            pred_j = conf_mat[:, j].sum()
            gt_pred = conf_mat[i, j]
            acc[i,j] = gt_pred / (gt_i + pred_j - gt_pred) if (gt_i + pred_j - gt_pred) != 0 else 0
    return acc[1:, 1:], pred, gt

def compact_segm(seg_in):
    seg = seg_in.copy()
    uniq_id = np.unique(seg)
    cnt = 1
    for id in sorted(uniq_id):
        if id == 0:
            continue
        seg[seg==id] = cnt
        cnt += 1

    # every id (include non-plane should not be 0 for the later process in match_seg
    seg = seg + 1
    return seg

def project_to_mesh(from_mesh, to_mesh, attribute, attr_name, color_mesh=None, dist_thresh=None):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attr_name] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = attribute.copy()
    pred_colors = from_mesh.visual.vertex_colors  if color_mesh is None else color_mesh.visual.vertex_colors

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    mesh.vertex_attributes[attr_name] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02, error_map=True):
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    if error_map:
        # repeat but without downsampling
        mesh_pred = o3d.io.read_triangle_mesh(file_pred)
        mesh_trgt = o3d.io.read_triangle_mesh(file_trgt)
        verts_pred = np.asarray(mesh_pred.vertices)
        verts_trgt = np.asarray(mesh_trgt.vertices)
        _, dist1 = nn_correspondance(verts_pred, verts_trgt)
        _, dist2 = nn_correspondance(verts_trgt, verts_pred)
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        # recall_err_viz
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        dist1_n = dist1 / 0.3
        color = cmap(dist1_n)
        mesh_trgt.vertex_colors = o3d.utility.Vector3dVector(color[:, :3])

        # precision_err_viz
        dist2_n = dist2 / 0.4
        color = cmap(dist2_n)
        mesh_pred.vertex_colors = o3d.utility.Vector3dVector(color[:, :3])
    else:
        mesh_pred = mesh_trgt = None
    return metrics, mesh_pred, mesh_trgt


def process(scene, save_path='results'):
    folder = f'./logs/{scene}/results'
    num = np.max([ int(i.split('_')[0]) for i in os.listdir(folder) if '_final' in i ])
    mesh_file_eval_ori = os.path.join(folder, f'{num:02d}_final.obj')
    gt_folder = f'./planes_9/{scene}'
    file_mesh_trgt = os.path.join(gt_folder, 'annotation/planes_mesh.ply')

    with open(os.path.join(folder, f'{num:02d}_final_instance_id.txt')) as f:
        ori_instance_ids = [float(x.strip()) for x in f.readlines()]

    with open(os.path.join(gt_folder, 'fragments.pkl'), 'rb') as f:
        fragments = pickle.load(f)

    dist1 = []
    dist2 = []
    prec = []
    recall = []
    fscore = []
    ris = []
    vois = []
    scs = []

    mesh = trimesh.load(mesh_file_eval_ori)
    faces = mesh.faces.copy()
    instance_ids = np.array(ori_instance_ids)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)

    sample_points, sample_indices = trimesh.sample.sample_surface_even(mesh, mesh.vertices.shape[0] * 10)
    points_mask = np.ones((len(sample_points)), dtype=bool)
    for frag in fragments:
        vol_origin = frag['vol_origin']
        vol_size = frag['voxel_size']
        vol_dim = frag['vol_dim']
        vol_end = vol_origin + vol_size * vol_dim
        xmin, xmax = vol_origin[0], vol_end[0]
        ymin, ymax = vol_origin[1], vol_end[1]
        zmin, zmax = vol_origin[2], vol_end[2]
        vertices_mask_new = np.logical_and(
            np.logical_and(sample_points[:, 0] >= xmin, sample_points[:, 0] <= xmax),
            np.logical_and(sample_points[:, 1] >= ymin, sample_points[:, 1] <= ymax),
            np.logical_and(sample_points[:, 2] >= zmin, sample_points[:, 2] <= zmax),
        )
        points_mask = np.logical_and(points_mask, vertices_mask_new)

    sample_points = sample_points[points_mask]
    sample_indices = sample_indices[points_mask]

    vertices_eval = trimesh.Trimesh(vertices=sample_points, process=False)
    vertices_eval.export(mesh_file_eval_ori.replace('.obj', '.ply'))
    mesh_file_eval = mesh_file_eval_ori.replace('.obj', '.ply')

    # eval 3d geometry
    metrics_mesh, prec_err_pcd, recal_err_pcd = eval_mesh(mesh_file_eval, file_mesh_trgt, error_map=False)
    metrics = {**metrics_mesh}
    # o3d.io.write_triangle_mesh(os.path.join('./','%s_precErr.ply' % scene), prec_err_pcd)
    # o3d.io.write_triangle_mesh(os.path.join('./', '%s_recErr.ply' % scene), recal_err_pcd)

    dist1.append(metrics['dist1'])
    dist2.append(metrics['dist2'])
    prec.append(metrics['prec'])
    recall.append(metrics['recal'])
    fscore.append(metrics['fscore'])

    # prepare files for instance evaluation
    mesh_trgt = trimesh.load(file_mesh_trgt, process=False)

    new_pred_ins = np.array(instance_ids)[np.array(sample_indices).astype('int32')].astype('int32')

    # specify color to vertces_eval by color pool with new_pred_ins
    color_pool = np.random.rand(32768, 3) * 255
    color_pool = np.concatenate([color_pool, np.ones((32768, 1)) * 255], axis=1).astype(np.uint8)
    colors = color_pool[new_pred_ins]
    vertices_eval.visual.vertex_colors = colors

    mesh_planeIns_transfer = project_to_mesh(vertices_eval, mesh_trgt, new_pred_ins, 'plane_ins')

    planeIns = mesh_planeIns_transfer.vertex_attributes['plane_ins']

    plnIns_save_pth = os.path.join(save_path, 'plane_ins')
    if not os.path.isdir(plnIns_save_pth):
        os.makedirs(plnIns_save_pth)
    
    mesh_planeIns_transfer.export(os.path.join(plnIns_save_pth, '%s_planeIns_transfer.ply' % scene))
    np.savetxt(plnIns_save_pth + '/%s.txt'%scene, planeIns, fmt='%d')

    pred_pth = os.path.join(plnIns_save_pth, '{}.txt'.format(scene))
    gt_pth = os.path.join(f'./planes_9/instance/{scene}.txt')

    pred_ins = np.loadtxt(pred_pth).astype(np.int32)
    gt_ins = np.loadtxt(gt_pth).astype(np.int32)

    ri =  rand_score(gt_ins, pred_ins)
    h1, h2 = variation_of_information(gt_ins, pred_ins)
    voi = h1 + h2
    sc = compute_sc(gt_ins, pred_ins)

    ris.append(ri)
    vois.append(voi)
    scs.append(sc)

    return metrics, ri, voi, sc


if __name__ == '__main__':
    import glob
    now_scenes = sorted(glob.glob('./logs/scene????_??'))
    flag = False
    stats = []
    for scene in tqdm(now_scenes):
        scene = scene.split('/')[-1]
        metrics, ri, voi, sc = process(scene)
        stats.append([metrics['dist1'], metrics['dist2'], metrics['prec'], metrics['recal'], metrics['fscore'], ri, voi, sc])
        print('scene', scene, '\t'.join([f'{k}: {v:.4f}' for k, v in metrics.items()]))

    stats = np.array(stats)
    print(f'dist1:\t{np.mean(stats[:, 0]):.3f}')
    print(f'dist2:\t{np.mean(stats[:, 1]):.3f}')
    print(f'prec:\t{np.mean(stats[:, 2]):.3f}')
    print(f'recall:\t{np.mean(stats[:, 3]):.3f}')
    print(f'fscore:\t{np.mean(stats[:, 4]):.3f}')
    print(f'ri:\t{np.mean(stats[:, 5]):.3f}')
    print(f'voi:\t{np.mean(stats[:, 6]):.3f}')
    print(f'sc:\t{np.mean(stats[:, 7]):.3f}')
