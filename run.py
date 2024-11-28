from datetime import datetime
import os, time, random, argparse

import numpy as np
import cv2

from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from lib.load_data import load_data
from lib.alphatablets import AlphaTablets
from dataset.dataset import CustomDataset
from lib.keyframe import get_keyframes
from recon.run_recon import Recon


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--job", type=str, default=str(time.time()),
                        help='Job name')
    
    # learning options
    parser.add_argument("--lr_tex", type=float, default=0.01,
                        help='Learning rate of texture color')
    parser.add_argument("--lr_alpha", type=float, default=0.03,
                        help='Learning rate of texture alpha')
    parser.add_argument("--lr_plane_n", type=float, default=0.0001,
                        help='Learning rate of plane normal')
    parser.add_argument("--lr_plane_dis", type=float, default=0.0005,
                        help='Learning rate of plane distance')
    parser.add_argument("--lr_plane_dis_stage2", type=float, default=0.0002,
                        help='Learning rate of plane distance in stage 2')
    
    # loss weights
    parser.add_argument("--weight_alpha_inv", type=float, default=1.0,
                        help='Weight of alpha inv loss')
    parser.add_argument("--weight_normal", type=float, default=4.0,
                        help='Weight of direct normal loss')
    parser.add_argument("--weight_depth", type=float, default=4.0,
                        help='Weight of direct depth loss')
    parser.add_argument("--weight_distortion", type=float, default=20.0,
                        help='Weight of distortion loss')
    parser.add_argument("--weight_decay", type=float, default=0.9,
                        help='Weight of tablet alpha decay after a single step. -1 denotes automatic decay')
    
    # merging options
    parser.add_argument("--merge_normal_thres_init", type=float, default=0.97,
                        help='Threshold of init merging planes')
    parser.add_argument("--merge_normal_thres", type=float, default=0.93,
                        help='Threshold of merging planes')
    parser.add_argument("--merge_dist_thres1", type=float, default=0.5,
                        help='Threshold of init merging planes')
    parser.add_argument("--merge_dist_thres2", type=float, default=0.1,
                        help='Threshold of merging planes')
    parser.add_argument("--merge_color_thres1", type=float, default=0.3,
                        help='Threshold of init merging planes')
    parser.add_argument("--merge_color_thres2", type=float, default=0.2,
                        help='Threshold of merging planes')
    parser.add_argument("--merge_Y_decay", type=float, default=0.5,
                        help='Decay rate of Y channel')
    
    # optimization options
    parser.add_argument("--batch_size", type=int, default=3,
                        help='Batch size')
    parser.add_argument("--max_steps", type=int, default=32,
                        help='Max optimization steps')
    parser.add_argument("--merge_interval", type=int, default=13,
                        help='Merge interval')
    parser.add_argument("--max_steps_union", type=int, default=9,
                        help='Max optimization steps for union optimization')
    parser.add_argument("--merge_interval_union", type=int, default=3,
                        help='Merge interval for union optimization')
    
    parser.add_argument("--alpha_init", type=float, default=0.5,
                        help='Initial alpha value')
    parser.add_argument("--alpha_init_empty", type=float, default=0.,
                        help='Initial alpha value for empty pixels')
    parser.add_argument("--depth_inside_mask_alphathres", type=float, default=0.5,
                        help='Threshold of alpha for inside mask')
    
    parser.add_argument("--max_rasterize_layers", type=int, default=15,
                        help='Max rasterize layers')
    
    # logging/saving options
    parser.add_argument("--log_path", type=str, default='./logs',
                        help='path to save logs')
    parser.add_argument("--dump_images", type=bool, default=False)
    parser.add_argument("--dump_interval", type=int, default=1,
                        help='Dump interval')

    # update input
    parser.add_argument("--input_dir", type=str, default='',
                        help='input directory')
    
    args = parser.parse_args()
    
    from hydra import compose, initialize

    initialize(version_base=None, config_path='./')
    ori_cfg = compose(config_name=args.config)['configs']
    
    class Struct:
        def __init__(self, **entries): 
            self.__dict__.update(entries)

    cfg = Struct(**ori_cfg)

    # dump args and cfg
    import json
    os.makedirs(os.path.join(args.log_path, args.job), exist_ok=True)
    with open(os.path.join(args.log_path, args.job, 'args.json'), 'w') as f:
        args_json = {k: v for k, v in vars(args).items() if k != 'config'}
        json.dump(args_json, f, indent=4)

    import shutil
    shutil.copy(args.config, os.path.join(args.log_path, args.job, 'config.yaml'))

    if args.input_dir != '':
        cfg.data.input_dir = args.input_dir

    return args, cfg


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(cfg, kf_list, image_skip, load_plane=True):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data, kf_list=kf_list, image_skip=image_skip, load_plane=load_plane)
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


if __name__=='__main__':
    # load setup
    args, cfg = config_parser()
    merge_cfgs = {
        'normal_thres_init': args.merge_normal_thres_init,
        'normal_thres': args.merge_normal_thres,
        'dist_thres1': args.merge_dist_thres1,
        'dist_thres2': args.merge_dist_thres2,
        'color_thres1': args.merge_color_thres1,
        'color_thres2': args.merge_color_thres2,
        'Y_decay': args.merge_Y_decay
    }

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    seed_everything()

    recon = Recon(cfg.data.init, cfg.data.input_dir, cfg.data.dataset_type)
    recon.recon()

    kf_lists, image_skip = get_keyframes(cfg.data.input_dir)
    subset_mean_len = np.mean([ k[-1]-k[0]+1 for k in kf_lists ])
    kf_lists.append([0, -1])

    for set_num, kf_list in enumerate(kf_lists):
        if set_num == len(kf_lists) - 1:
            print(str(datetime.now()) + f': \033[94mUnion Optimization', '\033[0m')
        else:
            print(str(datetime.now()) + f': \033[94mSubset {set_num+1}/{len(kf_lists)-1}, Keyframes', kf_list, '\033[0m')

        union_optimize = False
        if kf_list[0] == 0 and kf_list[1] == -1:
            union_optimize = True
            if os.path.exists(os.path.join(args.log_path, args.job, 'ckpt', f'./ckpt_{set_num:02d}_{args.max_steps_union-1}.pt')):
                print(str(datetime.now()) + f': \033[94mUnion optimization already done!', '\033[0m')
                continue
        else:
            if os.path.exists(os.path.join(args.log_path, args.job, 'ckpt', f'./ckpt_{set_num:02d}_{args.max_steps-1}.pt')):
                print(str(datetime.now()) + f': \033[94mSubset {set_num+1}/{len(kf_lists)-1} optimization already done!', '\033[0m')
                continue

        if kf_list[-1] != -1:
            recon.run_sp(kf_list)

        # load images / poses / camera settings / data split
        data_dict = load_everything(cfg=cfg, kf_list=kf_list, image_skip=image_skip,
                                    load_plane=not union_optimize)

        images, mvp_mtxs, K = data_dict['images'], data_dict['mvp_mtxs'], data_dict['intr']
        index_init = data_dict['index_init']
        new_sps = data_dict['new_sps']
        cam_centers = data_dict['cam_centers']
        normals = data_dict['normals']
        mean_colors = data_dict['mean_colors']
        depths = data_dict['all_depths']

        sp_images = np.stack([ images[idx] for idx in index_init ])

        if not union_optimize:
            pp = AlphaTablets(plane_params=data_dict['planes_all'], 
                            sp=[ sp_images , new_sps, mvp_mtxs[index_init]], 
                            cam_centers=cam_centers[index_init],
                            mean_colors=mean_colors,
                            HW=images[0].shape[0:2],
                            alpha_init=args.alpha_init, alpha_init_empty=args.alpha_init_empty,
                            inside_mask_alpha_thres=args.depth_inside_mask_alphathres,
                            merge_cfgs=merge_cfgs)
        else:
            pp = AlphaTablets(ckpt_paths=[
                os.path.join(args.log_path, args.job, 'ckpt', f'./ckpt_{ii:02d}_{args.max_steps-1}.pt')
                for ii in range(len(kf_lists) - 1)
            ], merge_cfgs=merge_cfgs,
            alpha_init=args.alpha_init, alpha_init_empty=args.alpha_init_empty,
            inside_mask_alpha_thres=args.depth_inside_mask_alphathres)

        dataset = CustomDataset(images, mvp_mtxs, normals, depths)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device))

        optimizer = torch.optim.Adam([
            {'params': pp.tex_color, 'lr': args.lr_tex},
            {'params': pp.tex_alpha, 'lr': args.lr_alpha},
            {'params': pp.plane_n, 'lr': args.lr_plane_n},
            {'params': pp.plane_dis, 'lr': args.lr_plane_dis},
        ])

        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)
        crop = cfg.data.crop

        max_steps = args.max_steps if not union_optimize else args.max_steps_union
        merge_interval = args.merge_interval if not union_optimize else args.merge_interval_union

        psnr_lst = []
        time0 = time.time()
        print(str(datetime.now()) + ': \033[92mI', 'Start AlphaTablets Optimization', '\033[0m')
        for global_step in trange(0, max_steps):
            avg_psnr = 0
            for batch_idx, (imgs, mvp_mtxs, normals, depths) in enumerate(dataloader):
                mvp_mtxs = mvp_mtxs.cuda()
                imgs = imgs.cuda()
                normals = normals.cuda()
                depths = depths.cuda()

                render_result = pp(mvp_mtxs,
                                   max_rasterize_layers=args.max_rasterize_layers)

                try:
                    render_result, render_normal, render_depth, alpha_loss, distort_loss, inside_mask, plane_inside_mask = render_result
                except:
                    continue

                # gradient descent step
                optimizer.zero_grad(set_to_none=True)
                crop_mask = torch.zeros_like(imgs)
                crop_mask[:, crop:-crop, crop:-crop] = 1
                mask = crop_mask * inside_mask
                loss = F.mse_loss(render_result * mask, imgs * mask)
                direct_normal_loss = F.mse_loss(render_normal, normals)
                direct_depth_loss = F.mse_loss(render_depth * mask[..., 0] * plane_inside_mask[..., 0], depths * mask[..., 0] * plane_inside_mask[..., 0])
                psnr = -10 * torch.log10(loss)

                loss += alpha_loss * args.weight_alpha_inv
                loss += distort_loss * args.weight_distortion
                loss += direct_normal_loss * args.weight_normal
                loss += direct_depth_loss * args.weight_depth
                avg_psnr += psnr.item()

                loss.backward()
                optimizer.step()

                if global_step % args.dump_interval == 0 and batch_idx == 0 and args.dump_images:
                    optimizer.zero_grad(set_to_none=True)
                    mvp_mtxs = torch.from_numpy(data_dict['mvp_mtxs'][0:1]).cuda().float()
                    render_result, alpha_acc = pp(mvp_mtxs, return_alpha=True,
                                                  max_rasterize_layers=args.max_rasterize_layers)

                    os.makedirs(os.path.join(args.log_path, args.job, 'dump_images'), exist_ok=True)

                    cv2.imwrite(os.path.join(args.log_path, args.job, 'dump_images', f'{set_num:02d}_{global_step:05d}.png'), render_result[0].detach().cpu().numpy() * 255)
                    cv2.imwrite(os.path.join(args.log_path, args.job, 'dump_images', f'{set_num:02d}_{global_step:05d}_mask.png'), mask[0].detach().cpu().numpy() * 255)
                    cv2.imwrite(os.path.join(args.log_path, args.job, 'dump_images', f'{set_num:02d}_{global_step:05d}_alpha.png'), alpha_acc[0].detach().cpu().numpy() * 255)

                    error_map = torch.abs(render_result - torch.from_numpy(data_dict['images'][0][None, ...]).cuda().float())
                    error_map = error_map[0].detach().cpu().numpy()
                    # turn to red-blue, 0-255
                    error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min()) * 255
                    error_map = cv2.applyColorMap(error_map.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(args.log_path, args.job, 'dump_images', f'{set_num:02d}_{global_step:05d}_error_map.png'), error_map)

                if global_step % merge_interval == 0 and batch_idx == 0 and global_step > 0:
                    optimizer.zero_grad(set_to_none=True)

                    pp.weightCheck(torch.from_numpy(data_dict['mvp_mtxs']).cuda().float())

                    pp = AlphaTablets(pp=pp, merge_cfgs=merge_cfgs)
                    del optimizer
                    optimizer = torch.optim.Adam([
                        {'params': pp.tex_color, 'lr': args.lr_tex},
                        {'params': pp.tex_alpha, 'lr': args.lr_alpha},
                        {'params': pp.plane_n, 'lr': args.lr_plane_n},
                        {'params': pp.plane_dis, 'lr': args.lr_plane_dis_stage2},
                    ])

            scheduler.step()

            if global_step == max_steps - 1:
                with torch.no_grad():
                    os.makedirs(os.path.join(args.log_path, args.job, 'ckpt'), exist_ok=True)
                    pp.save_ckpt(os.path.join(args.log_path, args.job, 'ckpt', f'ckpt_{set_num:02d}_{global_step}.pt'))
                optimizer.zero_grad(set_to_none=True)

            if union_optimize:
                with torch.no_grad():
                    decay = args.weight_decay if args.weight_decay > 0 else max(1-0.00067*subset_mean_len, 0.8)
                    pp.tex_alpha.data = torch.log( decay / (1 - decay + torch.exp(-pp.tex_alpha)) )
            elif global_step < args.merge_interval:
                with torch.no_grad():
                    decay = args.weight_decay if args.weight_decay > 0 else max(1-0.00067*len(images), 0.8)
                    pp.tex_alpha.data = torch.log( decay / (1 - decay + torch.exp(-pp.tex_alpha)) )

        if union_optimize:
            os.makedirs(os.path.join(args.log_path, args.job, 'results'), exist_ok=True)
            export_name_obj = os.path.join(args.log_path, args.job, 'results', f'{set_num:02d}_final.obj')
            pp.export_mesh_with_weight_check(torch.from_numpy(data_dict['mvp_mtxs']).cuda().float(), name=export_name_obj)

        os.makedirs(os.path.join(args.log_path, args.job, 'plys'), exist_ok=True)
        export_name = os.path.join(args.log_path, args.job, 'plys', f'{set_num:02d}_final.ply')
        pp.export_ply(torch.from_numpy(data_dict['mvp_mtxs'][::8]).cuda().float(), name=export_name)

        if set_num == len(kf_lists) - 1:
            print(str(datetime.now()) + f': \033[94mUnion optimization finished!', '\033[0m')
        else:
            print(str(datetime.now()) + f': \033[94mSubset {set_num+1}/{len(kf_lists)-1} optimization finished', '\033[0m')
    
    # create complete lock
    with open(os.path.join(args.log_path, args.job, 'complete.lock'), 'w') as f:
        f.write('done')
