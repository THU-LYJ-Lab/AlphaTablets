from .load_colmap import load_colmap_data
from .load_replica import load_replica_data


def load_data(args, kf_list, image_skip, load_plane):
    if args.dataset_type == 'scannet':
        images, intr, sfm_poses, sfm_camprojs, cam_centers, \
              all_depths, normals, planes_all, mvp_mtxs, index_init, \
              new_sps, mean_colors = load_colmap_data(args, kf_list=kf_list,
                                                      image_skip=image_skip,
                                                      load_plane=load_plane)
        print('Loaded scannet', intr, len(images), sfm_poses.shape, sfm_camprojs.shape, args.input_dir)

        data_dict = dict(
            poses=sfm_poses, images=images,
            intr=intr, sfm_camprojs=sfm_camprojs,
            cam_centers=cam_centers,
            all_depths=all_depths,
            normals=normals, planes_all=planes_all,
            mvp_mtxs=mvp_mtxs,
            index_init=index_init,
            new_sps=new_sps,
            mean_colors=mean_colors,
        )
        return data_dict


    elif args.dataset_type == 'replica':
        images, intr, sfm_poses, sfm_camprojs, cam_centers, \
              all_depths, normals, planes_all, mvp_mtxs, index_init, \
              new_sps, mean_colors = load_replica_data(args, kf_list=kf_list,
                                                       image_skip=image_skip,
                                                       load_plane=load_plane)

        print('Loaded replica', intr, len(images), sfm_poses.shape, sfm_camprojs.shape, args.input_dir)

        data_dict = dict(
            poses=sfm_poses, images=images,
            intr=intr, sfm_camprojs=sfm_camprojs,
            cam_centers=cam_centers,
            all_depths=all_depths,
            normals=normals, planes_all=planes_all,
            mvp_mtxs=mvp_mtxs,
            index_init=index_init,
            new_sps=new_sps,
            mean_colors=mean_colors,
        )
        return data_dict


    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

