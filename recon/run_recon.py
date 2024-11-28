import os
import glob
from datetime import datetime


class Recon:
    def __init__(self, config, input_dir, dataset_type):
        self.config = config
        self.input_dir = input_dir
        self.image_dir = os.path.join(input_dir, "images")
        self.output_root = input_dir
        self.dataset_type = dataset_type
        self.preprocess()

    def preprocess(self):
        if not os.path.exists(self.image_dir):
            if self.dataset_type == 'scannet':
                original_image_dir = os.path.join(self.input_dir, "color")
                if not os.path.exists(original_image_dir):
                    raise ValueError(f'Image directory {self.image_dir} not found')
                old_dir = os.getcwd()
                os.chdir(self.input_dir)
                os.symlink("color", "images")
                os.chdir(old_dir)
                print(str(datetime.now()) + ': \033[92mI', f'Linked {original_image_dir} to {self.image_dir}', '\033[0m')
            elif self.dataset_type == 'replica':
                original_image_dir = os.path.join(self.input_dir, "results")
                if not os.path.exists(original_image_dir):
                    raise ValueError(f'Image directory {self.image_dir} not found')
                os.makedirs(self.image_dir)
                old_dir = os.getcwd()
                os.chdir(self.image_dir)
                for img in glob.glob(os.path.join("../results", "frame*")):
                    os.symlink(os.path.join("../results", os.path.split(img)[-1]), os.path.split(img)[-1])
                os.chdir(old_dir)
                print(str(datetime.now()) + ': \033[92mI', f'Linked {original_image_dir} to {self.image_dir}', '\033[0m')
            else:
                raise NotImplementedError(f'Unknown dataset type {self.dataset_type} exiting')
            

    def recon(self):
        if os.path.exists(os.path.join(self.output_root, "recon.lock")):
            print(str(datetime.now()) + ': \033[92mI', 'Monocular estimation already done!', '\033[0m')
            return
        
        from .run_depth import omnidata_normal, metric3d_depth
        
        depth_dir = os.path.join(self.output_root, "aligned_dense_depths")
        print(str(datetime.now()) + ': \033[92mI', 'Running Depth Estimation ...', '\033[0m')
        metric3d_depth(self.image_dir, self.output_root, depth_dir, self.dataset_type)
        print(str(datetime.now()) + ': \033[92mI', 'Depth Estimation Done!', '\033[0m')

        normal_dir = os.path.join(self.output_root, "omnidata_normal")
        print(str(datetime.now()) + ': \033[92mI', 'Running Normal Estimation ...', '\033[0m')
        omnidata_normal(self.image_dir, self.output_root, normal_dir)
        print(str(datetime.now()) + ': \033[92mI', 'Normal Estimation Done!', '\033[0m')

        # create lock file
        open(os.path.join(self.output_root, "recon.lock"), "w").close()

    def run_sp(self, kf_list):
        from .run_sp import run_sp
        print(str(datetime.now()) + ': \033[92mI', 'Running SuperPixel Subdivision ...', '\033[0m')
        sp_dir = run_sp(self.image_dir, self.output_root, kf_list)
        print(str(datetime.now()) + ': \033[92mI', 'SuperPixel Subdivision Done!', '\033[0m')
        return sp_dir
