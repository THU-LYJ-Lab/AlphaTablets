import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, mvp_mtxs, normals, depths):
        self.images = images
        self.mvp_mtxs = mvp_mtxs
        self.normals = normals
        self.depths = depths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        mvp_mtx = torch.from_numpy(self.mvp_mtxs[idx]).float()
        normal = torch.from_numpy(self.normals[idx])
        depth = torch.from_numpy(self.depths[idx])

        return img, mvp_mtx, normal, depth
