import os
import glob

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, min_depth=0, max_depth=-1, time=None) -> None:
        self.data_path = data_path
        self.img_paths = sorted(glob.glob(os.path.join(data_path, 'img/*.png')))
        self.img_num = len(self.img_paths)
        self.depth_paths = sorted(glob.glob(os.path.join(data_path, 'depth/*.png')))
        self.max_depth = max_depth
        self.use_gt = use_gt
        self.K = self.load_intrinsic(time)
        self.gt_pose = self.load_gt_pose() if self.use_gt else None

    def load_intrinsic(self, time):
        if time == '2011_10_03':
            K = np.eye(3)
            K[0, 0] = 718.8560
            K[1, 1] = 718.8560
            K[0, 2] = 607.1928
            K[1, 2] = 185.2157
        elif time == '2011_09_30':
            K = np.eye(3)
            K[0, 0] = 707.0912
            K[1, 1] = 707.0912
            K[0, 2] = 601.8873
            K[1, 2] = 183.1104

        return K

    def get_init_pose(self):
        if self.gt_pose is not None:
            return self.gt_pose[0]
        else:
            pose = np.eye(4)
            return pose

    def load_gt_pose(self):
        gt_file = os.path.join(self.data_path, 'pose.txt')
        gt_pose = np.loadtxt(gt_file)
        pose = np.concatenate((gt_pose, np.tile(np.array([0, 0, 0, 1]), (gt_pose.shape[0], 1))), axis=1).reshape(-1, 4, 4)
        return pose

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = depth / 10.0
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        return depth

    def load_image(self, path):
        rgb = cv2.imread(path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb / 255.0

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        img = torch.from_numpy(self.load_image(self.img_paths[index])).float()
        depth = self.load_depth(self.depth_paths[index])
        depth = None if depth is None else torch.from_numpy(depth).float()
        pose = self.gt_pose[index] if self.use_gt else None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    loader = DataLoader('../../data/apollo/Record001', use_gt=True)
    for data in loader:
        index, img, depth, K, pose = data
        print(pose)
        # cv2.imshow('img', img.numpy())
        # cv2.imshow('depth', depth.numpy())
        # cv2.waitKey(0)

