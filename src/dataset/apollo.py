import os
import glob

import cv2
import numpy as np
import torch
import csv
from glob import glob
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, min_depth=0, max_depth=-1, left=True) -> None:
        self.data_path = data_path
        self.img_paths = sorted(glob(os.path.join(self.data_path, 'img/*.jpg')))
        self.img_num = len(self.img_paths)
        self.depth_paths = sorted(glob(os.path.join(self.data_path, 'depth/*.png')))
        self.max_depth = max_depth
        self.use_gt = use_gt
        self.K = self.load_intrinsic(left)
        self.gt_pose = self.load_gt_pose() if self.use_gt else None

    def load_intrinsic(self, left):
        if left:
            K = np.eye(3)
            K[0, 0] = 2304.54786556982
            K[1, 1] = 2305.875668062
            K[0, 2] = 1686.23787612802
            K[1, 2] = 1354.98486439791
        else:
            K = np.eye(3)
            K[0, 0] = 2300.39065314361
            K[1, 1] = 2301.31478860597
            K[0, 2] = 1713.21615190657
            K[1, 2] = 1342.91100799715

        return K

    def get_init_pose(self):
        if self.gt_pose is not None:
            return self.gt_pose[0]
        else:
            pose = np.eye(4)
            return pose

    def load_gt_pose(self):
        gt_file = os.path.join(self.data_path, 'pose.txt')
        gt_pose = []
        with open(gt_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split()
                pose = np.asarray([float(val) for val in line[:16]]).reshape(4, 4)
                stamp = line[16]
                gt_pose.append({"pose": pose,
                                "stamp": stamp})
        index = sorted(range(len(gt_pose)), key=lambda x: gt_pose[x]["stamp"])
        pose = [gt_pose[i]["pose"] for i in index]
        return pose

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = depth / 6553.5
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
