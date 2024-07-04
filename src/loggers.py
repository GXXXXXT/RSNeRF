import os
import os.path as osp
import pickle
import shutil
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import yaml
import glob

from variations.render_helpers import Peak_Signal_to_Noise_Ratio


class BasicLogger:
    def __init__(self, args) -> None:
        self.args = args
        self.use_exp = args.use_exp
        self.log_dir = osp.join(args.log_dir, args.exp_name, args.logs_name)
        self.img_dir = osp.join(self.log_dir, "imgs")
        self.color_dir = osp.join(self.log_dir, "imgs/color")
        self.depth_dir = osp.join(self.log_dir, "imgs/depth")
        self.mesh_dir = osp.join(self.log_dir, "mesh")
        self.ckpt_dir = osp.join(self.log_dir, "ckpt")
        self.backup_dir = osp.join(self.log_dir, "bak")
        self.misc_dir = osp.join(self.log_dir, "misc")
        if args.cont is False:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.img_dir)
            os.makedirs(self.color_dir)
            os.makedirs(self.depth_dir)
            os.makedirs(self.ckpt_dir)
            os.makedirs(self.mesh_dir)
            os.makedirs(self.misc_dir)
            os.makedirs(self.backup_dir)

            self.log_config(args)

    def log_ckpt(self, mapper, share_data, stamp):
        paths = glob.glob(os.path.join(self.ckpt_dir, '*.pth'))
        for path in paths:
            os.remove(path)

        road_decoder_state = share_data.road_decoder.state_dict()
        exp_decoder_state = share_data.exp_decoder.state_dict() if self.use_exp else None
        map_states = mapper.map_states
        keyframe = {frame.stamp: {'pose': frame.pose.state_dict(),
                                  'optim': frame.optim.state_dict() if frame.optim is not None else None} for frame in mapper.keyframe_graph}
        embeddings = mapper.embeddings
        embed_optim = mapper.embed_optim.state_dict()
        frame_poses = mapper.frame_poses
        psnr = share_data.psnr
        svo = mapper.svo
        torch.save({
            "road_decoder_state": road_decoder_state,
            "exp_decoder_state": exp_decoder_state,
            "map_states": map_states,
            "keyframe": keyframe,
            "embeddings": embeddings,
            "embed_optim": embed_optim,
            "frame_poses": frame_poses,
            "psnr": psnr,
            "svo": svo},
            os.path.join(self.ckpt_dir, f"ckpt_{str(stamp).zfill(5)}.pth"))
        print(f"***** successfully saved ckpt_{str(stamp).zfill(5)}.pth *****")

    def load_ckpt(self):
        return torch.load(sorted(glob.glob(os.path.join(self.ckpt_dir, '*.pth')))[-1])

    def log_config(self, config):
        out_path = osp.join(self.backup_dir, "config.yaml")
        yaml.dump(config, open(out_path, 'w'))

    def log_mesh(self, mesh, name="final_mesh.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_triangle_mesh(out_path, mesh)

    def log_point_cloud(self, pcd, name="final_points.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_point_cloud(out_path, pcd)

    def log_numpy_data(self, data, name, ind=None):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if name == "final_voxels":
            paths = glob.glob(os.path.join(self.ckpt_dir, '*.pth'))
            for path in paths:
                os.remove(path)

        if ind is not None:
            np.save(osp.join(self.misc_dir, "{}-{:05d}.npy".format(name, ind)), data)
        else:
            np.save(osp.join(self.misc_dir, f"{name}.npy"), data)

    def log_debug_data(self, data, idx):
        with open(os.path.join(self.misc_dir, f"scene_data_{idx}.pkl"), 'wb') as f:
            pickle.dump(data, f)

    def log_raw_image(self, ind, rgb, depth):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        rgb = cv2.cvtColor(rgb*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.jpg".format(
            ind)), (rgb).astype(np.uint8))
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.png".format(
            ind)), (depth*5000).astype(np.uint16))

    def log_images(self, ind, gt_rgb, gt_depth, rgb, depth):
        gt_depth_np = gt_depth.detach().cpu().numpy()
        gt_color_np = gt_rgb.detach().cpu().numpy()
        depth_np = depth.squeeze().detach().cpu().numpy()
        color_np = rgb.detach().cpu().numpy()

        h, w = depth_np.shape
        gt_depth_np = cv2.resize(
            gt_depth_np, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_color_np = cv2.resize(
            gt_color_np, (w, h), interpolation=cv2.INTER_AREA)

        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0

        psnr = Peak_Signal_to_Noise_Ratio(color_np, gt_color_np)

        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        max_depth = np.max(gt_depth_np)
        axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        cv2.imwrite(osp.join(self.depth_dir, f"depth_{str(ind).zfill(5)}.png"), cv2.convertScaleAbs(depth_np))
        cv2.imwrite(osp.join(self.depth_dir, f"depth_residual_{str(ind).zfill(5)}.png"), cv2.convertScaleAbs(depth_residual))

        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap="plasma")
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        cv2.imwrite(osp.join(self.color_dir, f"color_{str(ind).zfill(5)}.png"), cv2.cvtColor(cv2.convertScaleAbs(color_np * 255.0), cv2.COLOR_BGR2RGB))
        cv2.imwrite(osp.join(self.color_dir, f"color_residual_{str(ind).zfill(5)}.png"), cv2.cvtColor(cv2.convertScaleAbs(color_residual * 255.0), cv2.COLOR_BGR2RGB))

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(osp.join(self.img_dir, "{:05d}.jpg".format(ind)), bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        plt.close()

        return psnr
