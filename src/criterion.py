import torch
import torch.nn as nn
import torch.nn.functional as F
from extractor import VitExtractor


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.ext = VitExtractor(model_name='dino_vits16', device='cuda:0')
        self.rgb_weight = args.criteria["rgb_weight"]
        self.depth_weight = args.criteria["depth_weight"]
        self.sdf_weight = args.criteria["sdf_weight"]
        self.fs_weight = args.criteria["fs_weight"]
        self.bk_weight = args.criteria["bk_weight"]
        self.vit_weight = args.criteria["vit_weight"]
        self.truncation = args.criteria["sdf_truncation"]
        self.max_dpeth = args.data_specs["max_depth"]

    def get_vit_feature(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        return self.ext.get_feature_from_input(x)[-1][0, 0, :]

    def get_loss(self, outputs, frame, use_color_loss=True, use_depth_loss=True, compute_sdf_loss=True,
                 compute_vit_loss=False, weight_depth_loss=False, sign=None):
        loss = 0
        loss_dict = {}

        pred_depth = outputs["depth"]
        pred_color = outputs["color"]
        pred_sdf = outputs["sdf"]
        z_vals = outputs["z_vals"]
        ray_mask = outputs["ray_mask"]
        weights = outputs["weights"]
        stamp = outputs["stamp"]

        img = []
        depth = []
        if compute_vit_loss:
            img_full = []
            truth_img_full = []
            for i in range(len(frame)):
                img += [frame[i].rgb.cuda()[frame[i].sample_mask]]
                depth += [frame[i].depth.cuda()[frame[i].sample_mask]]
                truth_img_full.append(frame[i].rgb.permute(2, 0, 1).cuda())
                pred_color_mask = pred_color[(stamp == i).squeeze(1)]
                img_full.append(torch.where((frame[i].depth == 0).unsqueeze(2),
                                            frame[i].rgb,
                                            F.embedding(frame[i].scope.cuda(), torch.cat((pred_color_mask,
                                                                                          torch.zeros(frame[i].N_rays - pred_color_mask.shape[0],
                                                                                                      pred_color_mask.shape[1],
                                                                                                      dtype=pred_color.dtype,
                                                                                                      device=pred_color.device)), dim=0))).permute(2, 0, 1).cuda())

                # truth_img = truth_img_full[-1].permute(1, 2, 0).detach().cpu()
                # gt_img = img_full[-1].permute(1, 2, 0).detach().cpu()

            # if sign == 'tracking':
                # import cv2
                # cv2.imshow("truth_img", cv2.cvtColor(truth_img.numpy(), cv2.COLOR_BGR2RGB))
                # cv2.imshow("gt_img", cv2.cvtColor(gt_img.numpy(), cv2.COLOR_BGR2RGB))
                # cv2.waitKey(1)

            truth_img_full = torch.stack(truth_img_full, dim=0)
            ref = self.get_vit_feature(truth_img_full).detach()
            img_full = torch.stack(img_full, dim=0)
        else:
            for i in range(len(frame)):
                img += [frame[i].rgb.cuda()[frame[i].sample_mask]]
                depth += [frame[i].depth.cuda()[frame[i].sample_mask]]

        img = torch.cat(img, dim=0).unsqueeze(0)
        depth = torch.cat(depth, dim=0).unsqueeze(0)

        gt_depth = depth[ray_mask]
        gt_color = img[ray_mask]

        # color_loss = self.compute_loss(
        #     gt_color, pred_color, loss_type='l1')
        if use_color_loss:
            color_loss = (gt_color - pred_color).abs().mean()
            loss += self.rgb_weight * color_loss
            loss_dict["color_loss"] = color_loss.item()

        if use_depth_loss:
            valid_depth = (gt_depth > 0.01) & (gt_depth < self.max_dpeth)
            depth_loss = (gt_depth - pred_depth).abs()

            if weight_depth_loss:
                depth_var = weights*((pred_depth.unsqueeze(-1) - z_vals)**2)
                depth_var = torch.sum(depth_var, -1)
                tmp = depth_loss/torch.sqrt(depth_var+1e-10)
                valid_depth = (tmp < 10*tmp.median()) & valid_depth
            depth_loss = depth_loss[valid_depth].mean()
            loss += self.depth_weight * depth_loss
            loss_dict["depth_loss"] = depth_loss.item()

        if compute_sdf_loss:
            fs_loss, bk_loss, sdf_loss = self.get_sdf_loss(
                z_vals, gt_depth, pred_sdf,
                truncation=self.truncation,
                loss_type='l2'
            )
            loss += self.fs_weight * fs_loss
            # loss += self.bk_weight * bk_loss
            loss += self.sdf_weight * sdf_loss
            loss_dict["fs_loss"] = fs_loss.item()
            # loss_dict["bk_loss"] = bk_loss.item()
            loss_dict["sdf_loss"] = sdf_loss.item()

        if compute_vit_loss:
            loss_vit = F.mse_loss(self.get_vit_feature(img_full), ref)
            loss += self.vit_weight * loss_vit
            loss_dict["vit_loss"] = loss_vit.item()

        loss_dict["loss"] = loss.item()
        # for k, v in loss_dict.items():
        #     print(f"{k}: {v}")
        # print()
        return loss, loss_dict

    def forward(self, outputs, frame, compute_vit_loss=False, weight_depth_loss=False, sign=None):
        return self.get_loss(outputs, frame, compute_vit_loss=compute_vit_loss, weight_depth_loss=weight_depth_loss, sign=sign)

    def compute_loss(self, x, y, mask=None, loss_type="l2"):
        if mask is None:
            mask = torch.ones_like(x).bool()
        if loss_type == "l1":
            return torch.mean(torch.abs(x - y)[mask])
        elif loss_type == "l2":
            return torch.mean(torch.square(x - y)[mask])

    def get_masks(self, z_vals, depth, epsilon):

        front_mask = torch.where(
            z_vals < (depth - epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        back_mask = torch.where(
            z_vals > (depth + epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        depth_mask = torch.where((depth > 0.0) & (depth < self.max_dpeth), torch.ones_like(depth), torch.zeros_like(depth))
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        num_fs_samples = torch.count_nonzero(front_mask).float()
        num_bk_samples = torch.count_nonzero(back_mask).float()
        num_sdf_samples = torch.count_nonzero(sdf_mask).float()
        num_samples = num_sdf_samples + num_fs_samples + num_bk_samples
        fs_weight = 1.0 - num_fs_samples / num_samples
        bk_weight = 1.0 - num_bk_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, back_mask, sdf_mask, fs_weight, bk_weight, sdf_weight

    def get_sdf_loss(self, z_vals, depth, predicted_sdf, truncation, loss_type="l2"):

        front_mask, back_mask, sdf_mask, fs_weight, bk_weight, sdf_weight = self.get_masks(z_vals, depth.unsqueeze(-1).expand(*z_vals.shape), truncation)
        fs_loss = self.compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type=loss_type,) * fs_weight
        sdf_loss = self.compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, depth.unsqueeze(-1).expand(*z_vals.shape) * sdf_mask, loss_type=loss_type,) * sdf_weight
        bk_loss = self.compute_loss(predicted_sdf * back_mask, -torch.ones_like(predicted_sdf) * back_mask, loss_type=loss_type,) * bk_weight

        return fs_loss, bk_loss, sdf_loss
