import sys
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .voxel_helpers import ray_intersect, ray_sample


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    all_size = (param_size + buffer_size) / 1024**2  # in MB
    return all_size


def MSE(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = torch.tensor((image_pred-image_gt)**2)
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def Peak_Signal_to_Noise_Ratio(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(MSE(image_pred, image_gt, valid_mask, reduction))


def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter):
    if current_iter < total_reg_iter:
        freq_mask = torch.zeros(pos_enc_length)  # all invisible
        ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1
        ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
        int_ptr = int(ptr)
        freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
        freq_mask[int_ptr * 3: int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
        return torch.clip(freq_mask, 1e-8, 1 - 1e-8)  # for numerical stability
    else:
        return torch.ones(pos_enc_length)


@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)   # [N, 8, 1]
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)   # [N, 8, 16]

    point_feats = (weights * point_feats).sum(1)
    return point_feats


def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
            point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel


@torch.enable_grad()
def get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size):
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / voxel_size + 0.5).unsqueeze(1)
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5
    feats = trilinear_interp(p, q, point_feats).float()
    # if self.args.local_coord:
    # feats = torch.cat([(p-.5).squeeze(1).float(), feats], dim=-1)
    return feats


@torch.enable_grad()
def get_features(samples, map_states, voxel_size):
    # encoder states
    point_feats = map_states["voxel_vertex_idx"].cuda()  # [1688, 8]
    point_xyz = map_states["voxel_center_xyz"].cuda()   # [1688, 3]
    values = map_states["voxel_vertex_emb"].cuda()

    # ray point samples
    sampled_idx = samples["sampled_point_voxel_idx"].long()   # [ray_o.shape[0]]
    sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)   # [ray_o.shape[0], 3]
    sampled_dis = samples["sampled_point_distance"]   # [ray_o.shape[0]]

    point_xyz = F.embedding(sampled_idx, point_xyz)  # [ray_o.shape[0], 3]
    point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)  # [ray_o.shape[0], 504], [ray_o.shape[0], 128]
    feats = get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size)
    inputs = {"dists": sampled_dis, "emb": feats}
    return inputs


@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    points_all = map_states["voxel_center_xyz_all"]
    values = map_states["voxel_vertex_emb"]

    chunk_size = 32
    res = bits  # -1

    @torch.no_grad()
    def get_scores_once(feats, points, values):
        # sample points inside voxels
        start = -.5
        end = .5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features(
            {
                "sampled_point_xyz": sampled_xyz,
                "sampled_point_voxel_idx": sampled_idx,
                "sampled_point_ray_direction": None,
                "sampled_point_distance": None,
            },
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points_all,
                "voxel_vertex_emb": values,
            },
            voxel_size
        )

        # evaluation with density
        sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
        return sdf_values.reshape(-1, res ** 3, 4).detach().cpu()

    return torch.cat([get_scores_once(feats[i: i + chunk_size],
                                      points[i: i + chunk_size],
                                      values) for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


@torch.no_grad()
def eval_points(sdf_network, map_states, sampled_xyz, sampled_idx, voxel_size):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz_all"]
    values = map_states["voxel_vertex_emb"]

    # sampled_xyz = sampled_xyz.reshape(1, 3) + points.unsqueeze(1)
    # sampled_idx = sampled_idx[None, :].expand(*sampled_xyz.size()[:2])
    sampled_idx = sampled_idx.reshape(-1)
    sampled_xyz = sampled_xyz.reshape(-1, 3)

    if sampled_xyz.shape[0] == 0:
        return

    field_inputs = get_features(
        {
            "sampled_point_xyz": sampled_xyz,
            "sampled_point_voxel_idx": sampled_idx,
            "sampled_point_ray_direction": None,
            "sampled_point_distance": None,
        },
        {
            "voxel_vertex_idx": feats,
            "voxel_center_xyz": points,
            "voxel_vertex_emb": values,
        },
        voxel_size
    )

    # evaluation with density
    sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
    return sdf_values.reshape(-1, 4)[:, :3].detach().cpu()

    # return torch.cat([
    #     get_scores_once(feats[i: i + chunk_size],
    #                     points[i: i + chunk_size], values)
    #     for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


def render_rays(
        rays_o,
        rays_d,
        map_states,
        sdf_network,
        step_size,
        voxel_size,
        truncation,
        max_voxel_hit,
        max_distance,
        exposure=None,
        chunk_size=50000,
        return_raw=False,
):
    centres = map_states["voxel_center_xyz"]
    childrens = map_states["voxel_structure"]

    stamp = rays_o[:, :, 3:]
    rays_o = rays_o[:, :, :3].contiguous()

    intersections, hits = ray_intersect(rays_o, rays_d, centres, childrens, voxel_size, max_voxel_hit, max_distance)
    assert (hits.sum() > 0)

    ray_mask = hits.view(1, -1)
    intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }

    rays_o = rays_o[ray_mask].reshape(-1, 3)
    rays_d = rays_d[ray_mask].reshape(-1, 3)
    stamp = stamp[ray_mask].reshape(-1, 1).long()

    samples = ray_sample(intersections, step_size=step_size)

    sampled_depth = samples['sampled_point_depth']
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits
    sample_mask = sampled_idx.ne(-1)
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0

    sampled_xyz = ray(rays_o.unsqueeze(
        1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2))
    sampled_dir = rays_d.unsqueeze(1).expand(*sampled_depth.size(), rays_d.size()[-1])
    sampled_dir = sampled_dir / (torch.norm(sampled_dir, 2, -1, keepdim=True) + 1e-8)
    samples['sampled_point_xyz'] = sampled_xyz
    samples['sampled_point_ray_direction'] = sampled_dir

    # apply mask
    samples_valid = {name: s[sample_mask] for name, s in samples.items()}

    num_points = samples_valid['sampled_point_depth'].shape[0]

    field_outputs = []
    if chunk_size < 0:
        chunk_size = num_points

    for i in range(0, num_points, chunk_size):
        chunk_samples = {name: s[i:i + chunk_size]
                         for name, s in samples_valid.items()}

        # get encoder features as inputs
        chunk_inputs = get_features(chunk_samples, map_states, voxel_size)

        # forward implicit fields
        chunk_outputs = sdf_network(chunk_inputs)

        field_outputs.append(chunk_outputs)

    field_outputs = {name: torch.cat([r[name] for r in field_outputs], dim=0) if value is not None else None for
                     name, value in field_outputs[0].items()}

    outputs = {'sample_mask': sample_mask}

    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1)  # [N_rays, 111]
    color = masked_scatter(sample_mask, field_outputs['color'])  # [N_rays, 111, 3]
    # colour = torch.sigmoid(colour)
    sample_mask = outputs['sample_mask']

    valid_mask = torch.where(
        sample_mask, torch.ones_like(
            sample_mask), torch.zeros_like(sample_mask)
    )

    # convert sdf to weight
    def sdf2weights(sdf_in, trunc):
        weights = torch.sigmoid(sdf_in / trunc) * torch.sigmoid(-sdf_in / trunc)

        signs = sdf_in[:, 1:] * sdf_in[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)
        mask = torch.where(
            z_vals < z_min + trunc,
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        weights = weights * mask * valid_mask
        # weights = weights * valid_mask
        return weights / (torch.sum(weights, dim=-1, keepdims=True) + 1e-8), z_min

    z_vals = samples["sampled_point_depth"]  # [N_rays, 111]

    weights, z_min = sdf2weights(sdf, truncation)  # [N_rays, 111], [N_rays, 1]
    if exposure is not None:
        exposure = exposure[stamp.squeeze(1)].unsqueeze(1).expand(*weights.size(), exposure.size()[-1])
        color = color * exposure[..., :3] + exposure[..., 3:]
    rgb = torch.sum(weights[..., None] * color, dim=-2)
    depth = torch.sum(weights * z_vals, dim=-1)

    return {
        "weights": weights,
        "color": rgb,
        "depth": depth,
        "z_vals": z_vals,
        "sdf": sdf,
        "ray_mask": ray_mask,
        "stamp": stamp,
        "raw": z_min if return_raw else None
    }


def bundle_adjust_frames(
        keyframe_graph,
        map_states,
        exp_network,
        sdf_network,
        loss_criteria,
        voxel_size,
        step_size,
        num_iterations=10,
        truncation=0.1,
        max_voxel_hit=10,
        max_distance=10,
        learning_rate=None,
        embed_optim=None,
        model_optim=None,
        update_pose=True,
):
    # optimize_params = [{'params': embeddings, 'lr': learning_rate[0]}]
    optimizers = []
    if embed_optim is not None:
        optimizers += [embed_optim]
    if model_optim is not None:
        # optimize_params += [{'params': sdf_network.parameters(),
        #                      'lr': learning_rate[0]}]
        optimizers += [model_optim]

    # optimize_params=[]
    for keyframe in keyframe_graph:
        if keyframe.stamp != 0 and update_pose:
            optimizers += [keyframe.optim]
            # keyframe.pose.requires_grad_(True)
            # optimize_params += [{
            #     'params': keyframe.pose.parameters(), 'lr': learning_rate[1]
            # }]

    # if len(optimize_params) != 0:
    #     pose_optim = torch.optim.Adam(optimize_params)
    #     optimizers += [pose_optim]
    # progress_bar = tqdm(range(0, num_iterations), position=0)
    # progress_bar.set_description("mapping iteration")
    # for cur_iter in progress_bar:
    for cur_iter in range(num_iterations):
        render = lambda rays_o, rays_d, exposure: render_rays(
            rays_o,
            rays_d,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            exposure=exposure
        )

        rays_os = []
        rays_ds = []
        exposure = []

        for i in range(len(keyframe_graph)):
            sample_mask = keyframe_graph[i].sample_rays().cuda()
            rays_d = keyframe_graph[i].rays_d[sample_mask].cuda()

            rays_d = rays_d @ keyframe_graph[i].get_rotation().cuda().transpose(-1, -2)
            rays_o = keyframe_graph[i].get_translation().reshape(1, -1).expand_as(rays_d).cuda().contiguous()  # [N_rays, 3]
            rays_o = torch.cat((rays_o, torch.ones_like(rays_o[:, :1]) * i), dim=-1)

            exposure += [{'rgb_mean': keyframe_graph[i].rgb_mean,
                          'lum': keyframe_graph[i].lum}]

            rays_os += [rays_o]
            rays_ds += [rays_d]

        rays_o = torch.cat(rays_os, dim=0).unsqueeze(0)
        rays_d = torch.cat(rays_ds, dim=0).unsqueeze(0)
        exposure = {name: torch.cat([r[name] for r in exposure], dim=0) for name in exposure[0]} if exp_network is not None else None
        exposure = exp_network(exposure) if exposure is not None else None

        outputs = render(rays_o, rays_d, exposure)

        loss, _ = loss_criteria(outputs, keyframe_graph, compute_vit_loss=False, sign='mapping')

        for optim in optimizers:
            optim.zero_grad()
        loss.backward()

        for optim in optimizers:
            optim.step()


def track_frame(
        frame_pose,
        curr_frame,
        map_states,
        exp_network,
        sdf_network,
        loss_criteria,
        voxel_size,
        N_rays=512,
        step_size=0.05,
        num_iterations=10,
        truncation=0.1,
        learning_rate=1e-3,
        max_voxel_hit=10,
        max_distance=10,
        depth_variance=False,
        model_optim=None
):
    optims = []

    init_pose = deepcopy(frame_pose).cuda() if curr_frame.pose is None else deepcopy(curr_frame.pose).cuda()
    if curr_frame.pose is None:
        init_pose.requires_grad_(True)
        optims += [torch.optim.Adam(init_pose.parameters(), lr=learning_rate)]
    else:
        optims += [None]

    if model_optim is not None:
        optims += [model_optim]

    # progress_bar = tqdm(range(0, num_iterations), position=0)
    # progress_bar.set_description("tracking iteration")
    # for cur_iter in progress_bar:
    for cur_iter in range(num_iterations):
        render = lambda rays_o, rays_d, exposure: render_rays(
            rays_o,
            rays_d,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            exposure=exposure
        )

        sample_mask = curr_frame.sample_rays().cuda()
        rays_d = curr_frame.rays_d[sample_mask].cuda()

        rays_d = rays_d @ init_pose.rotation().transpose(-1, -2)
        rays_d = rays_d.unsqueeze(0)  # [1, N_rays, 3]
        rays_o = init_pose.translation().reshape(1, 1, -1).expand_as(rays_d).cuda().contiguous()  # [1, N_rays, 3]
        rays_o = torch.cat((rays_o, torch.zeros_like(rays_o[:, :, :1])), dim=-1)
        exposure = exp_network({'rgb_mean': curr_frame.rgb_mean,
                                'lum': curr_frame.lum}) if exp_network is not None else None

        outputs = render(rays_o, rays_d, exposure)

        loss, loss_dict = loss_criteria(outputs, [curr_frame], compute_vit_loss=False, weight_depth_loss=depth_variance, sign='tracking')

        for optim in optims:
            optim.zero_grad() if optim is not None else None
        loss.backward()

        for optim in optims:
            optim.step() if optim is not None else None

    return init_pose, optims[0], outputs["ray_mask"].view(N_rays), loss_dict
