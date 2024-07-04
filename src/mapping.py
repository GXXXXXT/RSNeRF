import time
from copy import deepcopy
import random
import numpy as np

import torch

from criterion import Criterion
from loggers import BasicLogger
from frame import RGBDFrame
from se3pose import OptimizablePose
from utils_loc.import_util import get_decoder, get_property
from variations.render_helpers import bundle_adjust_frames, render_rays, fill_in, count_parameters, get_model_size
from utils_loc.mesh_util import MeshExtractor

torch.classes.load_library(
    './third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so')


def get_network_size(net):
    size = 0
    for param in net.parameters():
        size += param.element_size() * param.numel()
    return size / 1024 / 1024


class Mapping:
    def __init__(self, args, data_stream, logger: BasicLogger, vis=None, ckpt=None, **kwargs):
        super().__init__()
        self.args = args
        self.data_stream = data_stream
        self.logger = logger
        self.visualizer = vis
        self.loss_criteria = Criterion(args)
        self.road_decoder = get_decoder('roadmlp', args).cuda()
        print(self.road_decoder)
        print(f"road_decoder has {count_parameters(self.road_decoder)} parameters")
        print(f"road_decoder occupies {get_model_size(self.road_decoder):.2f} MB of memory")

        self.keyframe_graph = []
        self.initialized = False

        mapper_specs = args.mapper_specs

        # optional args
        # self.update_pose = False if args.data_specs["use_gt"] else True
        self.update_pose = True
        self.use_exp = args.use_exp
        self.keyframe_freq = get_property(args.debug_args, "keyframe_freq", 5)
        self.ckpt_freq = get_property(args.debug_args, "ckpt_freq", 10)
        self.final_iter = get_property(mapper_specs, "final_iter", 0)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(args.debug_args, "save_data_freq", 10)

        # required args
        # self.overlap_th = mapper_specs["overlap_th"]
        self.voxel_size = mapper_specs["voxel_size"]
        self.window_size = mapper_specs["window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"]
        self.step_size = self.step_size * self.voxel_size
        self.max_distance = args.data_specs["max_depth"]

        embed_dim = args.decoder_specs["in_dim"]
        self.render_res = args.debug_args["render_res"]
        self.mesh_freq = args.debug_args["mesh_freq"]
        self.mesher = MeshExtractor(args)

        use_local_coord = mapper_specs["use_local_coord"]
        self.embed_dim = embed_dim - 3 if use_local_coord else embed_dim
        num_embeddings = mapper_specs["num_embeddings"]
        self.embeddings = torch.zeros(
            (num_embeddings, self.embed_dim),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))
        torch.nn.init.normal_(self.embeddings, std=0.01)
        self.embed_optim = torch.optim.Adam([self.embeddings], lr=5e-3)

        self.model_optim = torch.optim.AdamW(self.road_decoder.parameters(), lr=5e-3)

        self.svo = torch.classes.svo.Octree()
        self.svo.init(512, embed_dim, self.voxel_size)

        self.frame_poses = []
        self.depth_maps = []
        self.last_tracked_frame_id = 0

        if args.cont:
            self.road_decoder.load_state_dict(ckpt['road_decoder_state'])
            self.model_optim = torch.optim.Adam(self.road_decoder.parameters(), lr=5e-3)
            self.map_states = ckpt['map_states']
            self.embeddings = ckpt['embeddings']
            self.embed_optim.load_state_dict(ckpt['embed_optim'])
            self.frame_poses = ckpt['frame_poses']
            for f, v in ckpt['keyframe'].items():
                fid, rgb, depth, K, _ = self.data_stream[f]
                frame = RGBDFrame(fid, rgb, depth, K)
                frame.pose = OptimizablePose.from_matrix(torch.tensor(self.frame_poses[int(f)], dtype=torch.float32))
                frame.optim = torch.optim.Adam(frame.pose.parameters(), lr=1e-3)
                frame.pose.load_state_dict(v['pose'])
                frame.optim.load_state_dict(v['optim'])
                frame.quadtree_sample_rays(self.n_rays)
                self.current_keyframe = frame
                self.keyframe_graph += [frame]
            self.svo = ckpt['svo']
            self.initialized = True
            print("****** mapping recover successfully ******")

    def spin(self, share_data, kf_buffer, lock):
        print("mapping process started!")
        while True:
            torch.cuda.empty_cache()
            if not kf_buffer.empty():
                lock.acquire() if lock is not None else None
                tracked_frame = kf_buffer.get()
                # self.create_voxels(tracked_frame)

                if not self.initialized:
                    if self.mesher is not None:
                        self.mesher.rays_d = tracked_frame.get_rays()
                    self.insert_keyframe(tracked_frame)
                    self.create_voxels(tracked_frame)
                    while kf_buffer.empty():
                        self.do_mapping(share_data, update_pose=self.update_pose)
                        break
                        # self.update_share_data(share_data, tracked_frame.stamp)
                    self.initialized = True
                else:
                    self.do_mapping(share_data, tracked_frame, update_pose=self.update_pose)
                    # if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                    if (tracked_frame.stamp - self.current_keyframe.stamp) > self.keyframe_freq or tracked_frame.hit_ratio < 0.1:
                        self.insert_keyframe(tracked_frame)
                        print(
                            f"********** current num kfs: {len(self.keyframe_graph)} **********")
                    self.create_voxels(tracked_frame)

                # self.create_voxels(tracked_frame)
                tracked_pose = deepcopy(tracked_frame.get_pose().detach()).cpu().numpy()
                self.frame_poses += [tracked_pose]
                self.depth_maps += [tracked_frame.depth.clone().cpu()]

                if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:
                    self.logger.log_mesh(self.extract_mesh(
                        res=self.mesh_res, clean_mesh=True), name=f"mesh_{tracked_frame.stamp:05d}.ply")

                if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                    self.save_debug_data(tracked_frame)

                if tracked_frame.stamp % self.ckpt_freq == 0 and tracked_frame.stamp > 0:
                    self.logger.log_ckpt(self, share_data, tracked_frame.stamp)

                lock.release() if lock is not None else None
            elif share_data.stop_mapping:
                break

        print(f"********** post-processing {self.final_iter} steps **********")
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(share_data, tracked_frame=None,
                            update_pose=False, update_decoder=False)

        print("******* extracting final mesh *******")
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False)
        self.logger.log_mesh(mesh)
        # try:
        #     mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False)
        #     self.logger.log_mesh(mesh)
        # except Exception as e:
        #     print("error in extract_mesh: ", e)

        self.logger.log_numpy_data(np.asarray(self.frame_poses), "frame_poses")
        self.logger.log_numpy_data(self.extract_voxels(), "final_voxels")
        print(f"road_decoder has {count_parameters(self.road_decoder)} parameters")
        print(f"road_decoder occupies {get_model_size(self.road_decoder):.2f} MB of memory")
        print("******* mapping process died *******")

    def do_mapping(
            self,
            share_data,
            tracked_frame=None,
            update_pose=True,
            update_decoder=True
    ):
        # self.map.create_voxels(self.keyframe_graph[0])
        self.road_decoder.train()
        exp_decoder = None
        if self.initialized and self.use_exp:
            exp_decoder = deepcopy(share_data.exp_decoder).cuda()
        optimize_targets = self.select_optimize_targets(tracked_frame)
        # optimize_targets = [f.cuda() for f in optimize_targets]

        bundle_adjust_frames(
            optimize_targets,
            self.map_states,
            exp_decoder,
            self.road_decoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.num_iterations if self.initialized else 100,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            learning_rate=[1e-2, 1e-3],
            embed_optim=self.embed_optim,
            model_optim=self.model_optim if update_decoder else None,
            update_pose=update_pose,
        )

        # optimize_targets = [f.cpu() for f in optimize_targets]
        self.update_share_data(share_data)
        # sleep(0.01)

    def select_optimize_targets(self, tracked_frame=None):
        targets = []
        selection_method = 'random'
        if len(self.keyframe_graph) <= self.window_size:
            targets = self.keyframe_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.keyframe_graph, self.window_size)
        elif selection_method == 'overlap':
            targets = self.keyframe_graph[-self.window_size:]

        if tracked_frame is not None and tracked_frame != self.current_keyframe:
            targets += [tracked_frame]
        return targets[::-1]

    def update_share_data(self, share_data, frameid=None):
        share_data.road_decoder = deepcopy(self.road_decoder).cpu()
        tmp_states = {}
        for k, v in self.map_states.items():
            tmp_states[k] = v.detach().cpu()
        share_data.states = tmp_states
        # self.last_tracked_frame_id = frameid

    def insert_keyframe(self, frame):
        # kf check
        print("insert keyframe")
        self.current_keyframe = frame
        self.keyframe_graph += [frame]
        # self.update_grid_features()

    def create_voxels(self, frame):
        points = frame.get_points().cuda()
        pose = frame.get_pose().cuda()
        points = points@pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')

        self.svo.insert(voxels.cpu().int())
        self.update_grid_features()

    @torch.enable_grad()
    def update_grid_features(self):
        voxels, children, features = self.svo.get_centres_and_children()
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxel_vertex_idx"] = features.cuda()
        map_states["voxel_center_xyz"] = centres
        map_states["voxel_structure"] = children
        num_embeddings = self.embeddings.shape[0]
        while features.shape[0] > num_embeddings:
            new_embeddings = torch.zeros((num_embeddings + 10000, self.embed_dim), dtype=torch.float32,
                                         device=torch.device("cuda"))
            new_embeddings[:num_embeddings, :] = self.embeddings.detach()
            new_embeddings.required_grad_ = True

            torch.nn.init.normal_(new_embeddings[num_embeddings:, :], std=0.01)
            self.embeddings = new_embeddings
            num_embeddings = self.embeddings.shape[0]
            self.embed_optim = torch.optim.Adam([self.embeddings], lr=5e-3)

        map_states["voxel_vertex_emb"] = self.embeddings
        self.map_states = map_states

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False):
        sdf_network = self.road_decoder
        sdf_network.eval()

        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        centres_all = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        voxels = voxels[~index, :]
        features = features[~index, :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = features.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["voxel_center_xyz_all"] = centres_all.cuda()
        encoder_states["voxel_vertex_emb"] = self.embeddings

        mesh = self.mesher.create_mesh(self.road_decoder, encoder_states, self.voxel_size, voxels,
                                       frame_poses=self.frame_poses[-1], depth_maps=self.depth_maps[-1],
                                       clean_mseh=clean_mesh, require_color=True, offset=-10, res=res)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, offset=-10):
        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
            self.voxel_size + offset
        print(torch.max(features)-torch.count_nonzero(index))
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame, offset=-10):
        pose = tracked_frame.get_pose().detach().cpu().numpy()
        pose[:3, 3] += offset
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels().detach().cpu().numpy()
        keyframe_poses = [p.get_pose().detach().cpu().numpy()
                          for p in self.keyframe_graph]

        for f in self.frame_poses:
            f[:3, 3] += offset
        for kf in keyframe_poses:
            kf[:3, 3] += offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": self.frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": keyframe_poses,
            "is_keyframe": (tracked_frame == self.current_keyframe)
        }, tracked_frame.stamp)
