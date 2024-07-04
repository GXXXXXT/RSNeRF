import numpy as np
import open3d as o3d

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ])

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [
        0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)

class Visualizer:
    def __init__(self, args, roadslam):
        self.args = args
        self.roadslam = roadslam
        self.one_frame = False
        self.pose_history = []

    def spin(self, share_data):
        print("visualization process start")
        engine = o3d.visualization.Visualizer()
        engine.create_window(window_name="Road-SLAM", width=1280, height=720, visible=True)
        while True:
            if self.one_frame:
                try:
                    engine.clear_geometries()
                    map_states = share_data.states

                    for center in map_states["voxel_center_xyz"]:
                        half_vsize = self.args.mapper_specs["voxel_size"] / 2
                        bbox = o3d.geometry.AxisAlignedBoundingBox(center - half_vsize, center + half_vsize)
                        bbox.color = [0, 0, 1]
                        engine.add_geometry(bbox, reset_bounding_box=False)

                    camera = self.create_camera(self.pose_history[-1], scale=0.05)
                    engine.add_geometry(camera, reset_bounding_box=False)
                    engine.run()

                    self.one_frame = False

                except Exception as e:
                    print(f"error creating window {e}")

            elif share_data.stop_mapping:
                break

        engine.destroy_window()

    def create_camera(self, pose, scale=0.05):
        camera = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector((scale * CAM_POINTS) @ pose[:3, :3].transpose(-1, -2).detach().cpu().numpy() + pose[:3, 3].detach().cpu().numpy()),
            lines=o3d.utility.Vector2iVector(CAM_LINES),
        )

        camera.paint_uniform_color((0, 0, 1))
        return camera

    def insert_frame_pose(self, pose):
        self.pose_history += [pose]
        self.one_frame = True
