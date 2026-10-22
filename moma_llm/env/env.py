# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import logging
import os
import random
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pybullet as pb
from gymnasium.utils import seeding
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image
import open3d as o3d
from moma_llm.navigation.navigation import find_floor_idx
from moma_llm.tasks.object_search_task import ObjectSearchTask
from moma_llm.topology.room_graph import (create_room_object_graph,
                                           get_closest_node)
from moma_llm.topology.simpleslam import SimpleSlam
from moma_llm.topology.topology import TopologyMapping, detect_rooms
from moma_llm.topology.graph import plot_graph
from moma_llm.utils.constants import (BLOCKING_DOORS, EXTERIOR_DOORS,
                                       NODETYPE, OCCUPANCY)
from moma_llm.utils.utils import get_obj_bounding_box
import time



log = logging.getLogger(__name__)


class OurIGibsonEnv(iGibsonEnv):
    def __init__(self, config_file, mode, action_timestep, physics_timestep, seed, scene_id=None, *args, **kwargs):
        # need to set the seed before the constructor, but the constructor is the one loading the config... so just load the config already once
        self.set_seed(seed=seed)

        super().__init__(config_file=config_file,
                         mode=mode,
                         action_timestep=action_timestep,
                         physics_timestep=physics_timestep,
                         scene_id=scene_id,
                         *args, 
                         **kwargs)
        
        self.config["consider_open_actions"] = (not self.config["should_open_all_interior_doors"]) or self.config["allow_inside_objects_as_targets"]

        self.slam = SimpleSlam(voxel_size=self.config["voxel_size"],
                               grid_size=int(np.ceil(self.config["grid_size_meter"] / self.config["voxel_size"])),
                               sensor_range=self.config["depth_high"],
                               min_points_for_detection=self.config["min_points_for_detection"],)

        self.topology_mapping = TopologyMapping(size=self.slam.bev_map_semantic.shape,
                                                voxel_size=self.config["voxel_size"])
        self.opened_doors = list()
        self.opened_windows = list()
        self.robot_traj = list()
        
        self.f, self.ax = plt.subplots(1, 4, figsize=(17, 5), width_ratios=[7/17, 5/17, 5/17, 5/17])
        self.made_tight_layout = False
        
        self._same_dist_count = 0 ###LKY
        self._last_dist = None ###LKY
    def set_seed(self, seed):
        if seed <= 0:
            seed = None
        self.np_random, _ = seeding.np_random(seed)
        # igibson internally uses a lot of np.random during scene initialization
        np.random.seed(seed)

    def load_reasonable_trav_map(self):
        """The gt traversibility maps from igibson are pretty garbage as they were recorded with closed doors. So fill the doors back in..."""
        def _world_to_orig_map(xy):
            return np.flip((np.array(xy) / self.scene.trav_map_default_resolution + self.scene.trav_map_original_size / 2.0)).astype(int)
        
        maps_path = os.path.join(self.scene.scene_dir, "layout")
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.scene.floor_map = []
        self.scene.floor_graph = []
        
        doors = self.scene.objects_by_category["door"]
        
        for floor in range(len(self.scene.floor_heights)):
            trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor))))

            for d in doors:
                if find_floor_idx(env=self, z=d.get_position()[2]) == floor:
                    world_frame_vertex_positions = get_obj_bounding_box(d)
                    map_vertex_positions = _world_to_orig_map(world_frame_vertex_positions[:, :2])
                    
                    x12 = sorted(map_vertex_positions[:, 0])
                    y12 = sorted(map_vertex_positions[:, 1])
                    trav_map[x12[0]:x12[1], y12[0]:y12[1]] = 255        

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (self.scene.trav_map_size, self.scene.trav_map_size))
            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            if self.scene.build_graph:
                self.scene.build_trav_graph(maps_path, floor, trav_map)            
            self.scene.floor_map.append(trav_map)
                
    def load(self):
        super().load()
        self.load_reasonable_trav_map()
        
    def reset(self):
        # igibson internally uses a lot of np.random during scene initialization
        np.random.seed(self.np_random.integers(2**32 - 1))
        self.robot_traj = list()
        self.slam.reset()

        self.rgb_frames = list()
        self.episode_info = {"num_low_level_steps": 0,
                             "num_high_level_steps": 0,
                             "scene_id": self.scene.scene_id,
                             "magic_open_actions": 0,
                             }
        self.episode_room_sem_acc = []
        self.episode_room_purity = []
        obs = super().reset()
        self.episode_info.update(self.task.task_info)
        self.obj_to_neglect = EXTERIOR_DOORS[self.simulator.scene.scene_id]
        if self.config["should_open_all_interior_doors"]:
            self.opened_doors = [o.name for o in self.scene.objects_by_category["door"] if o.name not in self.obj_to_neglect + BLOCKING_DOORS.get(self.scene.scene_id, [])]
        else:
            self.opened_doors = list()
        self.opened_windows = list()

        return obs

    def load_task_setup(self):
        if self.config["task"] == "object_search":   ########自己定义的 
            self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
            # s = 0.5 * G * (t ** 2)
            drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
            assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

            # ignore the agent's collision with these body ids
            self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
            # ignore the agent's collision with these link ids of itself
            self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

            # discount factor
            self.discount_factor = self.config.get("discount_factor", 0.99)

            # domain randomization frequency
            self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
            self.object_randomization_freq = self.config.get("object_randomization_freq", None)            

            self.task = ObjectSearchTask(self)  ###作者模仿igibson写的
        else:
            super().load_task_setup()

    def _add_frontier_points_to_rooms(self, occupancy_map, separated_voronoi_graph, room_object_graph):
        """find frontiers, add the center of each frontier to the room node that has the closest node to it in the separated_voronoi_graph"""
        robot_pos = self.robots[0].base_link.get_position()
        frontier_centers_meter, frontier_img, frontier_classification = self.slam.get_frontiers(robot_pos, occupancy_map=occupancy_map)
        if len(frontier_centers_meter):
            closest_nodes, _closest_nodes_dist = get_closest_node(query_coords_world=frontier_centers_meter, graph=separated_voronoi_graph, slam=self.slam)
            for close_node, frontier_point in zip(closest_nodes, frontier_centers_meter):
                room_id = separated_voronoi_graph.nodes.get(tuple(close_node))["room_id"]
                room_node = room_object_graph.nodes.get(NODETYPE.roomname(room_id))
                room_node["frontier_points"].add((tuple(frontier_point), frontier_classification[tuple(frontier_point)]))   
        return frontier_centers_meter, frontier_img     


    def get_state(self, compute_scene_graph: bool = False):
        # HACK as some doors like to close again over time
        if self.episode_info["num_low_level_steps"] % 15 == 0:
            for door in self.opened_doors:
                self.scene.objects_by_name[door].states[object_states.Open].set_value(True, fully=True)
        
        state = super().get_state()
        self.rgb_frames.append(state["rgb"])
        def check_target_match(state, target_id_file="target_id.txt", find_target_file="rgb_find_target.txt"):
            """
            检查target_id.txt中的ID是否与state["ins_seg"]去重后的值匹配
            参数:
                state: 包含ins_seg二维数组的状态字典
                target_id_file: 存储目标ID的文件路径
                find_target_file: 结果标记文件路径
            返回:
                bool: 是否存在匹配
            """
            # 获取去重后的ins_seg值并转换为字符串集合
            unique_values = set(map(str, np.unique(state["ins_seg"])))
            # print(unique_values)
            
            try:
                # 读取target_id文件中的所有ID
                with open(target_id_file, 'r') as f:
                    target_ids = {line.strip() for line in f if line.strip()}
                
                # 检查交集
                match_found = not unique_values.isdisjoint(target_ids)
                if match_found:
                    # 写入结果
                    with open(find_target_file, 'w') as f:
                        f.write("True")
                
                    print(f"匹配检查结果: {match_found} | "
                        f"Unique值数量: {len(unique_values)} | "
                        f"TargetID数量: {len(target_ids)}")
                return match_found
                
            except FileNotFoundError:
                print(f"[错误] {target_id_file} 文件不存在")
                return False
        
        check_target_match(state=state)

        robot_pos, robot_orn = self.robots[0].base_link.get_position_orientation()

        self.slam.update(state, extrinsic=self.simulator.renderer.V, scene=self.scene)

        robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos
        self.robot_traj.append(robot_in_wf)
        state["robot_traj"] = self.robot_traj

        if compute_scene_graph:
            wall_map = self.topology_mapping.update_maps(self.slam)
            voronoi_graph = self.topology_mapping.compute_voronoi_graph(self.slam, wall_map=wall_map)
            sparse_voronoi_graph = self.topology_mapping.sparsify_topology_graph()


            separated_voronoi_graph, _edge_delete_map, door_pos = detect_rooms(self.simulator,
                                                                              self.slam,
                                                                              sparse_voronoi_graph,
                                                                              wall_map,
                                                                              sdf_scale=self.config["topology"]["room_sdf_scale"],
                                                                              thresh=self.config["topology"]["room_sdf_thresh"],
                                                                              voxel_size=self.slam.voxel_size,
                                                                              obj_to_neglect=self.obj_to_neglect,
                                                                              opened_windows=self.opened_windows)
            
            room_graph, room_object_graph = create_room_object_graph(simulator=self.simulator,
                                                                     slam=self.slam,
                                                                     vor_graph=sparse_voronoi_graph,
                                                                     separated_vor_graph=separated_voronoi_graph,
                                                                     obj_to_neglect=self.obj_to_neglect,
                                                                     opened_doors=self.opened_doors,
                                                                     opened_windows=self.opened_windows,
                                                                     use_viewpoint_assignment=self.config["use_viewpoint_assignment"],)

            frontier_centers_meter, frontier_img = self._add_frontier_points_to_rooms(occupancy_map=self.slam.bev_map_occupancy,
                                                                                      separated_voronoi_graph=separated_voronoi_graph,
                                                                                      room_object_graph=room_object_graph)
            state["frontier_img"] = frontier_img

            closest_nodes, _closest_nodes_dist = get_closest_node(query_coords_world=np.array([robot_pos[:2]]), graph=separated_voronoi_graph, slam=self.slam)
            state["robot_current_room"] = NODETYPE.roomname(separated_voronoi_graph.nodes.get(tuple(closest_nodes[0]))["room_id"])

            state["room_graph"] = room_graph
            state["voronoi_graph"] = sparse_voronoi_graph
            state["separated_voronoi_graph"] = separated_voronoi_graph
            state["room_object_graph"] = room_object_graph
            state["door_pos"] = door_pos 

#####################    加一个state来保存clipped_map  ####################
            clipped_map = self.slam.update_2d_map()
            state["clipped_map"] = clipped_map
            
##############################################################################################################
            # current_room = state.get("robot_current_room", "room-0")  # 如果没有"robot_current_room"，则默认使用"room-0"
            
            # if rgb_image is None:
            #     print("No RGB image found in the state.")
            #     return

            # # 创建房间文件夹路径
            # room_folder = os.path.join("save_rgb", current_room)

            # # 如果文件夹不存在，则创建
            # if not os.path.exists(room_folder):
            #     os.makedirs(room_folder)
            #     print(f"Created folder: {room_folder}")

            # # 获取 output_rgb 文件夹中所有图片，并按文件名排序
            # output_rgb_folder = "output_rgb"
            # if not os.path.exists(output_rgb_folder):
            #     os.makedirs(output_rgb_folder)

            # # 获取文件夹中所有已保存的图片文件，按文件名排序
            # image_files = sorted([f for f in os.listdir(output_rgb_folder) if f.endswith('.png')],
            #                     key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)

            # # 获取文件夹中最后十张图片（如果不足十张，取所有）
            # last_images = image_files[-10:]
            
            # # 使用当前步骤编号和图片编号生成新的文件名，并保存
            # for idx, image_file in enumerate(last_images, start=1):
            #     # 生成新的文件名，避免命名冲突
            #     new_filename = f"image_{self.episode_info['num_low_level_steps']}_{idx}.png"
            #     new_filepath = os.path.join(room_folder, new_filename)

            #     # 复制图片内容
            #     img_path = os.path.join(output_rgb_folder, image_file)
            #     img = plt.imread(img_path)
            #     plt.imsave(new_filepath, img)
                
            #     print(f"Saved image to {new_filepath}")


        self.episode_info["num_low_level_steps"] += 1
        self.episode_info["num_high_level_steps"] += compute_scene_graph
        if self.episode_info["num_high_level_steps"] >= self.config["max_high_level_steps"]:
            # TODO: implement this better. Problem is that atm we don't call env.step() which would return the done normally
            self.episode_info["failure_reason"] = "max_high_level_steps timeout"
        self.episode_info["dist_travelled"] = np.linalg.norm(np.diff(np.stack(self.robot_traj)[:, :2, 3], axis=0), axis=-1).sum()

        current_dist = self.episode_info["dist_travelled"]
        # 检查距离是否变化
        if self._last_dist is not None and abs(current_dist - self._last_dist) < 1e-3:  # 使用小阈值比较浮点数
            self._same_dist_count += 1
        else:
            self._same_dist_count = 0
            self._last_dist = current_dist
            # # 距离变化时写入0
            # with open('stuck.txt', 'w') as f:
            #     f.write('0')
            
        print(f"[Trajectory] Current distance travelled: {current_dist:.3f} meters")

        # 检查是否卡住
        if self._same_dist_count >= 20:
            print("[Trajectory] Robot appears to be stuck! Terminating program.")
            # # 卡住时写入1
            # with open('stuck.txt', 'w') as f:
            #     f.write('1')

        return state



    def visualize(self, state=None):
        if state is None:
            state = self.get_state(compute_scene_graph=True)

        self.f.suptitle(f"{self.scene.scene_id}, {self.task.task_description}")

        margin = 20
        occupied_indices = np.where(self.slam.bev_map_occupancy == OCCUPANCY.OCCUPIED)
        min_x, max_x = max(0, np.min(occupied_indices[0])-margin), min(np.max(occupied_indices[0]+margin), self.slam.bev_map_occupancy.shape[0])
        min_y, max_y = max(0, np.min(occupied_indices[1])-margin), min(np.max(occupied_indices[1]+margin), self.slam.bev_map_occupancy.shape[1])
###lky   画出目标物的状态
        # patches = []
        # for o in self.task.reachable_targets:
        #     if o.name in state["room_object_graph"].nodes:
        #         c = "lime"
        #     elif self.task.new_obj_relations.get(o.name, 2*[None])[1] == object_states.Inside:
        #         c = "salmon"
        #     else:
        #         c = "orangered"
        #     world_frame_vertex_positions = get_obj_bounding_box(o)
        #     map_vertex_positions = self.slam.world2voxel(world_frame_vertex_positions[:, :2])
        #     x12 = sorted(map_vertex_positions[:, 1])
        #     y12 = sorted(map_vertex_positions[:, 0])
        #     # ensure min size on the map of 4x4
        #     x12[1] = max(x12[1], x12[0] + 4)
        #     y12[1] = max(y12[1], y12[0] + 4)
        # #     p = Polygon([(x12[0], y12[0]), (x12[1], y12[0]), (x12[1], y12[1]), (x12[0], y12[1])], facecolor=c, fill=True, alpha=0.7)
        # #     patches.append(p)

        #     min_x, max_x = min(min_x, y12[0] - margin), max(max_x, y12[1] + margin)
        #     min_y, max_y = min(min_y, x12[0] - margin), max(max_y, y12[1] + margin)
        bounds = (min_x, max_x, min_y, max_y)

        # 绘制第一个子图: room_object_graph
        plot_graph(state["room_object_graph"], self.slam.bev_map_occupancy, ax=self.ax[1], bounds=bounds)
        self.ax[1].set_title("room_object_graph")

        # 绘制第二个子图: separated_voronoi_graph (带前沿)
        plot_graph(state["separated_voronoi_graph"], self.slam.bev_map_occupancy + 5 * state["frontier_img"], ax=self.ax[2], bounds=bounds)
        self.ax[2].set_title("separated_voronoi_graph + frontiers")

        # 为新的子图，绘制 gt 房间类型颜色的图
        room_colors = {}
        node_colors = []
        for node in state["separated_voronoi_graph"].nodes:
            world_coords = self.unwrapped.slam.voxel2world(node)
            room_name = self.unwrapped.scene.get_room_type_by_point(world_coords)
            
            if room_name not in room_colors:  # 如果房间类型是新的，生成新的随机颜色
                random_color = [random.random() for _ in range(3)]  # 生成随机颜色
                room_colors[room_name] = random_color
            else:
                random_color = room_colors[room_name]
            
            node_colors.append(random_color)  # 分配颜色给节点


        # 绘制第三个子图: separated_voronoi_graph (gt房间颜色)
        plot_graph(state["separated_voronoi_graph"], self.slam.bev_map_occupancy + 5 * state["frontier_img"], ax=self.ax[3], bounds=bounds, node_colors=node_colors)
        self.ax[3].set_title("separated_voronoi_graph (GT Room Colors)")

        # 绘制目标与机器人轨迹
        robot_traj_pixel = self.slam.world2voxel(np.stack(self.robot_traj)[:, :3, 3])[:, :2]
        norm = mpl.colors.Normalize(vmin=0, vmax=2000)
        xy = robot_traj_pixel[:, ::-1].reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])
        for ax in self.ax[1:]:
            coll = LineCollection(segments, cmap=plt.cm.gray)
            coll.set_array(norm(np.arange(xy.shape[0])))
            ax.add_collection(coll)
            yaw = pb.getEulerFromQuaternion(self.robots[0].get_orientation())[2]
            ax.arrow(robot_traj_pixel[-1, 1], 
                    robot_traj_pixel[-1, 0], 
                    2 * np.sin(yaw), 
                    2 * np.cos(yaw), 
                    width=1.5, 
                    head_width=3.0,
                    fc="w")
            ax.set_axis_off()

        self.ax[2].scatter(state["door_pos"][:, 1], state["door_pos"][:, 0], c="r", s=35.0, marker="x")

        if not self.made_tight_layout:
            self.f.tight_layout()
            self.made_tight_layout = True
        
        # 保存图像
        def save_image_with_unique_name(file_path):
    # 获取文件名和扩展名
            base_name, ext = os.path.splitext(file_path)
            count = 1
            # 检查文件夹中是否已经存在该文件
            while os.path.exists(file_path):
                # 如果存在同名文件，修改文件名，加上编号
                file_path = f"{base_name}_{count}{ext}"
                count += 1
            # 保存图像
            self.f.savefig(file_path, bbox_inches='tight')
            print(f"Image saved as {file_path}")

    # 调用示例
        save_image_with_unique_name('runtime/image.png')
        # np.save('slam_bev_map_occupancy.npy', self.slam.bev_map_occupancy)


    def close(self):
        super().close()
        cv2.destroyAllWindows()
        plt.close(self.f)

    def plot_object_position(self, obj_name, color="lime", marker="x"):
        if isinstance(obj_name, str):
            pos = self.scene.objects_by_name[obj_name].get_position()
        else:
            pos = np.array(obj_name)
        pos_world = self.slam.world2voxel(pos[:2])
        self.ax[1].scatter(pos_world[1], pos_world[0], c=color, s=35.0, marker=marker)


def create_igibson_env(config_file, control_freq, scene_id, seed):
    low_level_env = OurIGibsonEnv(config_file=config_file,
                                  mode="gui_non_interactive" if os.environ["DISPLAY"] else "headless",        #远程改成"headless"
                                  action_timestep=1 / control_freq,
                                  physics_timestep=1 / 120.0,
                                  scene_id=scene_id,
                                  use_pb_gui=False,
                                  seed=seed)
    return low_level_env
