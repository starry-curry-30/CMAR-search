# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
from numba import njit, prange
import scipy.ndimage
import os
import time
import json

from igibson.utils.constants import MAX_CLASS_COUNT, SemanticClass

from moma_llm.utils.constants import OCCUPANCY, CLASS_NAME_TO_CLASS_ID, CLASS_ID_TO_CLASS_NAME
from moma_llm.navigation.frontier import find_frontiers, classify_frontiers
from moma_llm.utils.utils import get_obj_bounding_box


@njit(parallel=False)
def last_nonzero_numba(arr, value):
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[1]):
            for z in range(arr.shape[2], 0, -1):
                if arr[x, y, z] != 0:
                    value[x, y] = arr[x, y, z]
                    break
    return value

##################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualize_point_cloud(points, semantics):
    fig = plt.figure(figsize=(12, 12))  # 将图像尺寸放大
    ax = fig.add_subplot(111, projection='3d')

    # 使用语义信息进行颜色编码
    unique_labels = np.unique(semantics)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = semantics == label
        class_name = CLASS_ID_TO_CLASS_NAME.get(label, "Unknown")
        ax.scatter(points[indices, 0], points[indices, 1], points[indices, 2],
                   color=colors(i), label=f'{class_name}', marker='o', s=5)  # 缩小点的大小

        # 计算每个类别点云的中点，并在中点处标注类别名称
        mid_point = points[indices].mean(axis=0)
        ax.text(mid_point[0], mid_point[1], mid_point[2], class_name, fontsize=10, color='black')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    # 打印出所有唯一标签及其对应的名称
    print("Unique labels in seg:")
    for label in unique_labels:
        class_name = CLASS_ID_TO_CLASS_NAME.get(label, "Unknown")
        print(f"Label ID: {label}, Class Name: {class_name}")


    save_path = "./image"
    # 保存图像到指定文件路径
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

    # 显示图形
    plt.show()

    # 显示图形并阻塞程序，直到关闭图形界面后继续运行
    plt.show(block=True)

class SimpleSlam:
    def __init__(self, grid_size, voxel_size, sensor_range, min_points_for_detection):
        self.sensor_range = sensor_range
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.midpoint = grid_size // 2
        assert self.grid_size % 2 == 0
        self.data_file = 'instance_viewpoints.json' ###
        self.min_points_for_detection = min_points_for_detection
        self.reset()

    @property
    def seen_instances(self):
        return set(self.instance_viewpoints.keys())
    
    def reset(self):
        self.voxel_map = np.zeros([self.grid_size] * 3, dtype=np.float32)
        self.bev_map_semantic = np.zeros([self.grid_size] * 2, dtype=np.float32)
        self.bev_map_occupancy = np.zeros_like(self.bev_map_semantic)
        # for each instance, keep track of the closest viewpoint and distance to use it for the assignment of objects to the voronoi graph
        # instance_id: (viewpoint_position, distance-to-instance)
        self.instance_viewpoints = dict()
        self.viewpoints_info = dict()  ######lky 加入来放视点的rgb输入和位姿

    def world2voxel(self, world_coords):
        voxel_idx = np.round(world_coords / self.voxel_size).astype(int) + self.midpoint
        # assert np.logical_and(voxel_idx > 0, voxel_idx < self.grid_size - 1).all(), world_coords            ##############lky
        return voxel_idx
    
    def voxel2world(self, voxel_coords):
        if isinstance(voxel_coords, tuple):
            voxel_coords = np.array(voxel_coords)
        world_coords = (voxel_coords - self.midpoint) * self.voxel_size
        return world_coords

    # def _update_closest_viewpoints(self, scene, instance_seg, dist, extrinsic_inv):
    #     viewpoint_pos_world = extrinsic_inv[:3, 3]
    #     instance_seg_sorted = instance_seg[instance_seg.argsort()]
    #     ins_ids, ins_idxs, ins_counts = np.unique(instance_seg_sorted, return_index=True, return_counts=True)
    #     instance_seg_by_dist = np.split(dist, ins_idxs)
        
    #     def _get_volume(instance_id):
    #         bbox = scene.objects_by_id[instance_id].bounding_box
    #         return np.prod(bbox) if (bbox is not None) else np.inf
        
    #     for instance_id, dists, count in zip(ins_ids[1:], instance_seg_by_dist[1:], ins_counts[1:]):
    #         if (count > self.min_points_for_detection) or (_get_volume(instance_id) < 0.01):
    #             min_dist = np.min(dists)
    #             if self.instance_viewpoints.get(instance_id, (None, np.inf))[1] > min_dist:
    #                 self.instance_viewpoints[instance_id] = (viewpoint_pos_world, min_dist)


############################################################### lky  为了保存最近的3个视点以及他们state[rgb] 
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def _update_closest_viewpoints(self, scene, instance_seg, dist, extrinsic_inv, state, rgb_output_folder="viewpoint_rgb"):
        # 确保输出文件夹存在
        os.makedirs(rgb_output_folder, exist_ok=True)

        # 获取当前视角的世界坐标
        viewpoint_pos_world = extrinsic_inv[:3, 3]

        # 对实例分割结果进行排序
        instance_seg_sorted = instance_seg[instance_seg.argsort()]
        ins_ids, ins_idxs, ins_counts = np.unique(instance_seg_sorted, return_index=True, return_counts=True)

        # 按实例分割的距离分组
        instance_seg_by_dist = np.split(dist, ins_idxs)

        # 定义辅助函数：计算对象体积
        def _get_volume(instance_id):
            bbox = scene.objects_by_id[instance_id].bounding_box
            return np.prod(bbox) if (bbox is not None) else np.inf

        # 遍历所有实例并更新最近视角
        for instance_id, dists, count in zip(ins_ids[1:], instance_seg_by_dist[1:], ins_counts[1:]):
            # 检查是否满足更新条件
            if (count > self.min_points_for_detection) or (_get_volume(instance_id) < 0.01):
                # 计算最小距离
                min_dist = np.min(dists)

                # 如果该实例尚未记录或当前视角更近，则更新最近视角信息
                if self.instance_viewpoints.get(instance_id, (None, np.inf))[1] > min_dist:
                    self.instance_viewpoints[instance_id] = (viewpoint_pos_world, min_dist)

                # 更新最近的三张图像
                if instance_id not in self.viewpoints_info:
                    self.viewpoints_info[instance_id] = []  # 初始化最近视角列表

                # 当前视角信息
                current_viewpoint = {
                    "camera_pose": extrinsic_inv,
                    "distance": min_dist,
                    "ins_seg": state["ins_seg"]
                }

                # 添加当前视角到最近视角列表
                self.viewpoints_info[instance_id].append(current_viewpoint)

                # 保留最近的三个视角（按距离排序）
                self.viewpoints_info[instance_id].sort(key=lambda x: x["distance"])  # 按距离排序
                self.viewpoints_info[instance_id] = self.viewpoints_info[instance_id][:3]  # 保留最近的三个视角

                # 处理最近的三张图像并计算占比
                # for i, viewpoint in enumerate(self.viewpoints_info[instance_id]):
                #     rgb_image = viewpoint["ins_seg"]

                #     # 如果是三维数组 (H, W, 1)，去掉第三维
                #     if rgb_image.ndim == 3 and rgb_image.shape[2] == 1:
                #         rgb_image = rgb_image[:, :, 0]

                #     # 计算实例在图像中的比例
                #     total_pixels = rgb_image.size  # 图像总像素数
                #     instance_pixels = np.sum(rgb_image == instance_id)  # 当前实例的像素数
                #     instance_ratio = instance_pixels / total_pixels  # 实例占比

                #     # 打印实例比例
                #     print(f"Instance {instance_id} - Image {i+1}: occupies {instance_ratio * 100:.2f}% of the image.")

                #     # 为实例分配颜色并生成彩色图像
                #     colored_image = self._assign_colors_to_instances(rgb_image)

                #     # 找到实例的边界框
                #     mask = (rgb_image == instance_id)
                #     rows = np.any(mask, axis=1)
                #     cols = np.any(mask, axis=0)
                #     ymin, ymax = np.where(rows)[0][[0, -1]]
                #     xmin, xmax = np.where(cols)[0][[0, -1]]

                #     # 在彩色图像上绘制白色边框
                #     cv2.rectangle(colored_image, (xmin, ymin), (xmax, ymax), (1, 1, 1), thickness=8)

                #     # 保存图像到文件夹
                #     image_path = os.path.join(rgb_output_folder, f"{instance_id}_image_{i+1}.png")
                #     plt.imsave(image_path, colored_image)

# 辅助函数：为实例分配颜色
    def _assign_colors_to_instances(self, rgb_image):
        unique_ids = np.unique(rgb_image)
        num_instances = len(unique_ids)

        # 创建一个映射表，将实例 ID 映射到 [0, num_instances-1] 的范围内
        id_to_index = {instance_id: i for i, instance_id in enumerate(unique_ids)}

        # 生成随机颜色映射表
        np.random.seed(42)  # 固定随机种子以确保颜色一致
        colors = np.random.rand(num_instances, 3)  # 每个实例生成一个 RGB 颜色
        colors[id_to_index[0]] = [0, 0, 0]  # 背景（ID=0）设置为黑色

        # 将 rgb_image 中的实例 ID 映射到有效索引
        mapped_image = np.vectorize(id_to_index.get)(rgb_image)

        # 使用映射后的图像生成彩色图像
        colored_image = colors[mapped_image]
        return colored_image
    @property
    def clipping_range(self):
        rng = int(1 / self.voxel_size)
        return self.midpoint - rng, self.midpoint + rng
    
    def delete_obj_from_voxel_map(self, obj):
        world_frame_vertex_positions = get_obj_bounding_box(obj)
        map_vertex_positions = self.world2voxel(world_frame_vertex_positions[:, :2])
        x12 = sorted(map_vertex_positions[:, 0])
        y12 = sorted(map_vertex_positions[:, 1])
        self.voxel_map[x12[0]-1:x12[1]+1, y12[0]-1:y12[1]+1] = 0        

    @staticmethod
    def _get_outside_window_filter(state):
        window_mask = state["seg"] == CLASS_NAME_TO_CLASS_ID["window"]
        wall_mask = state["seg"] == CLASS_NAME_TO_CLASS_ID["walls"]
        if np.any(window_mask) > 0:
            window_instances = list(np.unique(state["ins_seg"][window_mask]))
            masked_windows = copy.deepcopy(window_mask).squeeze(-1)
            for window_instance in window_instances:
                single_window_mask = state["ins_seg"] == window_instance
                single_window_mask = cv2.dilate(single_window_mask.astype(np.uint8), np.ones((1, 1), np.uint8), iterations = 1)
                # check if window occupies any pixel in row 0
                min_horiz, max_horiz = np.min(single_window_mask.nonzero()[1]), np.max(single_window_mask.nonzero()[1])
                min_vert, max_vert = np.min(single_window_mask.nonzero()[0]), np.max(single_window_mask.nonzero()[0])
                # occupy outside lines
                if np.any(single_window_mask[0, :]) > 0:
                    single_window_mask[0:2, min_horiz:max_horiz] = True
                if np.any(single_window_mask[:, 0]) > 0:    
                    single_window_mask[min_vert:max_vert, 0:2] = True
                if np.any(single_window_mask[:, -1]) > 0:
                    single_window_mask[min_vert:max_vert, -2:-1] = True
                if np.any(single_window_mask[-1, :]) > 0:
                    single_window_mask[-2:-1, min_horiz:max_horiz] = True
                size_param = np.mean([max_horiz - min_horiz, max_vert - min_vert]) # TODO: Tune dilation size based on distance
                # fill holes in the single_window_mask
                instance_filled_sobel = scipy.ndimage.binary_fill_holes(single_window_mask) > 0
                masked_windows[instance_filled_sobel] = True
            masked_windows = cv2.dilate(masked_windows.astype(np.uint8), np.ones((5, 5), np.uint8), iterations = 1)
            outside_window_filter = cv2.erode(masked_windows.astype(np.uint8), np.ones((12, 12), np.uint8), iterations = 1) > 0
        else:
            outside_window_filter = np.zeros_like(state["seg"], dtype=np.uint8).squeeze(-1)
        return outside_window_filter




    def _update_voxel_map(self, state, extrinsic, scene):

        def visualize_point_cloud_with_matplotlib(point_cloud, segmentation, class_name_to_class_id):

            # 点云数据重塑
            pc = point_cloud.reshape(-1, 3)
            print("pc数组的第一行数据:", pc[0])
            # 生成颜色映射
            unique_labels = np.unique(segmentation)
            color_map = plt.get_cmap('tab20', len(unique_labels))
            colors = np.zeros((pc.shape[0], 3))


            for i, label in enumerate(unique_labels):
                mask = segmentation.flatten() == label
                colors[mask] = color_map(i)[:3]

            # 创建3D散点图
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=1)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Point Cloud with Semantic Segmentation')
            plt.show()

            # 语义分割图可视化
            seg = segmentation[:, :, 0]
            plt.figure(figsize=(10, 10))
            plt.imshow(seg, cmap='tab20')
            plt.title("Semantic Segmentation")
            plt.axis('off')

            # 打印语义标签
            class_id_to_name = {v: k for k, v in class_name_to_class_id.items()}
            unique_labels = np.unique(seg)
            for label in unique_labels:
                if label in class_id_to_name:
                    y, x = np.where(seg == label)
                    if len(x) > 0 and len(y) > 0:
                        plt.text(x.mean(), y.mean(), class_id_to_name[label], color='white', fontsize=8, ha='center',
                                 va='center',
                                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            plt.show(block = True)

      # visualize_point_cloud_with_matplotlib(state["pc"], state["seg"], CLASS_NAME_TO_CLASS_ID)
        
        pc = state["pc"].reshape(-1, 3)
        dist = np.linalg.norm(pc, axis=1)
        dist_dense = dist.reshape(state["seg"].shape)

        # FILTERING OBJECTS OUTSIDE OF WINDOWS
        outside_window_filter = self._get_outside_window_filter(state)

        # SENSING DEPTH FILTERING + WINDOW FILTERING
        within_sensing_range = (dist_dense < self.sensor_range).squeeze(-1)
        within_map_bounds = np.logical_and(~outside_window_filter, within_sensing_range)

        # overwrite everything within
        points = pc[within_map_bounds.reshape(-1)]

        rgb = copy.deepcopy(state["rgb"])


        rgb[~within_map_bounds] = 0
        seg = state["seg"][:, :, 0]

        seg[~within_map_bounds] = 0
        ins_seg = state["ins_seg"][:, :, 0]
        ins_seg[~within_map_bounds] = 0

        seg = seg[within_map_bounds].reshape(-1)
        ins_seg = ins_seg[within_map_bounds].reshape(-1)

        within_map_bounds = within_map_bounds.reshape(-1)   
        
        # Add a column of ones for the inverse transform
        points = np.c_[points, np.ones(points.shape[0])]
        
        extrinsic_inv = np.linalg.inv(extrinsic)
        self._update_closest_viewpoints(scene=scene, instance_seg=ins_seg, dist=dist[within_map_bounds], extrinsic_inv=extrinsic_inv, state=state)

        # Get map indices
        world_points = extrinsic_inv.dot(points.T).T
        voxel_idx = self.world2voxel(world_points)
        
        # voxel_idx have multiple values that refer to the same voxel. Which of those gets selected is more or less random
        # so first write the free-space values, then all other values to make sure that we don't write a freespace-class on top of an occupied-class
        is_floor = np.isin(seg, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]])
        self.voxel_map[voxel_idx[is_floor, 0], voxel_idx[is_floor, 1], voxel_idx[is_floor, 2]] = CLASS_NAME_TO_CLASS_ID["floors"]
        
        # remaining_idx = np.logical_and(~is_floor, ~np.isin(seg, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]]))
        voxel_idx_remaining = voxel_idx[~is_floor]
        seg_remaining = seg[~is_floor]
        self.voxel_map[voxel_idx_remaining[:, 0], voxel_idx_remaining[:, 1], voxel_idx_remaining[:, 2]] = seg_remaining
##############################################  
        # import time
        # from plyfile import PlyData, PlyElement
        # from datetime import datetime

        # vertex = np.empty(len(world_points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # vertex['x'] = world_points[:, 0]
        # vertex['y'] = world_points[:, 1]
        # vertex['z'] = world_points[:, 2]

        # # 将numpy数组转换为PlyElement
        # vertex_element = PlyElement.describe(vertex, 'vertex')

        # # 创建PlyData对象
        # ply_data = PlyData([vertex_element])

        # # 定义保存路径和文件名，用时间戳来命名
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
        # filename = f"pointcloud_{timestamp}.ply"

        # # 保存PLY文件
        # ply_data.write(os.path.join(save_dir, filename))

        # print(f"点云数据已成功保存为{os.path.join(save_dir, filename)}")





    def update_2d_map(self):
        # Clip out ceiling
        clipped_map = self.voxel_map[:, :, self.clipping_range[0] : self.clipping_range[1]]

        # sum from top to bottom
        value = np.zeros_like(self.bev_map_semantic)
        bev_map = last_nonzero_numba(clipped_map, value)
        self.bev_map_semantic = bev_map
        
        occupancy_map = np.zeros_like(self.bev_map_semantic)
        occupancy_map[bev_map > 0] = OCCUPANCY.OCCUPIED
        occupancy_map[np.isin(bev_map, [CLASS_NAME_TO_CLASS_ID["floors"], CLASS_NAME_TO_CLASS_ID["carpet"]])] = OCCUPANCY.FREE
        self.bev_map_occupancy = occupancy_map

        return clipped_map

    def update(self, state, extrinsic, scene):
        self._update_voxel_map(state, extrinsic, scene=scene)
        self.update_2d_map()

    def get_frontiers(self, agent_pos_meter, occupancy_map=None):
        if occupancy_map is None:
            occupancy_map = self.bev_map_occupancy
        assert set(np.unique(occupancy_map)) == set(OCCUPANCY), "occupancy map has to follow the values of the OCCUPANCY enum"
        occupancy_map = occupancy_map.astype(np.uint8)
        
        robot_pos_pixel = self.world2voxel(agent_pos_meter)[:2]
        frontier_centers_pixel, frontier_img = find_frontiers(occupancy_map, 
                                                              robot_pos_pixel,
                                                              smoothing_kernel_size=3)
        frontier_centers_world = self.voxel2world(frontier_centers_pixel)
        
        frontier_classification = classify_frontiers(slam=self, frontier_img=frontier_img, frontier_centers_world=frontier_centers_world)
        
        return frontier_centers_world, frontier_img, frontier_classification
        
