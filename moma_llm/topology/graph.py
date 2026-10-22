# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix

import moma_llm.topology
from moma_llm.utils.constants import NODETYPE


def aggregate_close_nodes(graph, voxel_size: float, thres_meter: float):
    nodes_list = list(graph.nodes)
    nodes_stacked = np.stack(nodes_list)
    dist_matrix_meter = distance_matrix(nodes_stacked, nodes_stacked) * voxel_size
    for i, node in enumerate(list(graph.nodes)):
        if i == 0:
            continue
        if dist_matrix_meter[i, :i].min() < thres_meter:
            closest_node = nodes_list[dist_matrix_meter[i, :i].argmin()]
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                graph.add_edge(closest_node, neighbor, dist=graph.get_edge_data(node, neighbor)["dist"])
            graph.remove_node(node)
            

def sparsify_graph(topology_graph: nx.Graph, voxel_size: float, obstacle_map: np.ndarray, sdf_scale: float):
    """
    Sparsify a topology graph by removing nodes with degree 2.
    This algorithm first starts at degree-one nodes (dead ends) and
    removes all degree-two nodes until confluence nodes are found.
    Next, we find close pairs of higher-order degree nodes and
    delete all nodes if the shortest path between two nodes consists
    only of degree-two nodes.
    Args:
        graph (nx.Graph): graph to sparsify
    Returns:
        nx.Graph: sparsified graph
    """
    graph = copy.deepcopy(topology_graph)
    
    if len(graph.nodes) < 10:
        return graph
    # all nodes with degree 1 or 3+
    new_node_candidates = [node for node in list(graph.nodes) if (graph.degree(node) != 2)]
    
    new_graph = nx.Graph()
    for i, node in enumerate(new_node_candidates):
        new_graph.add_node(node)
        
    all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
    
    sampled_edges_to_add = list()
    new_nodes = set(new_graph.nodes)
    new_nodes_list = list(new_graph.nodes)
    for i in range(len(new_graph.nodes)):
        for j in range(len(new_graph.nodes)):
            if i < j:
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                path = all_path_dense_graph[node1][node2]
                for node in path[1:-1]:
                    if graph.degree(node) > 2:
                        break
                else:
                    sampled_edges_to_add.append((path[0], path[-1], np.linalg.norm(np.array(path[0]) - np.array(path[-1]))))
                    dist = [graph.edges[path[k], path[k + 1]]["dist"] for k in range(len(path) - 1)]
                    mov_agg_dist = 0
                    predecessor = path[0]
                    # connect the nodes if there is a path between them that does not go through any other of the new nodes
                    if len(path) and len(set(path[1:-1]).intersection(new_nodes)) == 0:
                        for cand_idx, cand_node in enumerate(path[1:-1]):
                            mov_agg_dist += dist[cand_idx]
                            if mov_agg_dist * voxel_size > 0.6: 
                                sampled_edges_to_add.append((predecessor, cand_node, np.linalg.norm(np.array(predecessor) - np.array(cand_node))))
                                predecessor = cand_node
                                mov_agg_dist = 0
                            else:
                                continue
                        sampled_edges_to_add.append((predecessor, path[-1], np.linalg.norm(np.array(predecessor) - np.array(path[-1]))))
    
    for edge_param in sampled_edges_to_add:
        k, l, dist = edge_param
        if k not in new_graph.nodes:
            new_graph.add_node(k)
        if l not in new_graph.nodes:
            new_graph.add_node(l)
        new_graph.add_edge(k, l, dist=dist)
    return new_graph


# def plot_graph(graph, map_underlay, edge_filter=None, ax=None, bounds=None, node_colors=None):
#     if ax is None:
#         ax = plt.gca()
#     ax.clear()

#     if bounds is None:
#         min_x, max_x, min_y, max_y = 0, map_underlay.shape[0], 0, map_underlay.shape[1]
#     else:
#         min_x, max_x, min_y, max_y = bounds
#     ax.set_xlim(min_y, max_y)
#     ax.set_ylim(max_x, min_x)
#     ax.imshow(map_underlay)
    
#     if graph is None or len(graph.nodes()) == 0:
#         return
#     special_instances_count = 0
#     if isinstance(list(graph.nodes())[0], tuple):
#         # 当节点是 tuple 类型时
#         xy = np.array(graph.nodes())
#         if node_colors is not None:
#             ax.scatter(xy[:, 1], xy[:, 0], c=node_colors, s=5)
#         else:
#             # 默认情况下，所有节点的颜色是红色
#             ax.scatter(xy[:, 1], xy[:, 0], c='r', s=3)
#     else:
#         xy = []
#         node_types = []
#         special_object_categories = {"cup", "lollipop", "milk", "notebook", "olive", "yogurt", "light-bulb"}

#         for node, node_attr in graph.nodes(data=True):
#             xy.append(node_attr["pos_map"][:2])
#             node_types.append(node_attr["node_type"])
# ###################################################################################################
#             # # 如果当前节点有类别名称，并且不包含特定关键字，则添加文本标签
#             # if ("semantic_class_name" in node_attr and 
#             #     node_attr["semantic_class_name"] and 
#             #     not any(keyword in node_attr["semantic_class_name"].lower() for keyword in ["wall", "window", "picture", "door"])):
                
#             #     text_x = node_attr["pos_map"][1] + 0.05  # 向右移动一些
#             #     text_y = node_attr["pos_map"][0] + 0.05  # 向上移动一些
#             #     label_text = f"{node_attr['semantic_class_name']}"
#             #     ax.text(text_x, text_y, label_text, fontsize=8, ha='left', va='bottom')
            
#             # 如果节点有 closest_vor_node 属性，则用蓝色标记该位置并连接到物品节点
#             if "closest_vor_node" in node_attr:
#                 item_x, item_y = node_attr["pos_map"][:2]
#                 vor_x, vor_y = node_attr["closest_vor_node"]
#                 ax.scatter(vor_y, vor_x, c='b', s=10)  # 使用蓝色标记维诺图节点
                
#                 # 连接物品节点和其最近的维诺图节点
#                 ax.plot([item_y, vor_y], [item_x, vor_x], c='blue', linewidth=0.5)

#             # 检查节点是否属于指定的对象类别集合，并且具有 closest_vor_node 属性
#             # if ("semantic_class_name" in node_attr and 
#             #     node_attr["semantic_class_name"] in special_object_categories and 
#             #     "closest_vor_node" in node_attr):

#             #     item_x, item_y = node_attr["pos_map"][:2]
#             #     vor_x, vor_y = node_attr["closest_vor_node"]
#             #     ax.scatter(vor_y, vor_x, c='b', s=10)  # 使用蓝色标记维诺图节点

#             #     # 连接物品节点和其最近的维诺图节点
#             #     ax.plot([item_y, vor_y], [item_x, vor_x], c='blue', linewidth=0.5)
#             #     special_instances_count += 1
# ##################################################################################################



#         xy = np.stack(xy)
        
#         colors = ["c", "k", "m", "r", "g"]
#         sizes = [3, 3, 30, 3, 3]
#         for nt in np.unique(node_types):
#             if nt == NODETYPE.ROOT:
#                 continue
#             ax.scatter(xy[node_types == nt, 1], xy[node_types == nt, 0], c=colors[nt], s=sizes[nt])
        
#     for edge in graph.edges():
#         if "root" in edge:
#             continue
#         if isinstance(edge[0], tuple):
#             x1, y1 = edge[0]
#             x2, y2 = edge[1]
#         else:
#             x1, y1 = graph.nodes[edge[0]]['pos_map'][:2]
#             x2, y2 = graph.nodes[edge[1]]['pos_map'][:2]
#         if edge_filter is not None and edge in edge_filter:
#             ax.plot([y1, y2], [x1, x2], c='r', linewidth=0.5)
#         else:
#             ax.plot([y1, y2], [x1, x2], c='w', linewidth=0.5)

#     # 不需要立即显示图像，因为可能是在更大的绘图上下文中使用此函数
#     # plt.show() 
#     print(f"符合条件的实例数量: {special_instances_count}")
# # 注意：确保在调用plot_graph时传递正确的参数
# # 示例：
# # plot_graph(room_object_graph, self.slam.bev_map_occupancy, ax=self.ax[1], bounds=bounds)

def plot_graph(graph, map_underlay, edge_filter=None, ax=None, bounds=None, node_colors=None):
    if ax is None:
        ax = plt.gca()
    ax.clear()

    if bounds is None:
        min_x, max_x, min_y, max_y = 0, map_underlay.shape[0], 0, map_underlay.shape[1]
    else:
        min_x, max_x, min_y, max_y = bounds
    ax.set_xlim(min_y, max_y)
    ax.set_ylim(max_x, min_x)
    ax.imshow(map_underlay)
    
    if graph is None or len(graph.nodes()) == 0:
        return
    
    # Step 1: 读取 area.txt 文件，解析物品的区域信息
    area_info = {}
    try:
        with open("area.txt", "r") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    parts = line.strip().split()
                    if len(parts) >= 4:  # 确保格式是 name='xxx' room-X Area Y
                        name = parts[0].split('=')[1].strip("'")  # 提取 name='xxx' → xxx
                        room = int(parts[1].split('-')[1])        # 提取 room-1 → 1
                        area = int(parts[3])                      # 提取 Area 2 → 2（注意是 parts[3]）
                        area_info[name] = {"room": room, "area": area}
        # print(f"成功读取 {len(area_info)} 个物品的区域信息")
        # print("示例数据:", list(area_info.items())[:3])  # 打印前3个物品的区域信息
    except FileNotFoundError:
        # print("Warning: area.txt not found. No area-based coloring will be applied.")
        area_info = {}

    # Step 2: 为不同的 (room, area) 组合分配颜色
    room_area_colors = {}
    color_palette = plt.cm.get_cmap('tab20', 20)  # 使用20种颜色，不够时循环
    # print("\n房间-区域颜色分配情况:")

    if isinstance(list(graph.nodes())[0], tuple):
        # 当节点是 tuple 类型时（如点云数据）
        xy = np.array(graph.nodes())
        if node_colors is not None:
            ax.scatter(xy[:, 1], xy[:, 0], c=node_colors, s=5)
        else:
            ax.scatter(xy[:, 1], xy[:, 0], c='r', s=3)
    else:
        xy = []
        node_types = []
        special_instances_count = 0

        for node, node_attr in graph.nodes(data=True):
            # 1. 区域信息检查逻辑（保持不变）
            has_area_info = False
            room_area_key = None
            
            if "associated_items" in node_attr:  # 共享节点处理
                for item in node_attr["associated_items"]:
                    if item in area_info:
                        has_area_info = True
                        room = area_info[item]["room"]
                        area = area_info[item]["area"]
                        room_area_key = (room, area)
                        break
            elif "name" in node_attr and node_attr["name"] in area_info:  # 普通节点
                has_area_info = True
                room = area_info[node_attr["name"]]["room"]
                area = area_info[node_attr["name"]]["area"]
                room_area_key = (room, area)
            
            # 2. 只处理有区域信息的节点
            if has_area_info:
                # 分配颜色（保持不变）
                if room_area_key not in room_area_colors:
                    color = color_palette(len(room_area_colors) % 20)
                    room_area_colors[room_area_key] = color
                    # print(f"分配颜色 -> 房间 {room_area_key[0]} 区域 {room_area_key[1]}: RGB {color[:3]}")
                
                # 3. 修改点：对物品节点着色（不再是维诺图节点）
                item_x, item_y = node_attr["pos_map"][:2]
                ax.scatter(item_y, item_x, c=[room_area_colors[room_area_key]], s=30)  # 增大物品节点尺寸

                circle_radius = 30  # Adjust this value as needed
                circle = plt.Circle((item_y, item_x), circle_radius, 
                color=room_area_colors[room_area_key],
                fill=False, linewidth=2, alpha=0.7)
                ax.add_patch(circle)
                
                # 4. 保持原有连接线绘制（可选是否着色）
                if "closest_vor_node" in node_attr:
                    vor_x, vor_y = node_attr["closest_vor_node"]
                    # 连接线使用相同颜色（或固定颜色如灰色）
                    ax.plot([item_y, vor_y], [item_x, vor_x], 
                        c=room_area_colors[room_area_key],  # 与物品同色
                        linewidth=0.5)
                    
                    # 维诺图节点可设为固定颜色（如白色）或隐藏
                    ax.scatter(vor_y, vor_x, c='w', s=5)  # 白色小点
                
        # 绘制其他节点（如ROOT、WALL等）
        colors = ["c", "k", "m", "r", "g"]
        sizes = [3, 3, 30, 3, 3]
        for nt in np.unique(node_types):
            if nt == NODETYPE.ROOT:
                continue
            ax.scatter(xy[node_types == nt, 1], xy[node_types == nt, 0], 
                      c=colors[nt], s=sizes[nt])
        
    # # 绘制边（edges）
    # for edge in graph.edges():
    #     if "root" in edge:
    #         continue
    #     if isinstance(edge[0], tuple):
    #         x1, y1 = edge[0]
    #         x2, y2 = edge[1]
    #     else:
    #         x1, y1 = graph.nodes[edge[0]]['pos_map'][:2]
    #         x2, y2 = graph.nodes[edge[1]]['pos_map'][:2]
    #     if edge_filter is not None and edge in edge_filter:
    #         ax.plot([y1, y2], [x1, x2], c='r', linewidth=0.5)
    #     else:
    #         ax.plot([y1, y2], [x1, x2], c='w', linewidth=0.5)
