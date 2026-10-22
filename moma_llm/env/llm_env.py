# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
from igibson import object_states
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import distance_matrix

from moma_llm.env.env import OurIGibsonEnv
from moma_llm.env.high_level_env import HighLevelEnv
from moma_llm.llm.llm import LLM, Conversation, inflect_engine
from moma_llm.topology.room_graph import get_closest_node
from moma_llm.utils.constants import FRONTIER_CLASSIFICATION, NODETYPE
from moma_llm.llm.sbert import SentenceBERT
from moma_llm.navigation.navigation import (PyAstarHelper, find_floor_idx,
                                             get_circular_kernel,
                                             plan_waypoints)
from moma_llm.utils.constants import (BLOCKING_DOORS, CLASS_NAME_TO_CLASS_ID,
                                       EXTERIOR_DOORS, OCCUPANCY)
import pyastar
DIST_MAPPING = OrderedDict({
    3.0: "very close",
    10.0: "near",
    20.0: "far",
    np.inf: "distant"})


def distance_mapping(dist: float):
    for k, v in DIST_MAPPING.items():
        if dist < k:
            return v
    
    
def split_frontier_points(frontier_points):
    frontier_points_within, frontier_points_leading_out = [], []
    for frontier_point in frontier_points:
        if frontier_point[1] == FRONTIER_CLASSIFICATION.WITHIN:
            frontier_points_within.append(frontier_point[0])
        elif frontier_point[1] == FRONTIER_CLASSIFICATION.LEADING_OUT:
            frontier_points_leading_out.append(frontier_point[0])
        else:
            raise ValueError(f"Unknown frontier point type {frontier_point[1]}")
    return frontier_points_within, frontier_points_leading_out

@dataclass
class ActionHistory:
    action: str
    object_name_graph: str
    position: tuple
    subtask_success: bool
    opendoors_roompos: Any = None
    orig_api_call: str = None
    

class LLMEnv(HighLevelEnv):
    def __init__(self, env: OurIGibsonEnv, llm: LLM, seed: int) -> None:
        super().__init__(env, seed=seed)

        # action: (argument, description)
        self.possible_actions = {
            # "navigate": ("room_name, object_name", "navigate to this object in this room."),
            # "go_to_and_open": ("room_name, object_name", "go to this articulated object, door or container and open it."),
            "go_to_and_open": ("room_name, door", "go to this door and open it."),
            # "close": ("room_name, object_name", "close this articulated object, door or container."),
            "explore": ("room_name", "explore the unknown space near one of the rooms that is not fully explored yet."),
            "check":("room_name, area", "go to this area where the target object may appear and open all the container in the area"), ###
            # "recheck": ("room_name, area", "go to this area and pen the container in the area where the target object may appear "),##
            "done": ("", "call when the task is completed or if you are unable to take any further actions.")
        }
        if not self.env.config["consider_open_actions"]:
            del self.possible_actions["go_to_and_open"]
            del self.possible_actions["close"]
        
        self.llm = llm
        self.room_classification = dict()

    def reset(self, config_file, scene_id, episode_num):
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "env", "content": ""}
        return super().reset(config_file, scene_id, episode_num, compute_scene_graph=True)
    
    

    def classify_rooms(self, obs):
        room_classification: dict = self.llm.classify_rooms(obs)
        # nx can't handle duplicate node names, so add a number to the end if there are multiple instances of a room type
        tot_counter = Counter(room_classification.values())
        counter = defaultdict(int)
        for k, v in room_classification.items():
            if tot_counter[v] > 1:
                room_classification[k] = v + f"-{counter[v] + 1}"
                counter[v] += 1
        self.room_classification = room_classification
        self._label_rooms_on_map(obs)

    # def evaluate_room_labeling(self, obs):
    #     # 辅助函数：统一标签格式（去除下划线）
    #     def normalize_label(label):
    #         return label.replace('_', ' ') if label else "unknown"

    #     # 1. 获取分割后的Voronoi图连通组件
    #     separated_components = [obs['separated_voronoi_graph'].subgraph(c).copy() 
    #                         for c in nx.connected_components(obs['separated_voronoi_graph'])]
        
    #     components = {}
    #     c2room_gt_sem = defaultdict(list)
    #     c2room_pred_sem = defaultdict(list)

    #     print("\n===== 开始评估房间分割和分类 =====")
        
    #     # 2. 遍历每个连通组件
    #     for pred_c_idx, c in enumerate(separated_components):
    #         components[pred_c_idx] = c
    #         print(f"\n连通组件 {pred_c_idx}（包含 {len(c.nodes)} 个节点）:")
            
    #         # 3. 处理每个节点
    #         for n in c.nodes:
    #             world_coords = self.unwrapped.slam.voxel2world(n)
    #             seg_map_coords = self.unwrapped.scene.world_to_seg_map(world_coords)
    #             room_name = self.unwrapped.scene.get_room_type_by_point(world_coords)
                
    #             # 统一标签格式：真实标签去除下划线
    #             true_label = normalize_label(room_name)
    #             # 预测标签直接从分类结果获取（假设已经是空格格式）
    #             pred_label = self.room_classification[NODETYPE.roomname(c.nodes[n]["room_id"])]
                
    #             c2room_gt_sem[pred_c_idx].append(true_label)
    #             c2room_pred_sem[pred_c_idx].append(pred_label)
                
    #             print(f"  节点 {n}: 预测='{pred_label}', 真实='{true_label}'")

    #     # 4. 计算分类准确性
    #     pred_labels = []
    #     true_labels = []
    #     for c_idx in components.keys():
    #         pred_labels.extend(c2room_pred_sem[c_idx])
    #         true_labels.extend(c2room_gt_sem[c_idx])
        
    #     correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    #     semantic_accuracy = correct / len(true_labels) if true_labels else 0
    #     self.episode_room_sem_acc.append(semantic_accuracy)

    #     # 5. 计算纯度
    #     print("\n===== 纯度计算过程 =====")
    #     purity_scores = []
        
    #     for c_idx in components.keys():
    #         gt_labels = c2room_gt_sem[c_idx]
    #         if not gt_labels:
    #             continue
            
    #         label_counts = {}
    #         for label in gt_labels:
    #             label_counts[label] = label_counts.get(label, 0) + 1
            
    #         dominant_label, dominant_count = max(label_counts.items(), key=lambda x: x[1])
    #         component_purity = dominant_count / len(gt_labels)
    #         purity_scores.append(component_purity)
            
    #         print(f"连通组件 {c_idx}:")
    #         print(f"  真实标签分布: {label_counts}")
    #         print(f"  主导标签: '{dominant_label}' (占比: {dominant_count}/{len(gt_labels)} = {component_purity:.2%})")

    #     overall_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0
    #     self.episode_room_purity.append(overall_purity)

    #     # 6. 打印结果
    #     print("\n===== 评估结果 =====")
    #     print(f"分类准确性: {semantic_accuracy:.2%}")
    #     print(f"分割纯度: {overall_purity:.2%}")

    #     return {
    #         "semantic_accuracy": semantic_accuracy,
    #         "purity": overall_purity
    #     }

    def evaluate_room_labeling(self, obs):
        # 辅助函数：统一标签格式
        def normalize_label(label):
            return label.replace('_', ' ') if label else "unknown"

        print("\n===== 开始评估房间分割和分类 =====")
        
        # 1. 改进的文件读取方法
        room_item_data = {}
        try:
            with open("room_seg.txt", "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    
                    # 分割房间和物品部分
                    room_part, items_part = line.strip().split(":", 1)
                    room_type = room_part.strip()
                    
                    # 使用正则表达式匹配所有物品及其坐标
                    import re
                    item_matches = re.finditer(r'(\w+):(\([^)]+\))', items_part)
                    
                    for match in item_matches:
                        item_name = match.group(1)
                        coord_str = match.group(2)
                        room_item_data[item_name] = {
                            "room_type": room_type,
                            "closest_vor_node": coord_str
                        }
        except Exception as e:
            print(f"文件读取错误: {str(e)}")
            return {"semantic_accuracy": 0, "purity": 0}

        pred_labels = []
        true_labels = []
        room_purity_data = defaultdict(lambda: {"count": 0, "true_labels": []})

        # 2. 改进的坐标解析
        for item_name, item_data in room_item_data.items():
            coord_str = item_data["closest_vor_node"]  # 格式如 "(201.5, 96.5)"
            pred_label = item_data["room_type"]
            
            try:
                # 安全解析坐标
                coord_str = coord_str.strip("()")
                x_str, y_str = [s.strip() for s in coord_str.split(",", 1)]
                x = float(x_str)
                y = float(y_str)
                vor_coords = (x, y)
                
                # 获取世界坐标
                world_coords = self.unwrapped.slam.voxel2world(vor_coords)
                true_label = normalize_label(self.unwrapped.scene.get_room_type_by_point(world_coords))
                
                pred_labels.append(pred_label)
                true_labels.append(true_label)
                room_purity_data[pred_label]["count"] += 1
                room_purity_data[pred_label]["true_labels"].append(true_label)
                
                print(f"成功解析: {item_name} | 坐标: {vor_coords} | 预测: {pred_label} | 真实: {true_label}")
                
            except Exception as e:
                print(f"坐标解析失败 | 物品: {item_name} | 原始坐标: {coord_str} | 错误: {str(e)}")
                continue

        # 3. 计算分类准确性
        correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
        semantic_accuracy = correct / len(true_labels) if true_labels else 0
        self.episode_room_sem_acc.append(semantic_accuracy)

        # 4. 计算纯度
        print("\n===== 纯度计算过程 =====")
        purity_scores = []
        
        for pred_room, data in room_purity_data.items():
            if not data["true_labels"]:
                continue
            
            label_counts = {}
            for label in data["true_labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            dominant_label, dominant_count = max(label_counts.items(), key=lambda x: x[1])
            component_purity = dominant_count / len(data["true_labels"])
            purity_scores.append(component_purity)
            
            print(f"预测房间 '{pred_room}':")
            print(f"  真实标签分布: {label_counts}")
            print(f"  主导标签: '{dominant_label}' (占比: {dominant_count}/{len(data['true_labels'])} = {component_purity:.2%})")

        overall_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0
        self.episode_room_purity.append(overall_purity)

        # 5. 打印结果
        print("\n===== 评估结果 =====")
        print(f"分类准确性: {semantic_accuracy:.2%}")
        print(f"分割纯度: {overall_purity:.2%}")

        return {
            "semantic_accuracy": semantic_accuracy,
            "purity": overall_purity
        }

    def _label_rooms_on_map(self, obs, ax_idx=1):
        # delete all existing text
        for txt in self.env.ax[ax_idx].texts:
            Artist.remove(txt)

        # put labels on the map
        labeled_rooms_not_found = list()
        for room, labelled_room in self.room_classification.items():
            if room in obs["room_object_graph"].nodes:
                room_node = obs["room_object_graph"].nodes[room]
                room_center = room_node["pos_map"]
                self.env.ax[ax_idx].text(room_center[1], room_center[0], labelled_room, fontsize=8, color="m", horizontalalignment='center')
            else:
                labeled_rooms_not_found.append((room, labelled_room))
        if len(labeled_rooms_not_found) > 0:
            print("Rooms that were labeled but not contained in the room object graph: ", labeled_rooms_not_found)

    def parse_llm_action(self, response: str):
        action = None
        command_found = False
        for line in response.split("\n"):
            # Go through answers until we find a "command", then answer may be on the same line or the next. 
            # Just go line by line until there is something in the form of an api call
            command_found = command_found or line.lower().startswith("command")
            if not command_found:
                continue
            try:
                action, argument = re.search(r"(\w+)\((.*)\)", line).groups()
                argument = argument.replace("'", "").replace('"', "")
                if not action:
                    print("Empty action parsed:", (action, line))
                break
            except:
                continue
            
        if not action or action not in self.possible_actions.keys():
            # TODO: better to re-prompt the LLM instead?
            print("Could not parse any command. Assuming this is because the LLM did not find a reasonable action. So calling done() instead.")
            action = "done"
            argument = ""
            self.episode_info["failure_reason"] = "Unable to prompt LLM command."
            
        if self.possible_actions[action][0]:
            assert argument, (argument, response)
        return action, argument

    # def _create_prompt(self, 
    #                    obs,
    #                    task_description, 
    #                    labelled_rooms, 
    #                    current_room, 
    #                    room_dict, 
    #                    rooms_with_frontier_within, 
    #                    rooms_with_frontier_leading_out, 
    #                    rooms_with_closed_doors, 
    #                    close_objects, 
    #                    nlp_history,
    #                    room_distances,
    #                    *args, 
    #                    **kwargs) -> Conversation:
    #     system_prompt = f"You are a robot in an unexplored house. Your task is to {task_description}."
    #     system_prompt += f" You have the following actions available that you can use to achieve this task:\n"
    #     for i, (action, description) in enumerate(self.possible_actions.items()):
    #         system_prompt += f"""{i+1}. {action}({description[0]}): {description[1]}\n"""
    #     system_prompt += f"\nOutput Response Format:\n"\
    #     "Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.\n"\
    #     "Reasoning: justify why the next action is important to solve the task.\n"\
    #     "Command: function call"

    #     prompt = f"You are currently in the {current_room}. You are standing next to the following objects: [{', '.join(sorted(close_objects))}]."
    #     prompt += f" Furthermore, you have found the following rooms and objects in the house so far:\n"
    #     for room in sorted(labelled_rooms):
    #         objects = room_dict[room] + (["unexplored area"] if room in [r[0] for r in rooms_with_frontier_within] else [])
    #         prompt += f"""- {room}: [{", ".join(objects)}].\n"""

    #     if len(nlp_history):
    #         prompt += f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}.\n"

    #     rooms_with_frontier_descr = [f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out]
    #     prompt += f"These rooms have unexplored space leading out of the room: [{', '.join(rooms_with_frontier_descr)}].\n"
    #     if len(rooms_with_closed_doors):
    #         rooms_with_closed_doors_descr = [f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors]
    #         prompt += f"These rooms contain closed doors that might open up new space: [{', '.join(rooms_with_closed_doors_descr)}].\n"

    #     prompt += f"""What is the best next action to complete the task as efficiently as possible? If you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.\n"""  # In general, prioritize opening doors if there is not enough evidence to explore a promising room.\n""" 
    #     prompt += f"Remember:\n"\
    #         "1. Respond with a function call.\n"\
    #         "2. You can only use the objects and rooms that you have already found. Object names have to match the description exactly.\n"\
    #         "3. You can only explore rooms that are listed as having unexplored space.\n"\
    #         "4. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.\n"\
    #         "5. If some actions failed repeatedly, they may not be possible.\n"

    #     conversation = Conversation(messages=[self.last_env_feedback,
    #                                           {"role": "system", "content": system_prompt},
    #                                           {"role": "user", "content": prompt}])
    #     return conversation

    def _create_prompt(self, 
                obs,
                task_description, 
                labelled_rooms, 
                current_room, 
                rooms_with_frontier_within, 
                rooms_with_frontier_leading_out, 
                rooms_with_closed_doors, 
                rooms_with_area_groups,###############LKY
                close_objects, 
                nlp_history,
                room_distances,
                *args, 
                **kwargs) -> Conversation:
        
        # 加载黑名单（完全保持不变）
        def _load_blacklist():
            try:
                with open("blacklist.txt", "r") as f:
                    return set(line.strip() for line in f if line.strip())
            except FileNotFoundError:
                return set()
        
        blacklist = _load_blacklist()

        # 系统提示修改部分（仅改动作列表生成逻辑）
        system_prompt = f"You are a robot in an unexplored house. Your task is to {task_description}."
        system_prompt += f" You have the following actions available that you can use to achieve this task:\n"
        
        # 初始动作列表（保持原有顺序和格式）
        action_list = [
            "1. explore(room_name): explore the unknown space near one of the rooms that is not fully explored yet.",
            "2. go_to_and_open(room_name, door): go to this door and open it.",
            "3. done(): call when the task is completed.",
            "4. check(room_name, area): open all containers in this area."
        ]

        # # 仅在存在未检查区域时添加check动作（新增逻辑）
        # graph = obs["room_object_graph"]
        # has_unchecked = any(
        #     container not in blacklist
        #     for room in graph.successors("root")
        #     for obj in graph.successors(room)
        #     for container in [graph.nodes[obj].get("name", "")]
        #     if graph.nodes[obj].get('role') == 'container' 
        #     and graph.nodes[obj].get('states')
        # )
        
        # if has_unchecked:
        #     action_list.append(
        #         "4. check(room_name, area): open all containers in this area."
        #     )

        system_prompt += "\n".join(action_list)  # 保持原有拼接方式
        system_prompt += f"\nOutput Response Format:\n"\
        "Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.\n"\
        "Reasoning: justify why the next action is important to solve the task.\n"\
        "Command: function call"

        # 以下全部保持原有代码不变 ▼▼▼
        prompt = f"You are currently in the room with {current_room}. You are standing next to the following objects: [{', '.join(sorted(close_objects))}]."
        prompt += f" Furthermore, you have found the following rooms and areas in the house so far:\n"
        
        graph = obs["room_object_graph"]
        
        with open("all_objects.txt", "w") as f:
            for node, attrs in graph.nodes(data=True):
                if attrs.get('node_type') not in [NODETYPE.ROOM, NODETYPE.ROOT]:
                    node_name = attrs.get("name", "")
                    f.write(node_name + "\n")

        labelled_rooms = list(graph.successors("root"))
        unchecked_areas = []
        formatted_areas = []

        for room_idx, room in enumerate(labelled_rooms):
            formatted_areas = []  # 每个房间开始时清空区域列表
            objects_in_room = sorted(list(graph.successors(room)))
            if not objects_in_room:
                continue

            areas_in_room = {}
            all_containers_in_room = []

            for obj in objects_in_room:
                attrs = graph.nodes[obj]
                area_name = attrs.get("area", "Unassigned Area")
                role = attrs.get("role", "").strip()
                name = attrs.get("name", "").strip()
                states = attrs.get("states", {})

                if role == "container" and states:
                    all_containers_in_room.append(name)
                    if area_name not in areas_in_room:
                        areas_in_room[area_name] = {
                            "containers": [],
                            "movable_items": [],
                            "other_items": [],  # New list for items with role "others"
                            "area_function": attrs.get("area_function", "Unknown function")
                        }
                    areas_in_room[area_name]["containers"].append(name)

            for obj in objects_in_room:
                attrs = graph.nodes[obj]
                area_name = attrs.get("area", "Unassigned Area")
                role = attrs.get("role", "").strip()
                name = attrs.get("name", "").strip()

                if role == "movable":
                    relations = attrs.get("relation", [])
                    for relation in relations:
                        if "on" in relation:
                            movable, container = relation.split(" on ")
                            movable = movable.strip()
                            container = container.strip().lower()

                            if area_name in areas_in_room:
                                if container in [c.lower() for c in areas_in_room[area_name]["containers"]]:
                                    areas_in_room[area_name]["movable_items"].append(movable)
                

            for obj in objects_in_room:
                attrs = graph.nodes[obj]
                area_name = attrs.get("area", "Unassigned Area")
                role = attrs.get("role", "").strip()
                name = attrs.get("name", "").strip()

                if role == "others":
                    if area_name in areas_in_room:
                        areas_in_room[area_name]["other_items"].append(name)

            prompt += f"- {room}: \n"
            # Removed the All Objects line as per request

            for area_name, area_data in areas_in_room.items():
                has_unchecked_containers = any(
                    container not in blacklist 
                    for container in area_data.get("containers", [])
                )
                
                if has_unchecked_containers:
                    unchecked_areas.append(f"({room}, {area_name})")
                    
                    containers = area_data.get("containers", [])
                    movable_items = list(set(area_data.get("movable_items", [])))
                    other_items = list(set(area_data.get("other_items", [])))  # Get unique other items
                    area_function = area_data.get("area_function", "Unknown function")

                    area_info = f"'{area_name}' (Function: {area_function})"
                    if containers:
                        area_info += f": Containers - [{', '.join(containers)}]"
                    
                    # Combine movable items and other items in the display
                    items_info = []
                    if movable_items:
                        items_info.append(f"Movable Objects - [{', '.join(movable_items)}]")
                    if other_items:
                        items_info.append(f"Others - [{', '.join(other_items)}]")
                    
                    if items_info:
                        area_info += "; " + "; ".join(items_info)
                        
                    formatted_areas.append(area_info)

            if room in [r[0] for r in rooms_with_frontier_within]:
                formatted_areas.append("Unexplored Area")

            if formatted_areas:
                prompt += f"  Areas: [\n    " + ",\n    ".join(formatted_areas) + "\n  ]\n"

        if len(nlp_history):
            prompt += f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}.\n"

        area_dist_dict = { 
            (area_group['room'], area_group['area']): area_group['distance']
            for area_group in rooms_with_area_groups
        }

        unchecked_areas_with_dist = []
        min_distances = {}

        for area_str in unchecked_areas:
            # 从字符串 "(room, area)" 中解析出房间和区域
            room_area_match = re.match(r"\((.*?),\s*(.*?)\)", area_str)
            if not room_area_match:
                continue
            
            room = room_area_match.group(1)
            area = room_area_match.group(2)
            
            # 直接查找距离信息（因为unchecked_areas已经确保是有效区域）
            dist = area_dist_dict.get((room, area))
            if dist is not None:
                # 如果找到距离信息，更新最小距离
                if area_str not in min_distances or dist < min_distances[area_str][0]:
                    min_distances[area_str] = (dist, room, area)
            else:
                # 如果没有距离信息，保持原样
                unchecked_areas_with_dist.append(area_str)

        # 添加具有最小距离的区域到结果列表
        for area_str, (dist, room, area) in min_distances.items():
            unchecked_areas_with_dist.append(f"{area_str}({distance_mapping(dist)})")


        prompt += f"These areas have closed objects that you can check: [{', '.join(unchecked_areas_with_dist)}].\n"


        rooms_with_frontier_descr = [f'{room}({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out]
        prompt += f"These rooms have unexplored space leading out of the room: [{', '.join(rooms_with_frontier_descr)}].\n"

        if rooms_with_frontier_within:
            rooms_within_descr = [f'{room}({distance_mapping(dist)})' for room, dist in rooms_with_frontier_within]
            prompt += f"These rooms have unexplored space within the room: [{', '.join(rooms_within_descr)}].\n"

        if len(rooms_with_closed_doors):
            rooms_with_closed_doors_descr = [f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors]
            prompt += f"These rooms contain closed doors that might open up new space: [{', '.join(rooms_with_closed_doors_descr)}].\n"
        else:
            prompt += f"There are no closed doors in the house.\n"    

        prompt += f"""What is the best next action to complete the task as efficiently as possible? If you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.\n"""  
        
        prompt += f"Remember:\n"\
            "1. Respond with a function call.\n"\
            "2. You can only use the objects and rooms that you have already found. Object names have to match the description exactly.\n"\
            "3. You can only explore rooms that are listed as having unexplored space.\n"\
            "4. You can only check areas that are listed as having closed objects,and can only check a certain area, not a container.\n"\
            "5. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.\n"\
            "6. If some actions failed repeatedly, they may not be possible.\n"\
            "7. Strictly follow my output format and do not output any additional symbols"
        
        conversation = Conversation(messages=[self.last_env_feedback,
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": prompt}])
        return conversation

    def _get_close_objects(self, graph, current_room, closeness_thresh: float):
        current_room_objects = list(graph.successors(current_room))
        if not current_room_objects:
            return []
        graph_node_pos = np.stack([graph.nodes.get(n)["pos_map"][:2] for n in current_room_objects])
        node_coords_world = self.unwrapped.slam.voxel2world(graph_node_pos)
        # [1, num_objects]
        dist_matrix = distance_matrix(self.unwrapped.robots[0].get_position()[np.newaxis, :2], node_coords_world)
        close_objects = np.array(current_room_objects)[np.squeeze(dist_matrix < closeness_thresh, 0)]
        return set([self.llm.to_human_readable_object_name(o) for o in close_objects])

    def _match_action_history(self, graph, separated_voronoi_graph):
        max_history_length = 5
        
        def _match_room_pos(position):
            if position is None:
                return None
            closest_nodes, _ = get_closest_node(np.array([position]), separated_voronoi_graph, self.slam)
            room_name = separated_voronoi_graph.nodes[tuple(closest_nodes[0])]["room_id"]
            return room_name
        
        nlp_history = []
        last_k = len(self.action_history) - max_history_length
        
        for i, h in enumerate(self.action_history):
            room_name, object_name, invalid = None, None, False
            
            if h.object_name_graph is not None:
                object_name = self.llm.to_human_readable_object_name(h.object_name_graph)
                if (h.opendoors_roompos is not None) and (h.object_name_graph in self.env.opened_doors):
                    room_name = _match_room_pos(h.opendoors_roompos)
                elif h.object_name_graph in graph.nodes:
                    room_name = list(graph.predecessors(h.object_name_graph))[0]
                else:
                    room_name = _match_room_pos(h.position)
            elif h.position is not None:
                room_name = _match_room_pos(h.position)
            
            nlp = None
            if h.action == "done":
                nlp = "done()"
            elif (h.action == "explore") and (room_name is not None): 
                nlp = f"explore({room_name})"
            elif (object_name is not None) and (room_name is not None):
                nlp = f"{h.action}({room_name}, {object_name})"
            # else:
            #     nlp = f"{h.orig_api_call} - invalid argument"
            #     invalid = True
            
            # if not invalid:
            #     nlp += f" - {'success' if h.subtask_success else 'failure'}"

            else:
                nlp = f"{h.orig_api_call}"
                # invalid = True
            
            if not invalid:
                nlp += f" - {'success' if h.subtask_success else 'failure'}"
                
            
            # keep history of last k actions & all invalid calls (e.g. trying to open an object that can't be opened)
            if i >= last_k or invalid:
                nlp_history.append(nlp)
        return nlp_history
    
    def send_query(self, conversation: str):
        response = self.llm.send_query(conversation=conversation)
        action, argument = self.parse_llm_action(response)
        return response, action, argument

    def take_action(self, obs: dict, task_description: str):
        def _apply_room_classification(obs):
  
            obs["room_object_graph"] = nx.relabel_nodes(obs["room_object_graph"], self.room_classification)
            for n, d in obs["separated_voronoi_graph"].nodes(data=True):
                d["room_id"] = self.room_classification[NODETYPE.roomname(d["room_id"])]
        self.classify_rooms(obs)

        parsed_relationships = self.llm.determine_object_relationships(obs) #########lky############
        self.llm.update_object_attributes(parsed_relationships, obs)###################################### 更新物品的属性
        obs = self.llm.determine_area_functions(obs)#################################################### 确定场景容器组
        # self.llm.determine_room_functions(obs)

        _apply_room_classification(obs)

        graph = obs["room_object_graph"]
        labelled_rooms = list(graph.successors("root"))

        room_dict = self.llm.create_room_object_dict(graph,
                                                     open_door_inclusion="ignore",
                                                     room_classification=self.room_classification)
        current_room = self.room_classification[obs["robot_current_room"]]

        def _get_closest_dist(points):#########算距离
            closest_idx, _closest_frontier, _costs, paths = self._find_closest_point(points)
            return self.env.slam.voxel_size * len(paths[closest_idx])

#####################################################################
        # 新增：建立房间-区域-物品的字典
        room_area_objects = {}

        for room_idx, room in enumerate(labelled_rooms):
            objects_in_room = sorted(list(graph.successors(room)))
            # 初始化房间的区域字典
            room_area_objects[room] = {}
            
            # 按区域分组物品
            for obj in objects_in_room:
                obj_data = graph.nodes.get(obj, {})
                if "area" in obj_data:
                    area = obj_data["area"]
                    if area not in room_area_objects[room]:
                        room_area_objects[room][area] = []
                    room_area_objects[room][area].append(obj)

        # 新增：计算每个区域物品组的最近距离
        rooms_with_area_groups = []

        for room in labelled_rooms:
            if room in room_area_objects:  # 确保房间有区域分组数据
                for area, objects in room_area_objects[room].items():
                    if objects:  # 确保区域内有物品
                        # 获取区域内所有物品的位置信息
                        object_nodes = [graph.nodes[obj] for obj in objects]
                        
                        # 计算最近距离
                        closest_dist = _get_closest_dist(object_nodes)
                        
                        # 添加到列表中
                        rooms_with_area_groups.append({
                            'room': room,
                            'area': area,
                            'distance': closest_dist,
                            'objects': objects
                        })

        # 按距离排序区域组
        rooms_with_area_groups = sorted(rooms_with_area_groups, key=lambda x: x['distance'])
        print(rooms_with_area_groups)



        rooms_with_frontier_within = []
        rooms_with_frontier_leading_out = []
        rooms_with_closed_doors = []

        for n in labelled_rooms:
            frontier_points = graph.nodes.get(n)["frontier_points"]
            if len(frontier_points):
                frontier_points = graph.nodes[n].get("frontier_points", {})
                frontier_points_within, frontier_points_leading_out = split_frontier_points(frontier_points)
                
                if frontier_points_within:
                    rooms_with_frontier_within.append((n, _get_closest_dist(frontier_points_within))) 
                if frontier_points_leading_out:
                    rooms_with_frontier_leading_out.append((n, _get_closest_dist(frontier_points_leading_out)))
            closed_doors = graph.nodes.get(n)["closed_doors"]
            if len(closed_doors):
                rooms_with_closed_doors.append((n, _get_closest_dist([graph.nodes.get(n) for n in closed_doors])))
        rooms_with_frontier_within = sorted(rooms_with_frontier_within, key=lambda x: x[1])
        rooms_with_frontier_leading_out = sorted(rooms_with_frontier_leading_out, key=lambda x: x[1])
        rooms_with_closed_doors = sorted(rooms_with_closed_doors, key=lambda x: x[1])
        close_objects = self._get_close_objects(graph=graph, current_room=current_room, closeness_thresh=2.5)
        nlp_history = self._match_action_history(graph=graph, separated_voronoi_graph=obs["separated_voronoi_graph"])
        
        def _calc_dist_to_room(current_room, separated_voronoi_graph) -> dict:
            # find distance from robot to closest voronoi node with that room label
            vnodes = defaultdict(list)
            for node_pos, node_data in separated_voronoi_graph.nodes(data=True):
                vnodes[node_data["room_id"]].append(node_pos)
            
            room_distances = {}
            for room in self.room_classification.values():
                if room == current_room:
                    room_distances[room] = "current location"
                else:
                    _idx, _closest_node, _costs, paths = self._find_closest_point([self.env.slam.voxel2world(pos) for pos in vnodes[room]])
                    dist = self.env.slam.voxel_size * len(paths)
                    room_distances[room] = distance_mapping(dist)
            return room_distances
        room_distances = _calc_dist_to_room(current_room, obs["separated_voronoi_graph"])


        conversation = self._create_prompt(obs = obs,
                                           task_description=task_description,
                                           labelled_rooms=labelled_rooms,
                                           current_room=current_room,
                                           room_dict=room_dict,
                                           rooms_with_frontier_within=rooms_with_frontier_within,
                                           rooms_with_frontier_leading_out=rooms_with_frontier_leading_out,
                                           rooms_with_closed_doors=rooms_with_closed_doors,
                                           rooms_with_area_groups=rooms_with_area_groups,####################LKY
                                           close_objects=close_objects,
                                           nlp_history=nlp_history,
                                           graph=graph,
                                           room_graph=obs["room_graph"],
                                           room_distances=room_distances)
        response, action, argument = self.send_query(conversation=conversation)
        
        print(f"argument is {argument}#################")

        robot_pose_pre = np.concatenate((self.env.robots[0].get_position_orientation()))
        subpolicy_success, done, self.last_env_feedback = self.execute_action(action=action,
                                                                              argument=argument,
                                                                              task_desc=task_description,
                                                                              graph=graph,
                                                                              vor_graph=obs["separated_voronoi_graph"],)
        print(f"subpolicy_success is {subpolicy_success}######################################")
        
        

        conversation.add_message(self.last_env_feedback)
        # self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])  ##LKY 不画图
        
        robot_pose_post = np.concatenate((self.env.robots[0].get_position_orientation()))

        num_retries, max_retries = 0, 5
        # only re-try if robot pose didn't change. Otherwise do a normal next high-level step with the new observation
        while (not subpolicy_success) and np.all((robot_pose_post - robot_pose_pre) < 0.1) and (not done) and (num_retries < max_retries):
            # recompute obs so that num_high_level_steps counter is correctly increased
            obs = self.env.get_state(compute_scene_graph=True)
            try:
                _apply_room_classification(obs)
            except:
                break
            retrial_prompt = f"The last action {action}({argument}) failed. Please try another command."
            conversation.add_message({"role": "user", "content": retrial_prompt})
            response, action, argument = self.send_query(conversation=conversation)

            

            subpolicy_success, done, self.last_env_feedback = self.execute_action(action=action,
                                                                                  argument=argument,
                                                                                  task_desc=task_description,
                                                                                  graph=graph,
                                                                                  vor_graph=obs["separated_voronoi_graph"])
            #######################

            conversation.add_message(self.last_env_feedback)
            # self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
            num_retries += 1
        if (num_retries == max_retries) and (not subpolicy_success) and (not done):
            done = True
            self.episode_info["failure_reason"] = "max retrials reached"
        self.episode_info["total_num_retrials"] += num_retries
        self.episode_info["steps_with_retrial"] += (num_retries > 0)

        if done:
            task_success = self.evaluate_success()
        else:
            task_success = False

        if sum([response == r for r in self.prev_responses]) >= 3:
            print("WARNING: LLM response is the same as in the previous steps. Probably stuck.")
            if not self.episode_info.get("failure_reason", None):
                self.episode_info["failure_reason"] = "llm stuck"
                self.episode_info["task_success"] = False
            done = True
        if len(self.prev_responses) > 6:
            del self.prev_responses[0]
        self.prev_responses.append(response)

        return done, task_success, self.episode_info

#####################################################################################
    # def open_all_containers_in_area(self, graph, vor_graph, argument):  
    #     """
    #     打开指定区域（area）中所有具有 states 且可以操作的容器。
    #     使用 Voronoi 图选择接近点来尝试打开每个容器。

    #     参数:
    #         graph: 图结构，包含节点和容器信息。
    #         vor_graph: Voronoi 图，用于路径规划。
    #         argument: 包含 area 的字符串，例如 "Area 1".

    #     返回:
    #         subtask_success: 布尔值，表示是否成功完成子任务。
    #         last_container_pos: 最后一个成功打开的容器位置。
    #         feedback: 操作的反馈信息。
    #     """

    #     def _get_possible_containers_in_area(area_name):
    #         """
    #         获取指定区域内所有可以被打开的容器节点。
    #         """
    #         possible_containers = []
    #         for node_name, node_info in graph.nodes.items():
    #             if node_info.get("area", None) == area_name:
    #                 # 确保节点有 states 属性并且当前是关闭状态
    #                 if "states" in node_info:
    #                     is_closed = not node_info["states"].get(object_states.Open, True)
    #                     if is_closed:
    #                         possible_containers.append((node_name, node_info))
    #         return possible_containers

    #     try:
    #         # 解析 area 名称：去除首尾空格、标准化格式（可选）
    #         area_name = argument.strip()

    #         # 获取该区域中所有可打开的容器
    #         containers = _get_possible_containers_in_area(area_name)

    #         if not containers:
    #             feedback = f"No closable containers found in area '{area_name}'."
    #             return False, None, feedback

    #         # 遍历并尝试打开每个容器
    #         opened_containers = []
    #         failed_containers = []
    #         last_container_pos = None

    #         for node_name, node_info in containers:
    #             obj = self.scene.objects_by_name.get(node_name, None)
    #             print(f"Trying to open container '{node_name}'.")

    #             if obj:
    #                 # --- 关键部分：模仿 open_graph_node 中的 Voronoi 路径选择 ---
    #                 pos = node_info.get("pos_map", [0, 0, 0])[:2]  # 只使用 x, y 坐标
    #                 if pos is None:
    #                     failed_containers.append((node_name, "No position information available."))
    #                     continue

    #                 vor_nodes = np.stack(vor_graph.nodes)
    #                 dists_to_container = np.linalg.norm(np.array(pos) - vor_nodes, axis=1)
    #                 is_close_to_container = dists_to_container <= max(2.0 / self.slam.voxel_size, dists_to_container.min())
    #                 points = [tuple(i) for i in self.env.slam.voxel2world(vor_nodes[is_close_to_container])]

    #                 # 找到最接近机器人的点
    #                 idx, point, _costs, _paths = self._find_closest_point(points, euclidean_heuristic=True)

    #                 # 尝试打开对象
    #                 open_success, feedback = self.open_object(obj, point)

    #                 if open_success:
    #                     opened_containers.append(node_name)
    #                     last_container_pos = point  # 更新为实际使用的路径点
    #                     print(f"Container '{node_name}' has been successfully opened.")
    #                 else:
    #                     failed_containers.append((node_name, feedback))
    #             else:
    #                 failed_containers.append((node_name, "Object not found in scene."))

    #         # 构造反馈信息
    #         feedback_messages = []
    #         if opened_containers:
    #             feedback_messages.append(f"Successfully opened containers: {', '.join(opened_containers)}.")
    #         if failed_containers:
    #             feedback_messages.append(f"Failed to open containers: {', '.join([f'{c}: {msg}' for c, msg in failed_containers])}.")

    #         # 判断子任务是否成功
    #         subtask_success = len(failed_containers) == 0

    #         return subtask_success, last_container_pos, " ".join(feedback_messages)

    #     except Exception as e:
    #         return False, None, f"An error occurred: {str(e)}"



    # def open_all_containers_in_area(self, graph, vor_graph, argument):
    #     """
    #     打开指定区域（area）中所有具有 states 且可以操作的容器。
    #     使用 Voronoi 图选择接近点来尝试打开每个容器，并根据容器位置计算一条最短路径顺序。

    #     参数:
    #         graph: 图结构，包含节点和容器信息。
    #         vor_graph: Voronoi 图，用于路径规划。
    #         argument: 包含 area 的字符串，例如 "Area 1".

    #     返回:
    #         subtask_success: 布尔值，表示是否成功完成子任务。
    #         last_container_pos: 最后一个成功打开的容器位置。
    #         feedback: 操作的反馈信息。
    #     """

    #     def _get_possible_containers_in_area(area_name):
    #         """
    #         获取指定区域内所有可以被打开的容器节点。
    #         """
    #         possible_containers = []
    #         for node_name, node_info in graph.nodes.items():
    #             if node_info.get("area", None) == area_name:
    #                 # 确保节点有 states 属性并且当前是关闭状态
    #                 if "states" in node_info:
    #                     is_closed = not node_info["states"].get(object_states.Open, True)
    #                     if is_closed:
    #                         possible_containers.append((node_name, node_info))
    #         return possible_containers

    #     def _compute_optimal_order(container_positions):
    #         """
    #         给定一组坐标，使用 TSP 计算访问这些坐标的最短路径顺序（索引列表）。
    #         container_positions: list of (x, y) positions.
    #         返回：最优访问顺序的索引列表。
    #         """
    #         if len(container_positions) <= 1:
    #             return list(range(len(container_positions)))

    #         G = nx.Graph()
    #         n = len(container_positions)

    #         # 构建距离矩阵
    #         for i in range(n):
    #             for j in range(i + 1, n):
    #                 dist = np.linalg.norm(np.array(container_positions[i]) - np.array(container_positions[j]))
    #                 G.add_edge(i, j, weight=dist)

    #         try:
    #             tsp_route = nx.approximation.traveling_salesman_problem(G, cycle=False)
    #             return tsp_route
    #         except Exception as e:
    #             print(f"TSP error: {e}, falling back to original order.")
    #             return list(range(len(container_positions)))

    #     try:
    #         area_name = argument.strip()

    #         # 获取该区域中所有可打开的容器
    #         containers = _get_possible_containers_in_area(area_name)

    #         if not containers:
    #             feedback = f"No closable containers found in area '{area_name}'."
    #             return False, None, feedback

    #         # 获取所有容器的位置
    #         containers_positions = [tuple(node_info.get("pos_map", [0, 0, 0])[:2]) for node_name, node_info in containers]

    #         # 计算最优访问顺序
    #         optimal_indices = _compute_optimal_order(containers_positions)

    #         # 按照最优顺序重新排列容器
    #         ordered_containers = [containers[i] for i in optimal_indices]
    #         ordered_positions = [containers_positions[i] for i in optimal_indices]

    #         opened_containers = []
    #         failed_containers = []
    #         last_container_pos = None

    #         # 遍历并尝试打开每个容器（按最优顺序）
    #         for idx, ((node_name, node_info), pos) in enumerate(zip(ordered_containers, ordered_positions)):
    #             obj = self.scene.objects_by_name.get(node_name, None)
    #             print(f"Trying to open container '{node_name}'.")

    #             if obj:
    #                 # --- 关键部分：模仿 open_graph_node 中的 Voronoi 路径选择 ---
    #                 pos_container = node_info.get("pos_map", [0, 0, 0])[:2]  # 只使用 x, y 坐标
    #                 if pos_container is None:
    #                     failed_containers.append((node_name, "No position information available."))
    #                     continue

    #                 vor_nodes = np.stack(vor_graph.nodes)
    #                 dists_to_container = np.linalg.norm(np.array(pos_container) - vor_nodes, axis=1)
    #                 is_close_to_container = dists_to_container <= max(2.0 / self.slam.voxel_size, dists_to_container.min())
    #                 points = [tuple(i) for i in self.env.slam.voxel2world(vor_nodes[is_close_to_container])]

    #                 # 找到最接近机器人的点
    #                 idx_closest, point_closest, _costs, _paths = self._find_closest_point(points, euclidean_heuristic=True)

    #                 # 尝试打开对象
    #                 open_success, feedback_open = self.open_object(obj, point_closest)

    #                 if open_success:
    #                     opened_containers.append(node_name)
    #                     last_container_pos = point_closest  # 更新为实际使用的路径点
    #                     print(f"Container '{node_name}' has been successfully opened.")
    #                 else:
    #                     failed_containers.append((node_name, feedback_open))
    #             else:
    #                 failed_containers.append((node_name, "Object not found in scene."))

    #         # 构造反馈信息
    #         feedback_messages = []
    #         if opened_containers:
    #             feedback_messages.append(f"Successfully opened containers in optimal order: {', '.join(opened_containers)}.")
    #         if failed_containers:
    #             feedback_messages.append(f"Failed to open containers: {', '.join([f'{c}: {msg}' for c, msg in failed_containers])}.")

    #         # 判断子任务是否成功
    #         subtask_success = len(failed_containers) == 0

    #         return subtask_success, last_container_pos, " ".join(feedback_messages)

    #     except Exception as e:
    #         return False, None, f"An error occurred: {str(e)}"
    def open_all_containers_in_area(self, graph, vor_graph, argument):
        """
        打开指定房间和区域中所有可操作的容器（格式：room_name,area_name）
        使用贪心算法最小化期望搜索成本 E[S] 优化访问顺序
        使用A*算法计算路径距离
        """

        def _get_path_distance(start_pos, end_pos):
            """使用A*算法计算两点间的路径距离"""
            try:
                # 直接使用第0层的地图
                m = self.env.scene.floor_map[0]  # 固定使用第0层
                
                # 创建占用地图
                occupancy_map = np.full_like(m, OCCUPANCY.FREE)
                occupancy_map[m == 0] = OCCUPANCY.OCCUPIED
                
                # 转换为路径规划权重
                weights = PyAstarHelper.map_to_weights(
                    occupancy_map,
                    inflation_radius_m=self.env.config["navigation_inflation_radius"],
                    resolution=self.env.slam.voxel_size,
                    add_wall_avoidance_cost=False
                )
                
                # 转换坐标到地图像素
                source_map = self.env.scene.world_to_map(start_pos[:2])
                target_map = self.env.scene.world_to_map(end_pos[:2])
                
                if weights[source_map[0], source_map[1]] == OCCUPANCY.OCCUPIED:
                    raise ValueError("起点位于障碍物内！")
                if weights[target_map[0], target_map[1]] == OCCUPANCY.OCCUPIED:
                    raise ValueError("终点位于障碍物内！")
                
                # 计算A*路径
                path_pixels = pyastar.astar_path(weights, source_map, target_map, allow_diagonal=True, costfn='linf')
                # print(f"path_pixels: {path_pixels}")
                if path_pixels is None:
                    return float('inf')  # 不可达
                
                # 计算路径长度
                path_world = self.env.scene.map_to_world(path_pixels)
                return np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
                
            except Exception as e:
                print(f"A*路径计算错误: {e}")
                return float('inf')  # 出错时返回无限大距离

        def _init_blacklist():
            """初始化黑名单文件（如果不存在则创建）"""
            try:
                with open("blacklist.txt", "a+") as f:  # a+模式：不存在则创建
                    pass
            except Exception as e:
                print(f"无法初始化黑名单文件: {str(e)}")

        def _load_blacklist():
            """加载黑名单内容"""
            try:
                with open("blacklist.txt", "r") as f:
                    return set(line.strip() for line in f if line.strip())
            except FileNotFoundError:
                return set()

        def _add_to_blacklist(container_name):
            """将容器添加到黑名单"""
            try:
                with open("blacklist.txt", "a") as f:
                    f.write(f"{container_name}\n")
            except Exception as e:
                print(f"无法写入黑名单: {str(e)}")

        def _get_possible_containers_in_room_and_area(room_name, area_name):
            """获取房间子图中指定区域的可打开容器节点（自动过滤黑名单）"""
            blacklist = _load_blacklist()
            containers = []
            for node_name in graph.successors(room_name):
                node_info = graph.nodes[node_name]
                if (node_info.get("area") == area_name and
                    node_info.get("role") == "container" and
                    node_name not in blacklist and  # 新增黑名单过滤
                    "states" in node_info and
                    not node_info["states"].get(object_states.Open, True)):
                    containers.append((node_name, node_info))
            return containers
        
        def _check_target_in_scene(target_object):
            """检查目标物品是否已经在场景中可见（通过遍历图中所有物品节点）"""
            if not target_object:
                return False
            
            # 遍历图中所有节点
            for node, attrs in graph.nodes(data=True):
                # 排除房间和根节点，只检查物品节点
                if attrs.get('node_type') not in [NODETYPE.ROOM, NODETYPE.ROOT]:
                    # 获取节点名称并去除前后空格，转为小写
                    node_name = attrs.get("name", "").strip().lower()
                    target_name = target_object.strip().lower()
                    
                    # 完全匹配（例如 "pen" 不会匹配 "pencil"）
                    if node_name == target_name:
                        return True

            return False

        def _get_target_containers():
            """从 object_relations.txt 读取目标物品及其容器（保持不变）"""
            try:
                with open("object_relations.txt", "r") as f:
                    lines = f.readlines()
                    target_object = next(
                        (line.split(":")[1].strip() for line in lines 
                        if line.startswith("Determined_Target:")),
                        None
                    )
                    if not target_object:
                        return [], None
                    
                    # 先检查目标物品是否已经在场景中
                    if _check_target_in_scene(target_object):
                        return [], target_object
                    
                    target_containers = []
                    in_batch_1 = False
                    for line in lines:
                        line = line.strip()
                        if line == "### Batch_1 ###":
                            in_batch_1 = True
                        elif line.startswith("### Batch_") and line.endswith("###"):
                            in_batch_1 = False
                        elif in_batch_1 and ":" in line:
                            container_name, *objects = line.split(":")
                            if target_object in objects[0].split(",")[0].strip():
                                target_containers.append(container_name.strip())
                    return target_containers, target_object
            except FileNotFoundError:
                return [], None

        def _compute_optimal_order(containers, target_object, start_pos):
            """
            使用A*路径距离的动态归一化贪心算法
            """
            # 1. 确保start_pos是numpy数组格式
            current_pos = np.array(start_pos[:2]) if isinstance(start_pos, (tuple, list)) else start_pos[:2]
            
            # 2. 计算初始概率
            sbert = SentenceBERT()
            target_sentence = "A " + target_object
            container_sentences = ["A " + name for name, _ in containers]
            sim_matrix = sbert.compute_cooccurrence_bipartite([target_sentence], container_sentences)
            P = np.exp(sim_matrix[0]) / np.sum(np.exp(sim_matrix[0]))

            print("\n" + "="*40)
            print("初始概率分布:")
            for idx, ((name, _), prob) in enumerate(zip(containers, P)):
                print(f"{idx:2d}. {name:<20}: {prob:.4f}")

            # 3. 贪心选择
            remaining_indices = list(range(len(containers)))
            ordered_indices = []
            step = 1
            
            while remaining_indices:
                # 计算所有容器的A*路径距离
                current_distances = []
                for i in remaining_indices:
                    container_pos = containers[i][1]["closest_vor_node"][:2]
                    container_pos = np.array(container_pos) if isinstance(container_pos, (tuple, list)) else container_pos
                    container_pos = self.env.slam.voxel2world(container_pos)
                    dist = _get_path_distance(current_pos, container_pos)
                    current_distances.append(dist)
                
                # 动态归一化
                current_P = [P[i] for i in remaining_indices]
                P_max, P_min = max(current_P), min(current_P)
                D_max, D_min = max(current_distances), min(current_distances)
                
                # print(f"\n--- 第 {step} 轮选择 ---")
                # print(f"当前位置: {current_pos}")
                # print(f"距离范围: {D_min:.2f} 到 {D_max:.2f}")
                
                best_score = -np.inf
                best_idx = None
                
                for i, idx in enumerate(remaining_indices):
                    dist = current_distances[i]
                    
                    norm_P = (P[idx] - P_min) / (P_max - P_min + 1e-6)
                    norm_D = 1 - (dist - D_min) / (D_max - D_min + 1e-6)
                    fail_prob = np.prod([1 - P[k] for k in remaining_indices if k != idx])
                    
                    score = 0.3*norm_P + 0.6*norm_D + 0.1*fail_prob
                    
                    # print(f"候选 {idx:2d}: {containers[idx][0]:<15} | "
                    #     f"距离={dist:6.2f} | P={P[idx]:.3f}→{norm_P:.2f} | "
                    #     f"评分={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                # 更新状态
                ordered_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                if best_idx in remaining_indices:
                    current_pos = containers[best_idx][1]["closest_vor_node"][:2]
                    current_pos = self.env.slam.voxel2world(current_pos)
                    current_pos = np.array(current_pos) if isinstance(current_pos, (tuple, list)) else current_pos
                
                # print(f"√ 选择: {containers[best_idx][0]} (评分={best_score:.3f})")
                step += 1

            ############# 打印结果 ###############
            print("\n" + "="*40)
            print("最终访问顺序:")
            current_pos = np.array(start_pos[:2]) if isinstance(start_pos, (tuple, list)) else start_pos[:2]
            total_cost = 0
            
            for order, idx in enumerate(ordered_indices):
                pos = containers[idx][1]["pos"][:2]
                pos = np.array(pos) if isinstance(pos, (tuple, list)) else pos
                dist = _get_path_distance(current_pos, pos)
                total_cost += dist * P[idx] * np.prod([1 - P[ordered_indices[k]] for k in range(order)])
                
                print(f"{order+1:2d}. {containers[idx][0]:<15} | "
                    f"距离={dist:6.2f} | P={P[idx]:.4f} | "
                    f"累计成本={total_cost:.2f}")
                
                current_pos = pos
            
            print("="*40)
            return ordered_indices

        # 主逻辑
        try:
            # 初始化黑名单
            _init_blacklist()
            
            room_name, area_name = [x.strip() for x in argument.split(",")]
            target_containers, target_object = _get_target_containers()
            
            # 如果目标物品已经在场景中可见，直接返回成功
            if target_object and _check_target_in_scene(target_object):
                return True, None, f"Target object '{target_object}' already in scene", True
            
            current_pos = self.env.robots[0].get_position()
            
            # 获取容器（从房间子图中筛选区域）
            containers = _get_possible_containers_in_room_and_area(room_name, area_name)

            if not containers:
                return False, None, f"No closable containers in {area_name}", False

            # 计算最优顺序
            ordered_indices = _compute_optimal_order(
                containers, 
                target_object,
                current_pos
            )
            ordered_containers = [containers[i] for i in ordered_indices]

            ################## 执行打开操作#############################
            opened_containers = []
            failed_containers = []
            last_container_pos = None
            done = False

            for node_name, node_info in ordered_containers:
                obj = self.scene.objects_by_name.get(node_name)
                if not obj:
                    failed_containers.append((node_name, "Object not found"))
                    continue

                # 获取容器位置并生成路径点
                pos_container = node_info["pos_map"][:2]
                vor_nodes = np.stack(vor_graph.nodes)
                dists = np.linalg.norm(pos_container - vor_nodes, axis=1)
                close_nodes = vor_nodes[dists <= max(2.0/self.slam.voxel_size, dists.min())]  
                waypoints = [tuple(p) for p in self.env.slam.voxel2world(close_nodes)]
                # 寻找最近路径点并尝试打开
                _, closest_point, _, _ = self._find_closest_point(waypoints, euclidean_heuristic=True)
   
       

                success, feedback, done_flag = self.open_object(obj, closest_point)
                if done_flag:
                    done = True
                    print(f"检查过程中发现目标物！！！！！！")
                    break

                if success:
                    opened_containers.append(node_name)
                    _add_to_blacklist(node_name)
                    last_container_pos = closest_point
                    if node_name in target_containers:
                        done = True
                        with open("check_find_target.txt", "w") as f:
                            f.write('1')
                        break
                else:
                    failed_containers.append((node_name, feedback))

            # 构造返回结果
            feedback_msg = (
                f"Opened: {opened_containers} | "
                f"Blacklisted: {opened_containers} | "
                f"Failed: {[f'{c}:{m}' for c, m in failed_containers]}"
            )
            return (
                len(failed_containers) <= 3,  # subtask_success
                last_container_pos,
                feedback_msg,
                done
            )

        except Exception as e:
            return False, None, f"Error: {str(e)}", False

    def open_graph_node(self, graph, vor_graph, argument):
        done = False
        def _get_possible_nodes(room_name, object_name):
            # Check that the chosen object is actually articulated while checking for doors as well
            if "door" in object_name:
                possible_nodes = [closed_int_door for closed_int_door in graph.nodes.get(room_name, {}).get("closed_doors", [])]
            else:
                # Check that the chosen object type is actually articulated
                possible_nodes = list()
                for n in graph.successors(room_name):
                    is_closed = (not graph.nodes[n]['states'].get(object_states.Open, True))
                    graph_name = self.llm.human_to_graph_name(object_name)
                    # account for plural form of object name
                    if ((graph_name in n) or (graph_name.strip('s') in n)) and is_closed:
                        possible_nodes.append(n)
            return possible_nodes, [graph.nodes[n] for n in possible_nodes]
        try:
            room_name, object_name = self._parse_room_object_argument(argument)
            possible_node_names, possible_nodes = _get_possible_nodes(room_name, object_name)
        except:
            room_name, object_name = "None", "None"
            possible_node_names, possible_nodes = [], []

        if not possible_nodes:
            feedback = f"Object opening failed: no closed object instances of type {object_name} in {room_name}."
            with open('stuck.txt', 'w') as f:
                f.write('1')
            return False, (None, graph.nodes.get(room_name, {}).get("pos_map", None)), feedback, done
        
        # for other objects, we take the closest voronoi node as determined by closest-viewpoint logic. For doors, just take the overall closest voronoi node, as they are connecting multiple rooms
        if "door" in argument:
            vor_nodes = np.stack(vor_graph.nodes)
            closest_vor_nodes = dict()
            for node_name, node in zip(possible_node_names, possible_nodes):
                pos = self.env.slam.world2voxel(np.array(node)) if isinstance(node, tuple) else node["pos_map"][:2]
                dists_to_door = np.linalg.norm(np.array(pos) - vor_nodes, axis=1)
                is_close_to_door = dists_to_door <= max(2.0 / self.slam.voxel_size, dists_to_door.min())
                points = [tuple(i) for i in self.env.slam.voxel2world(vor_nodes[is_close_to_door])]
                _idx, _point, vn_costs, _paths = self._find_closest_point(points)
                # also add 2x distance of node <-> door to, from those on the shortest path, also take the vornoi-node that is closest to the door
                vn_costs += 2 * dists_to_door[is_close_to_door]
                closest_to_robot = points[np.argmin(vn_costs)]
                closest_vor_nodes[node_name] = closest_to_robot        
            idx, point, _costs, _paths = self._find_closest_point(list(closest_vor_nodes.values()), euclidean_heuristic=True)
        else:
            idx, point, _costs, _paths = self._find_closest_point(possible_nodes)

        obj = self.scene.objects_by_name.get(possible_node_names[idx], None)
        open_success, feedback, done = self.open_object(obj, point)
        return open_success, (obj, graph.nodes[room_name]["pos_map"]), feedback, done

    @staticmethod
    def _parse_room_argument(room_name):
        # don't replace any '-' that connect the room to the room number
        room_name = room_name[:-3].replace("_", " ").replace("-", " ") + room_name[-3:].replace("_", "-")
        for v in DIST_MAPPING.values():
            room_name = room_name.replace(f"({v})", "")
        return room_name

    @staticmethod
    def _parse_room_object_argument(argument):
        room_name, object_name = argument.split(", ")
        # returns false if it doesn't have a singular. E.g. singular_noun("book") == False -> return original word
        singular = inflect_engine.singular_noun(object_name)
        if singular:
            object_name = singular
        return LLMEnv._parse_room_argument(room_name), object_name

    def check_target_in_scene(self, graph):
        """
        检查目标物品是否已经在场景中可见（通过遍历图中所有物品节点）
        从 object_relations.txt 读取目标物品
        
        参数:
            graph: 场景图对象
            
        返回:
            bool: 目标物品是否已在场景中
            str: 找到的目标物品名称 (如果找到)
        """
        # 1. 从object_relations.txt读取目标物品
        try:
            with open("object_relations.txt", "r") as f:
                lines = f.readlines()
                target_object = next(
                    (line.split(":")[1].strip() for line in lines 
                    if line.startswith("Determined_Target:")),
                    None
                )
                
                if not target_object:
                    return False, None
        except FileNotFoundError:
            return False, None
        
        # 2. 在图中搜索目标物品
        target_lower = target_object.lower()
        for node, attrs in graph.nodes(data=True):
            # 排除房间和根节点，只检查物品节点
            if attrs.get('node_type') not in [NODETYPE.ROOM, NODETYPE.ROOT]:
                node_name = attrs.get("name", "")
                node_lower = node_name.lower()
                
                # 关键修改：提取最后一个 _ 之前的部分
                if "_" in node_lower:
                    node_prefix = node_lower.rsplit("_", 1)[0]
                else:
                    node_prefix = node_lower
                    
                # 精确匹配
                if target_lower == node_prefix:
                    return True, target_object
        
        return False, None
    def execute_action(self, action, argument, task_desc, graph, vor_graph=None):
        with open ("execute_action.txt", "w") as f:
            f.write("True")
            
        print(f"Executing {action}({argument})")
        print (f"{argument}#############")
        print (type(argument))
        done = False
        feedback = ""
        ################################################################
        if action in ["navigate", "go_to_and_open", "explore"]:
            target_found, target_name = self.check_target_in_scene(graph)
            if target_found:
                done = True
                feedback = f"Target object '{target_name}' already in scene, skipping {action}"
                history = ActionHistory(action, 
                                        object_name_graph=None,
                                        position=None,
                                        subtask_success=True)
                
                history.orig_api_call = f"{action}({argument})"
                self.action_history.append(history)
                
                print(feedback)
                feedback_msg = {"role": "env", "content": "; ".join([f"{action}({argument}) skipped: True", feedback])}
                return True, done, feedback_msg
    
    ###########################################################################################
        if action == "navigate":
            assert vor_graph is not None, "voronoi graph is required for navigation"
            try:
                room_name, object_name = self._parse_room_object_argument(argument)
                possible_nodes = [n for n in graph.successors(room_name) if self.llm.human_to_graph_name(object_name) in n]
            except:
                room_name, object_name = "None", "None"
                possible_nodes = []
            if len(possible_nodes) == 0:
                subtask_success = False
                feedback = f"Navigation to object failed because there is no {object_name} in {room_name}"
                history = ActionHistory(action, object_name_graph=self.llm.human_to_graph_name(object_name), position=graph.nodes.get(room_name, {}).get("pos_map", None), subtask_success=subtask_success)
            else:
                idx, point, _costs, _paths = self._find_closest_point([graph.nodes.get(n) for n in possible_nodes])
                # don't require to be exactly at the object, as it might be on a table or similar
                subtask_success = self.navigate_to_point(np.array(point),
                                                         success_thres_dist=1.5,
                                                         face_target=True,
                                                         early_termination_dist=0.3)
                if not subtask_success:
                    feedback = "Navigation to object failed"
                history = ActionHistory(action, object_name_graph=possible_nodes[idx], position=point, subtask_success=subtask_success)
###############################################

        elif action == "check":
            assert vor_graph is not None, "voronoi graph is required for preliminary object navigation"
            subtask_success, lastobj, feedback, done = self.open_all_containers_in_area(graph, vor_graph, argument=argument)
            history = ActionHistory(action, 
                                    object_name_graph=None, 
                                    position=None, 
                                    subtask_success=subtask_success,
                                    opendoors_roompos= None)
            

        elif action == "go_to_and_open":
            assert vor_graph is not None, "voronoi graph is required for preliminary object navigation"
            subtask_success, (obj, room_pos), feedback, done_flag = self.open_graph_node(graph, vor_graph, argument=argument)
            if done_flag:
                done = True
                print("开门过程中发现目标物！！！！！！")
            history = ActionHistory(action, 
                                    object_name_graph=obj.name if (obj is not None) else None, 
                                    position=obj.get_position()[:2] if (obj is not None) else room_pos, 
                                    subtask_success=subtask_success,
                                    opendoors_roompos=room_pos)
        # elif action == "close":
        #     subtask_success = False
        #     raise NotImplementedError()
        elif action == "explore":
            try:
                room_name = self._parse_room_argument(argument)
                frontier_points = graph.nodes.get(room_name, {}).get("frontier_points", {})
                frontier_points_within, frontier_points_leading_out = split_frontier_points(frontier_points)
                # for now just always prioritize those frontiers leading out of the room
                frontier_points = frontier_points_leading_out if len(frontier_points_leading_out) else frontier_points_within
            except:
                room_name = "None"
                frontier_points = []
            if len(frontier_points) == 0:
                subtask_success = False
                feedback = f"Frontier exploration failed because there are not frontier points in {room_name}"
                history = ActionHistory(action, object_name_graph=None, position=graph.nodes.get(room_name, {}).get("pos_map", None), subtask_success=subtask_success)
            else:
                _closest_idx, closest_frontier, _costs, _paths = self._find_closest_point(list(frontier_points))
                subtask_success, done_flag = self.navigate_to_point(closest_frontier, success_thres_dist=0.5, face_target=True, early_termination_dist=0.5)
                if done_flag:
                    done = True
                    print("探索过程中发现目标物！！！！！！")
                if not subtask_success:
                    feedback = f"Navigation to the frontier point failed."
                    with open ("stuck.txt", "w") as f:
                        f.write('1')
                history = ActionHistory(action, object_name_graph=None, position=closest_frontier, subtask_success=subtask_success)
        elif action == "done":
            subtask_success = True
            done = True
            history = ActionHistory(action, object_name_graph=None, position=None, subtask_success=subtask_success)
        else:
            raise ValueError(f"Unknown action {action}")

        history.orig_api_call = f"{action}({argument})"
        self.action_history.append(history)

        print(f"Subtask success: {subtask_success}")
        if (self.episode_info.get("failure_reason", "") == "max_high_level_steps timeout"):
            done = True

        task_success = self.task.evaluate_success(env=self.env)
        if task_success:
            if self.config["ground_truth_done_decision"]:
                done = True
            elif self.episode_info.get("num_low_level_steps_gtDone", None) is None:
                # log as alternative metric to evaluate impact of done decision
                self.episode_info["num_low_level_steps_gtDone"] = self.episode_info["num_low_level_steps"]
                self.episode_info["num_high_level_steps_gtDone"] = self.episode_info["num_high_level_steps"]
                self.episode_info["magic_open_actions_gtDone"] = self.episode_info["magic_open_actions"]

        print(feedback)
        feedback_msg = {"role": "env", "content": "; ".join([f"{action}({argument}) success: {subtask_success}", feedback])}
        return subtask_success, done, feedback_msg

    def plot_conversation(self, conversation: Conversation, action: str, argument: str, ax: plt.Axes, font_size=6.5, add_fig_title: bool = True):
        if add_fig_title:
            self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, {action}({argument})")

        ax.clear()
        ax.set_axis_off()
        colors = {"system": "wheat", "user": "orange", "assistant": "orangered", "env": "green"}
        
        h = 0.05
        for m in conversation.messages_including_env:
            orig_txt = f"{m['role']}: {m['content']}"
            # remove empty lines
            txt = "".join([s for s in orig_txt.splitlines(True) if s.strip("\r").strip("\n")])
            txt = txt.strip("\n").strip("\r")
            # props = dict(boxstyle='round', facecolor=colors[m["role"]], alpha=0.5)
            props = dict(boxstyle='round', facecolor="white", alpha=0.0, edgecolor="white")
            t = ax.text(-0.05,
                        0.975 - h, 
                        txt,
                        transform=ax.transAxes, 
                        fontsize=font_size, 
                        verticalalignment='top', 
                        bbox=props,
                        wrap=True
                        )
            t._get_wrap_line_width = lambda : 1.05 * ax.get_window_extent().width
            self.env.f.canvas.draw()
            last_h = t.get_bbox_patch().get_height() / ax.get_window_extent().height
            h += last_h + 0.025
            
            b = t.get_bbox_patch()
            # bb = b.get_window_extent()
            p = FancyBboxPatch(xy=[-0.05, 1.0 - h], 
                               width=1.05, 
                               height=last_h, 
                               transform=ax.transAxes, 
                               alpha=0.5,
                               boxstyle="round,pad=0.01",
                               facecolor=colors[m["role"]],
                               edgecolor="k",
                               clip_on=False)
            ax.add_patch(p)
            
        if (h > 1.025) and (font_size > 4.5):
            # just try again with smaller font size to fit everything into the figure
            self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=ax, font_size=font_size - 0.5, add_fig_title=False)

    def visualize(self, state=None):
        super().visualize(state=state)
        self._label_rooms_on_map(state, ax_idx=1)


class JsonLLMEnv(LLMEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _create_prompt(self, 
                       task_description, 
                       current_room,
                       graph,
                       room_graph,
                       nlp_history,
                       *args, 
                       **kwargs) -> Conversation:
        system_prompt = f"You are a robot in an unexplored house. Your task is to {task_description}."
        system_prompt += f" You have the following actions available that you can use to achieve this task:\n"
        for i, (action, description) in enumerate(self.possible_actions.items()):
            system_prompt += f"""{i+1}. {action}({description[0]}): {description[1]}\n"""
        system_prompt += f"\nOutput Response Format:\n"\
        "Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.\n"\
        "Reasoning: justify why the next action is important to solve the task.\n"\
        "Command: function call"

        # roughly following SayPlan's prompt (see page 50):
        # {nodes: {room: [{id: bobs_room}, {id: toms_room}, {id: jacks_room}, {id: kitchen}, {id: livingroom}], 
        #          pose: [{id: pose1}, {id: pose2}, {id: pose3}, {id: pose4}, {id: pose5}], 
        #          agent: [{location: bobs_room, id: agent}], 
        #          asset: [{room: toms_room, state: free, affordances: [release], id: bed2}, 
        #                  {room: toms_room, state: closed, affordances: [open, close, release], id: wardrobe2}, 
        #                  {room: kitchen, state: closed, affordances: [open, close, release], id: fridge}, 
        #                  {room: kitchen, affordances: [turn_on, turn_off], state: off, id: coffee_machine}, 
        #                  {room: bobs_room, state: free, affordances: [release], id: bed1}, 
        #                  {room: bobs_room, state: closed, affordances: [open, close, release], id: wardrobe1}], 
        #          object: [{affordances: [pickup], state: inside_of(wardrobe1), attributes: "blue", id: coffee_mug}]
        #          }, 
        #  links: [bobs_room↔pose1, bobs_room↔agent, bobs_room↔bed1, bobs_room↔wardrobe1, 
        #          toms_room↔pose1, toms_room↔pose2, toms_room↔pose5, toms_room↔bed2,
        #          toms_room↔wardrobe2, jacks_room↔pose2, jacks_room↔pose3, kitchen↔pose3,
        #          kitchen↔pose4, kitchen↔pose5, kitchen↔fridge, kitchen↔coffee_machine,
        #          livingroom↔pose4, wardrobe1↔coffee_mug]}
        graph_with_robot = nx.Graph()
        for _node_name, node_data in graph.nodes(data=True):
            if node_data.get("semantic_class_name", None) == "door":
                continue
            if "room_id" in node_data:
                room_name = self.room_classification[NODETYPE.roomname(node_data["room_id"])]
            if node_data["node_type"] == NODETYPE.ROOM:
                keys = ["frontier_points", "closed_doors"]
                props = {k: node_data.get(k, set()) for k in keys if (k in node_data)}
                props["id"] = room_name
            elif node_data["node_type"] == NODETYPE.OBJECT:
                keys = []
                props = {k: node_data[k] for k in keys if (k in node_data)}
                props["id"] = node_data["name"]
                props["room"] = room_name
                if "states" in node_data:
                    for state, value in node_data["states"].items():
                        if state == object_states.Open:
                            props["state"] = "open" if value else "closed"
            else:
                continue
            props["node_type"] = str(node_data["node_type"]).split('.')[1].lower()
            graph_with_robot.add_node(props["id"], **props)
            
            if node_data["node_type"] == NODETYPE.OBJECT:
                graph_with_robot.add_edge(room_name, props["id"])
            elif node_data["node_type"] == NODETYPE.ROOM:
                for room in room_graph.neighbors(NODETYPE.roomname(node_data["room_id"])):
                    graph_with_robot.add_edge(room_name, room)
        
        graph_with_robot.add_node("robot", location=current_room, id="robot")
        graph_with_robot.add_edge(current_room, "robot")
        
        data = nx.node_link_data(graph_with_robot)
        links = list({f"{d['source']} - {d['target']}" for d in data["links"]})
        prompt = f"Scene Graph: {{nodes: {data['nodes']}, links: {links}}},\n".replace("'", "")
        
        if len(nlp_history):
            prompt += f"History: {', '.join(nlp_history)}.\n"

        prompt += f"""What is the best next action to complete the task as efficiently as possible? I you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.\n"""  # In general, prioritize opening doors if there is not enough evidence to explore a promising room.\n""" 
        prompt += f"Remember:\n"\
            "1. Respond with a function call\n"\
            "2. You can only use the objects and rooms that you have already found\n"\
            "3. You can only explore rooms that have frontier points\n"\
            "4. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.\n"\
            "5. If some actions failed repeatedly, they may not be possible.\n"

        conversation = Conversation(messages=[self.last_env_feedback,
                                              {"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        return conversation
    
