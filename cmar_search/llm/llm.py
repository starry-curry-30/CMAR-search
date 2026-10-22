# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import matplotlib.pyplot as plt
import copy
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pprint import pformat, pprint
from typing import List, Tuple, Union
import openai
import inflect
import networkx as nx
import numpy as np
from igibson import object_states
from openai import OpenAI, OpenAIError
from pygments import highlight
from pygments.formatters import Terminal256Formatter, TerminalFormatter
from pygments.lexers import PythonLexer
from sty import fg
from cmar_search.utils.constants import NODETYPE, POSSIBLE_ROOMS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import os
import torch
import clip
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from cmar_search.topology.room_graph import create_room_object_graph
# Public release: configure credentials through environment variables only.
client = None


def _get_client():
    global client
    if client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running CMAR-search.")
        client = OpenAI(api_key=api_key,
                        base_url=os.environ.get("CMAR_LLM_BASE_URL", "https://api.deepseek.com"))
    return client

inflect_engine = inflect.engine()


def pprint_color(obj, style="staroffice", width=200):
    txt = highlight(pformat(obj, width=width), PythonLexer(), Terminal256Formatter(style=style))
    print(txt, end="")
    return txt


class Conversation:
    def __init__(self, messages: List[dict], include_env_messages: bool = False) -> None:
        self._messages = messages
        self._include_env_messages = include_env_messages

    def add_message(self, message: dict):
        self._messages.append(message)

    @property
    def messages(self):
        if self._include_env_messages:
            return self._messages
        else:
            return [m for m in self._messages if m["role"].lower() not in ["env", "environment"]]

    @property
    def messages_including_env(self):
        return self._messages


@lru_cache(maxsize=None)
def send_query_cached(messages: list, model: str, temperature: float):
    assert temperature == 0.0, "Caching only works for temperature=0.0, as eitherwise we want to get different responses back"
    messages = [dict(m) for m in messages]
    return _get_client().chat.completions.create(model=model,
                                          messages=messages,
                                          temperature=temperature,
                                          )


def send_query(messages: list, model: str, temperature: float):
    if temperature == 0.0:
        hashable_messages = tuple(tuple(m.items()) for m in messages)
        return send_query_cached(messages=hashable_messages, model=model, temperature=temperature)
    else:
        return _get_client().chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=temperature,
                                              )


class LLM:
    def __init__(self,
                 model: str,
                 room_classification_model: str,
                 open_set_rooms: bool = False,
                 temperature: float = 0.0,
                 debug=False) -> None:
        # available_models = [m.id for m in client.models.list().data]    廖凯尧改
        # assert model in available_models, available_models            廖凯尧改
        self.model = model
        self.room_classification_model = room_classification_model
        self.temperature = temperature
        self.debug = debug
        self.open_set_rooms = open_set_rooms

    def send_query(self, conversation: Conversation, model: str = None):
        """Example input:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Knock knock."},
                {"role": "assistant", "content": "Who's there?"},
                {"role": "user", "content": "Orange."},
            ],
        """
        num_attempts = 0
        while True:
            try:
                response = send_query(model=model or self.model,
                                      messages=conversation.messages,
                                      temperature=self.temperature, )
                break
            except OpenAIError as e:
                print(f"Attempting again after {e}")
                num_attempts += 1
                time.sleep(5)
            assert num_attempts < 10, "Too many openai errors"

        role = response.choices[0].message.role
        content = response.choices[0].message.content

        if self.debug:
            pprint_color("#################################\n", width=200)
            pprint_color("\n+++++++++++++++++++++++++++++++++\n".join(
                [f"{m['role']}: {m['content']}" for m in conversation.messages]), width=200)
            pprint_color(f"+++++++++++++++++++++++++++++++++\n {role}: {content}", width=200, style="rrt")

        conversation.add_message({"role": role, "content": content})

        return content

    @staticmethod
    def to_human_readable_object_name(object_name: str, states=None) -> str:
        if states is not None:
            if object_states.Open in states:
                prefix = "opened " if states[object_states.Open] else "closed "
                object_name = prefix + object_name
        return re.sub(r'-\d+', '', re.sub(r'_\d+', '', object_name)).replace("_", "-")

    @staticmethod
    def human_to_graph_name(object_name: str) -> str:
        # strip states, replace spaces. Will still not be an exact match, as original node names include a number
        object_name = object_name.replace("closed", "").replace("opened", "").replace("-", "_")
        # replace numbers that stand alone (i.e. object counters in the beginning of the object name)
        # NOTE: not stripping the potential 's' for plural objects atm
        return re.sub(r'\b\d+\b', '', object_name).strip(" ")

    @staticmethod
    def create_room_object_dict(graph, open_door_inclusion: str = "as_object", room_classification=None,
                                include_explored: bool = False) -> dict:
        rooms = list(graph.successors("root"))
        room_dict = {}
        for room in rooms:
            objects = sorted(list(graph.successors(room)))
            # print(f"Room {room} has objects {objects}####################################")
            objects_readable = [LLM.to_human_readable_object_name(o, states=graph.nodes[o]["states"]) for o in objects]
            # can't add object_states to the name parsing since open doors are not a node in the graph atm
            if open_door_inclusion == "as_object":
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) for d in
                              list(graph.nodes[room]["open_doors"])]
            elif open_door_inclusion == "as_edge":
                assert room_classification is not None
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) + (
                    f" to {room_classification[d[1]]}" if d[1] is not None else "") for d in
                              list(graph.nodes[room]["open_doors"])]
            elif open_door_inclusion == "ignore":
                open_doors = []
            else:
                raise ValueError(f"Unknown open_door_inclusion: {open_door_inclusion}")
            objects_readable += open_doors

            if include_explored:
                has_frontiers = len(graph.nodes.get(room)["frontier_points"])
                if has_frontiers:
                    objects_readable.append("unexplored area")

            occurences = Counter(objects_readable)
            counted_objects = [
                f"{v} {inflect_engine.plural(k) if inflect_engine.plural(k) else (k + 's')}" if (v > 1) else k for k, v
                in occurences.items()]
            room_dict[room] = counted_objects
        return room_dict




    # def _parse_rooms(self, input_rooms: list, possible_rooms: list, response: str):
    #     room_classification = {}

    #     lines = response.split("\n")
    #     for line in lines:
    #         line = line.strip().lower()

    #         for room in input_rooms:
    #             if room in room_classification:
    #                 continue

    #             if room in line:
    #                 # 试图通过正则表达式提取房间类型和概率
    #                 match = re.search(r'\[.*\]', line)
    #                 if match:
    #                     # 提取到的房间类型和概率部分
    #                     room_info_str = match.group(0)
                        
    #                     # 去除多余的符号并分割每个候选项
    #                     room_info_str = room_info_str[1:-1]  # 去掉外层的 []

    #                     # 处理每个候选项
    #                     room_info = room_info_str.split("), (")

    #                     # 处理每个候选项，确保去掉多余的括号
    #                     parsed_candidates = []
    #                     for candidate in room_info:
    #                         parts = candidate.split(", ")
    #                         if len(parts) == 2:
    #                             room_type = parts[0].strip()
    #                             probability_str = parts[1].strip()

    #                             # 使用正则去除概率中的非数字字符（如右括号）
    #                             probability_str = re.sub(r'[^\d\.]', '', probability_str)

    #                             # 使用正则去除房间类型前后的括号
    #                             room_type = re.sub(r'^[\(\)]*(.*?)[\(\)]*$', r'\1', room_type).strip()

    #                             try:
    #                                 probability = float(probability_str)
    #                                 parsed_candidates.append((room_type, probability))
    #                             except ValueError:
    #                                 continue

    #                     # 确保我们解析到三个候选房间类型
    #                     if len(parsed_candidates) == 3:
    #                         room_classification[room] = parsed_candidates
    #                     else:
    #                         room_classification[room] = []  # 如果数据不完整，设置为空

    #     assert set(room_classification.keys()) == set(input_rooms), f"Did not find all input_rooms in the response: {room_classification}"

    #     if self.debug:
    #         pprint(room_classification)

    #     return room_classification

    def _parse_rooms(self, input_rooms: list, possible_rooms: list, response: str):
        room_classification = {}

        lines = response.split("\n")
        for line in lines:
            line = line.lower()
            for room in input_rooms:
                if room in room_classification.keys():
                    # assume that the first mention of a room is the correct one, as usually it responds with a list of the rooms, then adds an explanation
                    continue
                if not self.open_set_rooms:
                    if (room in line) or (room.replace("_", " ") in line):
                        for possible_room in possible_rooms:
                            if possible_room in line:
                                room_classification[room] = possible_room
                                break
                        # Allow open-vocab answers, better than failing
                        if (room not in room_classification) and (":" in line):
                            room_classification[room] = line.split(":")[1].strip()
                else:
                    if (room in line) or (room.replace("_", " ") in line):
                        room_classification[room] = line.split(":")[-1].strip()
                        break
        # If only one room is explored so far the LLM tends to answer in a full sentence instead of list
        # NOTE: could delete the following, have not observed this behaviour in a while
        if room_classification.keys() != set(input_rooms) and len(input_rooms) == 1 and len(lines) == 1:
            for possible_room in possible_rooms:
                if possible_room in lines[0]:
                    room_classification[room] = possible_room
                    break

        assert room_classification.keys() == set(
            input_rooms), f"Did not find all input_rooms in the response: {room_classification}"

        if self.debug:
            pprint(room_classification)
        return room_classification
    def classify_rooms(self, obs,
                       system_prompt: str = "You are a helpful assistant, visiting a new apartment.") -> dict:
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root"))

        room_dict = self.create_room_object_dict(graph, include_explored=False)

        prompt = f"You observe {len(rooms)} rooms, they contain the following objects:\n"
        for room in rooms:
            prompt += f"- {room} contains [{', '.join(room_dict[room])}].\n"
        if not self.open_set_rooms:
            prompt += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. If you are unsure, classify them as other room.\n"
        else:
            prompt += "Please classify the rooms. If you are unsure, classify them as other room.Only output one room category, no other information required\n"
        prompt += "Output Response Format:\n" \
                  "A list with bullet points of the form\n" \
                  "- room-X: room type\n"
        if not self.open_set_rooms:
            prompt += "Remember: you can only use the given categories."

        conversation = Conversation(messages=[{"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        for i in range(3):
            try:
                response = self.send_query(conversation=conversation, model=self.room_classification_model)
                output = self._parse_rooms(input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
                break
            except:
                conversation.add_message({"role": "user", "content": "Please classify all rooms listed."})
                assert i < 2, "Could not parse room classification"
        return output



#     def classify_rooms(self, obs,
#                    system_prompt: str = "You are a helpful assistant, visiting a new apartment.") -> dict:
#         graph = obs["room_object_graph"]
#         rooms = list(graph.successors("root"))

#         room_dict = self.create_room_object_dict(graph, include_explored=False)

#         prompt = f"You observe {len(rooms)} rooms, they contain the following objects:\n"
#         for room in rooms:
#             prompt += f"- {room} contains [{', '.join(room_dict[room])}].\n"
#         if not self.open_set_rooms:
#             prompt += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. "
#             prompt += "If you are unsure, classify them as other room. Provide the top 3 most likely room types and their probabilities for each room.\n"
#         else:
#             prompt += "Please classify the rooms and provide the top 3 most likely room types and their probabilities.\n"
#         prompt += "Output Response Format:\n" \
#                 "A list with bullet points of the form\n" \
#                 "- room-X: [(room type 1, probability 1), (room type 2, probability 2), (room type 3, probability 3)]\n"

#         conversation = Conversation(messages=[{"role": "system", "content": system_prompt},
#                                             {"role": "user", "content": prompt}])
#         for i in range(3):
#             try:
#                 response = self.send_query(conversation=conversation, model=self.room_classification_model)
#                 llm_output = self._parse_rooms(input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
#                 break
#             except:
#                 conversation.add_message({"role": "user", "content": "Please classify all rooms listed and provide top 3 room types with probabilities."})
#                 assert i < 2, "Could not parse room classification"

#         # 集成 CLIP 判断逻辑
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model, preprocess = clip.load("ViT-B/32", device=device)
        
#         final_room_classification = {}

#         # CLIP图像特征计算和余弦相似度计算
#         for room, llm_candidates in llm_output.items():
#             text_labels, llm_probabilities = zip(*llm_candidates)
#             text = clip.tokenize(text_labels).to(device)
#             with torch.no_grad():
#                 text_features = model.encode_text(text)

#             # 检查是否有对应的文件夹
#             room_folder = f"save_rgb/room-{room.split('-')[1]}"  # 假设room的格式为"room-X"
            
#             # 如果文件夹存在
#             if os.path.isdir(room_folder):
#                 clip_similarities = np.zeros(len(text_labels))
#                 image_features_list = []

#                 # 加载该房间对应的所有图片
#           #      print(f"\n[DEBUG] CLIP: Processing images in {room_folder} for room {room}...")
#                 for filename in os.listdir(room_folder):
#                     if filename.endswith(".png") or filename.endswith(".jpg"):
#                         image_path = os.path.join(room_folder, filename)
#                         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#                         with torch.no_grad():
#                             image_features = model.encode_image(image)
#                             image_features_list.append((filename, image_features))
#    #                     print(f"[DEBUG] CLIP: Processed image {filename} in {room_folder}")

#                 # 计算每个房间类型与图片特征的余弦相似度
#       #          print(f"[DEBUG] CLIP: Calculating cosine similarities for {room}...")
#                 for filename, image_features in image_features_list:
#                     similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
#                     # 输出每个类别的相似度
#                     # print(f"\n[DEBUG] CLIP: Image {filename} similarities with room types:")
#                     # for idx, label in enumerate(text_labels):
#                     #     print(f"- {label}: similarity = {similarities[idx]:.4f}")
                    
#                     clip_similarities += similarities

#                 clip_similarities /= len(image_features_list)  # 计算平均相似度

#                 # 打印 CLIP 分类结果
#                 clip_probabilities = clip_similarities / clip_similarities.sum()  # 将相似度转化为概率
#     #            print(f"\n[DEBUG] CLIP Classification for {room}:")
#                 for label, prob in zip(text_labels, clip_probabilities):
#                     print(f"- {label}: {prob:.4f}")
                
#                 # 最终得分：CLIP * 0.4 + LLM * 0.6
#                 final_scores = np.array(clip_probabilities) * 0.4 + np.array(llm_probabilities) * 0.6
#             else:
#                 # 如果没有对应的文件夹，则只依赖LLM的结果
#   #              print(f"[DEBUG] CLIP: No images found for {room}. Using only LLM results.")
#                 final_scores = np.array(llm_probabilities)

#             # 打印 LLM 分类结果和概率
#    #         print(f"[DEBUG] LLM Classification for {room}:")
#             for label, prob in zip(text_labels, llm_probabilities):
#                 print(f"- {label}: {prob:.4f}")

#             # 计算最终分类
#             best_match_index = final_scores.argmax()  # 选择得分最高的房间类型
#             final_room_classification[room] = text_labels[best_match_index]  # 更新房间的最终分类

#         # 打印最终分类结果
#         print("\n[DEBUG] Final Room Classifications:")
#         for room, room_type in final_room_classification.items():
#             print(f"{room}: {room_type}")

#         return final_room_classification



####################################################################

    # def determine_object_relationships(self, obs):
    #     # 从观察数据中获取房间对象图
    #     graph = obs["room_object_graph"]
    #     # 获取物体间的关系列表
    #     object_relations = self.find_intersecting_objects(graph)

    #     # 构建输入给LLM的提示信息
    #     # prompt = """In a home scenario, which relationships are reasonable assuming there are only ON and INSIDE relationships? Help me determine the ON and INSIDE relationships. \n"""
    #     # prompt += "Here are all the intersecting objects:\n"

    #     prompt = """You need to determine which objects are containers, such as tables, straight_chair, etc., and which are movable items, such as cups, milk, etc. 
    #     Then, you need to determine which item relationships are common in family settings and only keep them.\n"""

    #     prompt += """Each movable item can only and must be placed on top of or inside one container. For example, notebook_10001 can only appear in one pair of relationships. However, each container may hold multiple movable items. 
    #     **Strictly ensure that no movable item appears in more than one relationship. If a movable item appears in multiple relationships, retain only the most logical and likely one.**\n"""

    #     prompt += """Please note that containers are more likely to have an ON relationship or an INSIDE relationship. For example, tables are more likely to have an ON relationship, and cabinets are more likely to have an INSIDE relationship. If the given relationship differs from your judgment, delete it.\n"""

    #     prompt += """Please delete the relationship between two containers (such as Straight_chair_2 is on table_1001) and recurring movable items in the answer. Retain only the most logical and likely relationships.
    #     Your answer should strictly follow this rule: **Each movable item can only appear once in the final output, either "on" or "inside" a single container. Do not allow any movable item to appear in multiple relationships.**
        
    #     Your answer should be in the format of:
    #     - 'A certain movable item is on a certain container'
    #     - 'A certain movable item is inside a certain container'
    #     No need for analysis.
    #     For example:
    #     - cup_10018 is inside top_cabinet_28
    #     - milk_10012 is on kitchen_counter_1021\n"""

    #     prompt += "Here are all the intersecting objects:\n"

    #     # 将所有关系直接添加到提示信息中
    #     for relation in object_relations:
    #         prompt += f"- {relation}\n"
            

    #     # 初始化对话
    #     conversation_messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ]
    #     conversation = Conversation(conversation_messages)

    #     # 发送查询并接收LLM的响应
    #     response = self.send_query(conversation=conversation, model=self.model)

    #     # 解析LLM返回的结果
    #     relationships = response.split("\n")
    #     parsed_relationships = []
    #     for relationship in relationships:
    #             # 去掉开头的短划线和空格
    #             cleaned_relationship = relationship.strip().lstrip('-').strip()
    #             if "is on" in cleaned_relationship or "is inside" in cleaned_relationship:
    #                 parsed_relationships.append(cleaned_relationship)

    #     return parsed_relationships
    def determine_object_relationships(self, obs):
        graph = obs["room_object_graph"]
        intersecting_pairs, non_intersecting_objects = self.find_intersecting_objects(graph)
        filtered_keywords = {"carpet", "door", "window"}

        # Step 1: 生成名称映射（移除末尾 _编号）
        name_mapping = {}  # 格式: {原始名称: 通用名称}
        reverse_mapping = {}  # 格式: {通用名称: [原始名称1, 原始名称2, ...]}
        
        # 处理相交物体对中的名称
        for pair in intersecting_pairs:
            for obj in pair:
                if any(keyword in obj for keyword in filtered_keywords):
                    continue
                generic_name = obj.rsplit('_', 1)[0]  # 移除最后一个下划线及之后的部分
                name_mapping[obj] = generic_name
                reverse_mapping.setdefault(generic_name, []).append(obj)
        
        # 处理非相交物体中的名称
        for obj in non_intersecting_objects:
            if any(keyword in obj for keyword in filtered_keywords):
                continue
            generic_name = obj.rsplit('_', 1)[0]
            name_mapping[obj] = generic_name
            reverse_mapping.setdefault(generic_name, []).append(obj)

        # Step 2: 获取所有唯一的通用名称（用于LLM分类）
        unique_generic_names = set(name_mapping.values())

        # Step 3: 调用LLM分类通用名称
        prompt = """Classify objects into:
    - Containers: Objects that can hold other items (e.g., table, shelf).
    - Movable Items: Objects that can be placed on/in containers (e.g., cup, book).
    Format: Containers: [obj1, obj2]; Movable Items: [obj3, obj4]
    You don't need to provide any other information 
    Objects to classify:
    - """ + "\n- ".join(unique_generic_names) + "\n"

        # 构造Conversation对象（假设需要）
        conversation_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        conversation = Conversation(conversation_messages)
        response = self.send_query(conversation=conversation, model=self.model)

        # Step 4: 解析LLM响应（改进版，兼容单行响应）
        generic_containers = []
        generic_movable_items = []
        try:
            # 处理 Containers
            containers_start = response.find("Containers:") + len("Containers:")
            containers_end = response.find(";", containers_start)
            if containers_end == -1:
                containers_end = response.find("\n", containers_start)
            generic_containers = [
                x.strip().strip("]") for x in 
                response[containers_start:containers_end].strip(" []").split(",")
            ]
            
            # 处理 Movable Items
            movable_start = response.find("Movable Items:") + len("Movable Items:")
            movable_end = response.find("\n", movable_start)
            generic_movable_items = [
                x.strip().strip("]") for x in 
                response[movable_start:movable_end].strip(" []").split(",")
            ]
        except Exception as e:
            print(f"LLM响应解析失败: {e}")

        # Step 5: 将通用名称的分类结果映射回原始名称
        containers = []
        movable_items = []
        for generic_name in generic_containers:
            containers.extend(reverse_mapping.get(generic_name.strip(), []))
        for generic_name in generic_movable_items:
            movable_items.extend(reverse_mapping.get(generic_name.strip(), []))

        # Step 6: 生成最终的valid_pairs（改进版，显式处理相交容器）
        valid_pairs = []
        
        # 处理相交物体对
        for pair in intersecting_pairs:
            obj1, obj2 = pair
            if any(keyword in obj1 for keyword in filtered_keywords) or any(keyword in obj2 for keyword in filtered_keywords):
                continue
            
            # 情况1：容器 + 可移动物品
            if obj1 in containers and obj2 in movable_items:
                valid_pairs.append((obj1, obj2))
            elif obj2 in containers and obj1 in movable_items:
                valid_pairs.append((obj2, obj1))
            
            # 新增逻辑：相交容器也生成 "nothing" 配对
            if obj1 in containers and obj2 in containers:
                valid_pairs.append((obj1, "nothing"))
                valid_pairs.append((obj2, "nothing"))
        
        # 处理非相交的容器
        for obj in non_intersecting_objects:
            if obj in containers and not any(keyword in obj for keyword in filtered_keywords):
                valid_pairs.append((obj, "nothing"))

        # 去重（避免重复添加）
        valid_pairs = list(set(valid_pairs))
        print("Final valid_pairs:", valid_pairs)
        return valid_pairs

    def update_object_attributes(self, valid_pairs, obs):
        graph = obs["room_object_graph"]
        relations_dict = {}
        
        for container, movable_item in valid_pairs:
            container = container.strip()
            movable_item = movable_item.strip()

            # 更新容器属性
            if container not in relations_dict:
                relations_dict[container] = {"role": "container", "relation": []}
            if movable_item != "nothing":
                relation_str = f"{movable_item} on {container}"
                relations_dict[container]["relation"].append(relation_str)

            # 更新可移动物品属性（如果不是nothing）
            if movable_item != "nothing":
                if movable_item not in relations_dict:
                    relations_dict[movable_item] = {"role": "movable", "relation": []}
                relations_dict[movable_item]["relation"].append(relation_str)
        
        # 更新图中所有节点的属性
        for node, attrs in graph.nodes(data=True):
            node_name = attrs.get('name', "").strip()
            if node_name in relations_dict:
                graph.nodes[node].update(relations_dict[node_name])
            else:
                graph.nodes[node].setdefault("role", "others")
                graph.nodes[node].setdefault("relation", [])
        
        # print("\n=== 物品角色信息 ===")
        # for node, attrs in graph.nodes(data=True):
        #     node_name = attrs.get('name', "").strip()
        #     role = attrs.get('role', "未定义")
        #     relations = attrs.get('relation', [])
        #     print(f"物品: {node_name}")
        #     print(f"角色: {role}")
        #     print(f"关系: {relations if relations else '无'}")
        #     print("-" * 30)
        return obs
    
    def bbox_intersect(self, bbox1, bbox2, z_threshold):
        """
        判断两个边界框是否相交，考虑Z轴接近度。
        :param bbox1: 第一个边界框的顶点数组 (n, 3)
        :param bbox2: 第二个边界框的顶点数组 (m, 3)
        :param z_threshold: Z轴接近度阈值
        :return: bool 表示是否相交
        """
        min_bbox1 = np.min(bbox1, axis=0)
        max_bbox1 = np.max(bbox1, axis=0)
        
        min_bbox2 = np.min(bbox2, axis=0)
        max_bbox2 = np.max(bbox2, axis=0)
        
        # 检查X和Y轴上的重叠
        overlap_x = not (max_bbox1[0] < min_bbox2[0] or max_bbox2[0] < min_bbox1[0])
        overlap_y = not (max_bbox1[1] < min_bbox2[1] or max_bbox2[1] < min_bbox1[1])
        
        # 对于Z轴，同时检查是否重叠或足够接近
        overlap_z = not (max_bbox1[2] < min_bbox2[2] or max_bbox2[2] < min_bbox1[2])
        distance_z = min(abs(max_bbox1[2] - min_bbox2[2]), abs(max_bbox2[2] - min_bbox1[2]))
        close_z = distance_z <= z_threshold
        
        # 只要X、Y轴重叠，且Z轴重叠或接近即可认为相交
        return overlap_x and overlap_y and (overlap_z or close_z)

    def find_intersecting_objects(self, room_object_graph):
        intersecting_pairs = []
        object_names = [node for node, attrs in room_object_graph.nodes(data=True) if attrs.get('node_type') == NODETYPE.OBJECT]
        interaction_map = {name: False for name in object_names}  # 初始化为False，表示暂无交互
        
        objects = [(node, attrs) for node, attrs in room_object_graph.nodes(data=True) if attrs.get('node_type') == NODETYPE.OBJECT]
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                node_i, attrs_i = objects[i]
                node_j, attrs_j = objects[j]
                
                bbox_i = np.array(attrs_i['bbox_world_vertices'])
                bbox_j = np.array(attrs_j['bbox_world_vertices'])
                
                is_intersecting = self.bbox_intersect(bbox_i, bbox_j, z_threshold=0.01)
                if is_intersecting:
                    intersecting_pairs.append((node_i, node_j))
                    interaction_map[node_i] = True  # 标记为有交互
                    interaction_map[node_j] = True  # 标记为有交互
        
        # 打印不与其他物体相交的物体名称
        non_intersecting_objects = [name for name, has_interaction in interaction_map.items() if not has_interaction]
        # print("Non-intersecting objects:", non_intersecting_objects)
        # print("Intersecting object pairs:", intersecting_pairs)
        
        return intersecting_pairs, non_intersecting_objects



    # def determine_area_functions(self, obs): #无修正
    #     """
    #     根据图结构的层级关系获取房间及其物品，并进行区域划分。
        
    #     :param obs: 观察数据，包含房间对象图。
    #     """
    #     # 获取房间对象图
    #     graph = obs["room_object_graph"]
        
    #     # 获取所有房间节点
    #     rooms = list(graph.successors("root"))
        
    #     # 初始化房间数据结构
    #     room_data_dict = {}  # key: room_name, value: {"containers": [], "container_positions": [], "movable_items": {}}
        
    #     for room in rooms:
    #         # 获取房间中的所有物品
    #         objects = sorted(list(graph.successors(room)))
            
    #         containers = []
    #         container_positions = []
    #         movable_items = {}
            
    #         for obj in objects:
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', "").strip()
    #             name = attrs.get('name', "").strip()
                
    #             if role == "container":
    #                 # 提取 pos_map 属性
    #                 pos_map = attrs.get('pos_map')
    #                 if isinstance(pos_map, tuple) and len(pos_map) == 3:
    #                     x, y, _ = pos_map
    #                 else:
    #                     print(f"Invalid or missing position for node {name}, using default (0, 0, 0)")
    #                     x, y, _ = 0, 0, 0
                    
    #                 containers.append(name)
    #                 container_positions.append([x, y])
    #                 movable_items[name] = []  # 初始化容器对应的空列表
                    
    #         # 然后再次遍历所有物品，只处理可移动物品
    #         for obj in objects:
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', "").strip()
    #             name = attrs.get('name', "").strip()
                
    #             if role == "movable":
    #                 relations = attrs.get('relation', [])
    #                 for relation in relations:
    #                     if "on" in relation:
    #                         movable, container = relation.split(" on ")
    #                         movable = movable.strip()
    #                         container = container.strip()
    #                         if container in movable_items:
    #                             movable_items[container].append(movable)

            
    #         # 存储房间数据
    #         room_data_dict[room] = {
    #             "containers": containers,
    #             "container_positions": container_positions,
    #             "movable_items": movable_items
    #         }
        
    #     # 遍历每个房间，分别处理其容器
    #     for room, room_data in room_data_dict.items():
    #         containers = room_data["containers"]
    #         container_positions = room_data["container_positions"]
    #         movable_items = room_data["movable_items"]
            
    #         # 如果当前房间没有容器，则跳过
    #         if not containers:
    #             print(f"No containers found in room {room}.")
    #             continue
            
    #         # 构建 LLM 输入提示信息
    #         prompt = f"""Below is the list of all containers in {room} along with their positions and any movable items placed on them:

    # Containers in {room}:
    # """
    #         for i, (container, pos) in enumerate(zip(containers, container_positions)):
    #             x, y = pos
    #             items_on_container = ", ".join(movable_items[container]) if movable_items[container] else "None"
    #             prompt += f"{i + 1}. Name: {container}, Position: ({x:.2f}, {y:.2f}), Movable Items: {items_on_container}\n"

    #         prompt += """
    # Please group these containers into distinct areas based on their spatial proximity and semantic similarity. For each area, provide:
    # 1. The list of containers included in the area.
    # 2. A brief description of the function of the area (e.g., "This area is used for cooking").
    # 3. Your reasoning for grouping these containers together (e.g., "These containers are close to each other and contain kitchen utensils").

    # Output your response in the following format, Do not add any other symbols:

    # Area 1:
    # - Containers: [list of container names]
    # - Function: [description of the area's function]
    # - Reasoning: [explanation of why these containers belong together]

    # Area 2:
    # - Containers: [list of container names]
    # - Function: [description of the area's function]
    # - Reasoning: [explanation of why these containers belong together]

    # ...
    # """

    #         # 初始化对话
    #         conversation_messages = [
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt}
    #         ]
    #         conversation = Conversation(conversation_messages)

    #         # 发送查询并接收 LLM 的响应
    #         response = self.send_query(conversation=conversation, model=self.model)

    #         # 解析 LLM 输出结果
    #         areas = {}
    #         current_area = None
    #         lines = response.strip().split("\n")
    #         for line in lines:
    #             if line.startswith("Area"):
    #                 current_area = line.split(":")[0].strip()
    #                 areas[current_area] = {"containers": [], "function": ""}
    #             elif line.startswith("- Containers:"):
    #                 containers_in_area = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
    #                 areas[current_area]["containers"] = [c.strip() for c in containers_in_area if c.strip()]
    #             elif line.startswith("- Function:"):
    #                 areas[current_area]["function"] = line.split(":")[1].strip()

    #         # 更新 obs["room_object_graph"] 中的节点属性
    #         for area_name, area_data in areas.items():
    #             for container in area_data["containers"]:
    #                 if container in graph.nodes:
    #                     graph.nodes[container]["area"] = area_name
    #                     graph.nodes[container]["area_function"] = area_data["function"]

    #         # 处理非容器节点（例如可移动物品和others）
    #         for obj in graph.successors(room):
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', "").strip()
    #             name = attrs.get('name', "").strip()
                
    #             # 获取物品位置
    #             obj_pos = attrs.get('pos_map', (0, 0, 0))
    #             if isinstance(obj_pos, tuple) and len(obj_pos) == 3:
    #                 obj_x, obj_y, _ = obj_pos
    #             else:
    #                 obj_x, obj_y = 0, 0
                
    #             if role == "movable":
    #                 relations = attrs.get('relation', [])
    #                 for relation in relations:
    #                     if "on" in relation:
    #                         _, container = relation.split(" on ")
    #                         container = container.strip()
    #                         if container in graph.nodes:
    #                             # 继承容器的区域信息
    #                             attrs["area"] = graph.nodes[container].get("area", "Unassigned Area")
    #                             attrs["area_function"] = graph.nodes[container].get("area_function", "Unknown function")
                
    #             elif role == 'others' and 'door' not in name and 'window' not in name and 'picture' not in name:
    #                 # 对于others角色，计算到所有有区域物品的距离，分配到最近的物品的区域
    #                 min_distance = float('inf')
    #                 nearest_item = None
                    
    #                 # 遍历所有有区域的物品
    #                 for other_obj in graph.successors(room):
    #                     other_attrs = graph.nodes[other_obj]
    #                     if "area" in other_attrs and other_attrs["area"] != "Unassigned Area":
    #                         # 获取物品位置
    #                         other_pos = other_attrs.get('pos_map', (0, 0, 0))
    #                         if isinstance(other_pos, tuple) and len(other_pos) == 3:
    #                             other_x, other_y, _ = other_pos
    #                         else:
    #                             other_x, other_y = 0, 0
                            
    #                         # 计算欧式距离
    #                         distance = ((obj_x - other_x)**2 + (obj_y - other_y)**2)**0.5
                            
    #                         if distance < min_distance:
    #                             min_distance = distance
    #                             nearest_item = other_obj
                    
    #                 if nearest_item:
    #                     # 继承最近物品的区域信息
    #                     attrs["area"] = graph.nodes[nearest_item].get("area", "Unassigned Area")
    #                     attrs["area_function"] = graph.nodes[nearest_item].get("area_function", "Unknown function")
    #                     attrs["nearest_item"] = nearest_item  # 记录最近的物品
    #                     attrs["distance_to_nearest"] = min_distance  # 记录距离

    #     return obs

    def determine_area_functions(self, obs):
        """
        从realworld.txt文件读取房间和容器信息，进行区域划分，并保存结果到real.txt
        """
        room_data_dict = {}
        
        try:
            with open('realworld.txt', 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError:
            print("realworld.txt文件未找到")
            return obs
        
        # 解析文件内容
        current_room = None
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测房间行
            if line.endswith(':'):
                current_room = line.replace(':', '').strip()
                room_data_dict[current_room] = {
                    "containers": [],
                    "container_positions": [],
                    "movable_items": {}
                }
                continue
                
            # 解析容器行 - 现在格式统一了，更容易解析
            if current_room and '(' in line and ')' in line and 'movable object:' in line:
                # 提取容器名称（括号前的部分）
                container_name = line.split('(')[0].strip()
                
                # 提取位置
                pos_start = line.find('(') + 1
                pos_end = line.find(')')
                pos_str = line[pos_start:pos_end]
                pos_parts = [p.strip() for p in pos_str.split(',')]
                
                try:
                    x = float(pos_parts[0])
                    y = float(pos_parts[1])
                except (ValueError, IndexError):
                    x, y = 0, 0
                    print(f"Warning: Invalid position format in line: {line}")
                    
                # 提取可移动物品
                movable_objects = []
                items_start = line.find('movable object:') + len('movable object:')
                items_str = line[items_start:].strip()
                
                # 处理物品字符串
                if items_str.lower() == 'none' or not items_str:
                    movable_objects = ["None"]
                else:
                    # 处理逗号分隔的物品列表，去除可能的空格和空字符串
                    items = [item.strip() for item in items_str.split(',') if item.strip()]
                    movable_objects.extend(items)
                
                # 添加到房间数据
                room_data_dict[current_room]["containers"].append(container_name)
                room_data_dict[current_room]["container_positions"].append([x, y])
                room_data_dict[current_room]["movable_items"][container_name] = movable_objects
                
                print(f"Added container: {container_name} to room {current_room}")
        
        # 打印读取结果进行验证
        print("\n=== 读取结果验证 ===")
        for room_name, room_data in room_data_dict.items():
            print(f"房间: {room_name}")
            print(f"容器数量: {len(room_data['containers'])}")
            for container, pos, items in zip(room_data['containers'], room_data['container_positions'], room_data['movable_items'].values()):
                print(f"  {container} {pos} -> {items}")
            print()
        
        # 处理每个房间
        all_results = []
        
        for room_name in sorted(room_data_dict.keys()):
            room_data = room_data_dict[room_name]
            containers = room_data["containers"]
            container_positions = room_data["container_positions"]
            movable_items = room_data["movable_items"]
            
            if not containers:
                print(f"No containers found in room {room_name}.")
                continue
            
            # 构建 LLM 输入提示信息
            prompt = f"""Below is the list of all containers in {room_name} along with their positions and any movable items placed on them:

    Containers in {room_name}:
    """
            for i, (container, pos) in enumerate(zip(containers, container_positions)):
                x, y = pos
                items_list = movable_items[container]
                # 如果物品列表是["None"]，显示"None"，否则显示具体物品
                if items_list == ["None"]:
                    items_on_container = "None"
                else:
                    items_on_container = ", ".join(items_list)
                
                prompt += f"{i + 1}. Name: {container}, Position: ({x:.2f}, {y:.2f}), Movable Items: {items_on_container}\n"

            prompt += """
    Please group these containers into distinct areas based on their spatial proximity and semantic similarity. For each area, provide:
    1. The list of containers included in the area.
    2. A brief description of the function of the area (e.g., "This area is used for cooking").
    3. Your reasoning for grouping these containers together (e.g., "These containers are close to each other and contain kitchen utensils").

    Output your response in the following format, Do not add any other symbols:

    Area 1:
    - Containers: [list of container names]
    - Function: [description of the area's function]
    - Reasoning: [explanation of why these containers belong together]

    Area 2:
    - Containers: [list of container names]
    - Function: [description of the area's function]
    - Reasoning: [explanation of why these containers belong together]

    ...
    """

            print(f"\nProcessing room: {room_name}")
            print(f"Number of containers: {len(containers)}")
            
            # 初始化对话
            conversation_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            conversation = Conversation(conversation_messages)

            # 发送查询并接收 LLM 的响应
            response = self.send_query(conversation=conversation, model=self.model)
            
            # 保存结果
            room_result = f"=== {room_name} ===\n{response}\n\n"
            all_results.append(room_result)
            
            print(f"Completed processing room: {room_name}")

        # 将所有结果保存到real.txt文件
        try:
            with open('real.txt', 'w', encoding='utf-8') as output_file:
                output_file.writelines(all_results)
            print(f"\nSuccessfully saved results for {len(room_data_dict)} rooms to real.txt")
            
            # 也打印结果到控制台
            print("\n=== 最终结果 ===")
            for result in all_results:
                print(result)
                
        except Exception as e:
            print(f"Error saving results to file: {e}")

        return obs

    # def determine_area_functions(self, obs):
    #     """
    #     根据图结构的层级关系获取房间及其物品，并进行区域划分。
    #     增加区域修正功能：
    #     1. 以每个容器节点位置为圆心，半径30的圆内检查区域归属
    #     2. 如果没有同区域节点但有其他区域节点，则归为附近最多的区域
    #     3. 如果完全孤立，则创建新区域
    #     """
    #     # 获取房间对象图
    #     graph = obs["room_object_graph"]
        
    #     # 获取所有房间节点
    #     rooms = list(graph.successors("root"))
        
    #     # 初始化房间数据结构
    #     room_data_dict = {}  # key: room_name, value: {"containers": [], "container_positions": [], "movable_items": {}}
        
    #     for room in rooms:
    #         # 获取房间中的所有物品
    #         objects = sorted(list(graph.successors(room)))
            
    #         containers = []
    #         container_positions = []
    #         movable_items = {}
            
    #         for obj in objects:
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', "").strip()
    #             name = attrs.get('name', "").strip()
                
    #             if role == "container":
    #                 # 提取 pos_map 属性
    #                 pos_map = attrs.get('pos_map')
    #                 if isinstance(pos_map, tuple) and len(pos_map) == 3:
    #                     x, y, _ = pos_map
    #                 else:
    #                     print(f"Invalid or missing position for node {name}, using default (0, 0, 0)")
    #                     x, y, _ = 0, 0, 0
                    
    #                 containers.append(name)
    #                 container_positions.append([x, y])
    #                 movable_items[name] = []  # 初始化容器对应的空列表
                    
    #                 # print(f"Room: {room}, Container: {name}, Position: ({x:.2f}, {y:.2f})")

    #         # 然后再次遍历所有物品，只处理可移动物品
    #         for obj in objects:
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', "").strip()
    #             name = attrs.get('name', "").strip()
                
    #             if role == "movable":
    #                 relations = attrs.get('relation', [])
    #                 for relation in relations:
    #                     if "on" in relation:
    #                         movable, container = relation.split(" on ")
    #                         movable = movable.strip()
    #                         container = container.strip()
    #                         if container in movable_items:
    #                             movable_items[container].append(movable)

            
    #         # 存储房间数据
    #         room_data_dict[room] = {
    #             "containers": containers,
    #             "container_positions": container_positions,
    #             "movable_items": movable_items
    #         }
        
    #     # 遍历每个房间，分别处理其容器
    #     for room, room_data in room_data_dict.items():
    #         containers = room_data["containers"]
    #         container_positions = room_data["container_positions"]
    #         movable_items = room_data["movable_items"]
            
    #         # 如果当前房间没有容器，则跳过
    #         if not containers:
    #             print(f"No containers found in room {room}.")
    #             continue
            
    #         # 构建 LLM 输入提示信息
    #         prompt = f"""Below is the list of all containers in {room} along with their positions and any movable items placed on them:

    # Containers in {room}:
    # """
    #         for i, (container, pos) in enumerate(zip(containers, container_positions)):
    #             x, y = pos
    #             items_on_container = ", ".join(movable_items[container]) if movable_items[container] else "None"
    #             prompt += f"{i + 1}. Name: {container}, Position: ({x:.2f}, {y:.2f}), Movable Items: {items_on_container}\n"

    #         prompt += """
    # Please group these containers into distinct areas based on their spatial proximity and semantic similarity. For each area, provide:
    # 1. The list of containers included in the area.
    # 2. A brief description of the function of the area (e.g., "This area is used for cooking").
    # 3. Your reasoning for grouping these containers together (e.g., "These containers are close to each other and contain kitchen utensils").

    # Output your response in the following format, Do not add any other symbols:

    # Area 1:
    # - Containers: [list of container names]
    # - Function: [description of the area's function]
    # - Reasoning: [explanation of why these containers belong together]

    # Area 2:
    # - Containers: [list of container names]
    # - Function: [description of the area's function]
    # - Reasoning: [explanation of why these containers belong together]

    # ...
    # """

    #         # 初始化对话
    #         conversation_messages = [
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt}
    #         ]
    #         conversation = Conversation(conversation_messages)

    #         # 发送查询并接收 LLM 的响应
    #         response = self.send_query(conversation=conversation, model=self.model)

    #         # 解析 LLM 输出结果
    #         areas = {}
    #         current_area = None
    #         lines = response.strip().split("\n")
    #         for line in lines:
    #             if line.startswith("Area"):
    #                 current_area = line.split(":")[0].strip()
    #                 areas[current_area] = {"containers": [], "function": ""}
    #             elif line.startswith("- Containers:"):
    #                 containers_in_area = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
    #                 areas[current_area]["containers"] = [c.strip() for c in containers_in_area if c.strip()]
    #             elif line.startswith("- Function:"):
    #                 areas[current_area]["function"] = line.split(":")[1].strip()

    #         # 更新 obs["room_object_graph"] 中的节点属性
    #         for area_name, area_data in areas.items():
    #             for container in area_data["containers"]:
    #                 if container in graph.nodes:
    #                     graph.nodes[container]["area"] = area_name
    #                     graph.nodes[container]["area_function"] = area_data["function"]

    #         # 处理非容器节点（例如可移动物品）
    #         for obj in graph.successors(room):
    #             attrs = graph.nodes[obj]
    #             if attrs.get('role', "").strip() == "movable":
    #                 relations = attrs.get('relation', [])
    #                 for relation in relations:
    #                     if "on" in relation:
    #                         _, container = relation.split(" on ")
    #                         container = container.strip()
    #                         if container in graph.nodes:
    #                             # 继承容器的区域信息
    #                             attrs["area"] = graph.nodes[container].get("area", "Unassigned Area")
    #                             attrs["area_function"] = graph.nodes[container].get("area_function", "Unknown function")


    #         print(f"\n=== 开始处理房间 {room} 的区域修正 ===")

    #         # 统计初始区域容器数量
    #         area_counts = {}
    #         for container in containers:
    #             area = graph.nodes[container].get("area", "")
    #             if area not in area_counts:
    #                 area_counts[area] = 0
    #             area_counts[area] += 1

    #         # 第一阶段：收集决策
    #         pending_corrections = []
    #         areas_to_delete = set()
    #         original_areas = {c: graph.nodes[c].get("area", "") for c in containers}

    #         for i, container in enumerate(containers):
    #             if container not in graph.nodes:
    #                 continue
                    
    #             current_area = original_areas[container]
    #             x1, y1 = container_positions[i]
                
    #             # 检查半径30单位内的容器
    #             same_area_neighbors = []
    #             other_area_neighbors = []  # 格式: (容器名, 区域, 距离)
                
    #             for j, other_container in enumerate(containers):
    #                 if i == j or other_container not in graph.nodes:
    #                     continue
                        
    #                 x2, y2 = container_positions[j]
    #                 distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    
    #                 if distance <= 30:
    #                     other_area = original_areas[other_container]
    #                     if other_area == current_area:
    #                         same_area_neighbors.append((other_container, distance))
    #                     elif other_area:
    #                         other_area_neighbors.append((other_container, other_area, distance))
                
    #             # 决策逻辑
    #             if area_counts[current_area] > 1:  # 多容器区域
    #                 if same_area_neighbors or not other_area_neighbors:
    #                     print(f"✅ 多容器区域 {current_area} 的 {container}: 保持")
    #                     continue
    #                 else:
    #                     other_area_neighbors.sort(key=lambda x: x[2])
    #                     closest_area = other_area_neighbors[0][1]
    #                     print(f"🔄 多容器区域 {current_area} 的 {container}: 修正到 {closest_area}")
    #                     pending_corrections.append((container, current_area, closest_area, "多容器区域修正", False))
    #             else:  # 单容器区域
    #                 if not other_area_neighbors:
    #                     print(f"✅ 单容器区域 {current_area} 的 {container}: 保持")
    #                     continue
    #                 else:
    #                     # 找出所有邻近的单容器区域（包括自己）
    #                     candidate_areas = {area for _, area, _ in other_area_neighbors 
    #                                     if area_counts[area] == 1}
    #                     candidate_areas.add(current_area)
                        
    #                     # 选择序号最小的区域作为目标
    #                     target_area = min(candidate_areas)
                        
    #                     if current_area != target_area:
    #                         # 判断是否单容器互相合并（目标区域也是单容器）
    #                         if area_counts[target_area] == 1:
    #                             action_type = "单容器区域合并（功能重置为None）"
    #                             set_function_to_none = True
    #                         else:
    #                             action_type = "单容器区域合并"
    #                             set_function_to_none = False
                            
    #                         print(f"🗑️ 单容器区域 {current_area} 的 {container}: 合并到 {target_area}（{action_type}）")
    #                         pending_corrections.append((container, current_area, target_area, action_type, set_function_to_none))
    #                         areas_to_delete.add(current_area)
    #                     else:
    #                         print(f"✅ 单容器区域 {current_area} 的 {container}: 作为最小序号区域保持")

    #         # 第二阶段：应用修正
    #         print("\n=== 应用修正 ===")
    #         corrected_nodes = []

    #         for container, old_area, new_area, action, set_function_to_none in pending_corrections:
    #             # 更新容器区域
    #             graph.nodes[container]["area"] = new_area
                
    #             # 处理功能描述
    #             if set_function_to_none:
    #                 graph.nodes[container]["area_function"] = None  # 单容器互并时重置
    #             else:
    #                 # 继承目标区域的功能（如果存在）
    #                 if new_area in graph.nodes:
    #                     graph.nodes[container]["area_function"] = graph.nodes[new_area].get("area_function", "")
    #                 else:
    #                     graph.nodes[container]["area_function"] = ""  # 默认值
                
    #             # 更新关联物品
    #             for item in movable_items.get(container, []):
    #                 if item in graph.nodes:
    #                     graph.nodes[item]["area"] = new_area
    #                     graph.nodes[item]["area_function"] = graph.nodes[container].get("area_function", "")
                
    #             corrected_nodes.append((container, old_area, new_area, action))
    #             print(f"应用修正: {container} {old_area} → {new_area} ({action})")

    #         # 清理空区域
    #         for area in areas_to_delete:
    #             remaining_containers = [c for c in containers if graph.nodes[c].get("area") == area]
    #             if not remaining_containers:
    #                 print(f"清理: 删除空区域 {area}")

    #         # 打印汇总报告
    #         print("\n=== 区域修正汇总 ===")
    #         if not corrected_nodes:
    #             print("没有需要修正的容器")
    #         else:
    #             print("以下容器被重新分配区域:")
    #             for node, old_area, new_area, reason in corrected_nodes:
    #                 items = movable_items.get(node, [])
    #                 items_str = f" (含物品: {', '.join(items)})" if items else ""
    #                 print(f"- {node}{items_str}: {old_area} → {new_area} ({reason})")

    #         if areas_to_delete:
    #             print("\n以下区域将被删除:")
    #             for area in areas_to_delete:
    #                 print(f"- {area}")
    #         else:
    #             print("\n没有区域被删除")

    #         print("\n=== 处理其他物体（others）的区域分配 ===")

    #         # 获取房间内所有物体
    #         all_objects = list(graph.successors(room))

    #         for obj in all_objects:
    #             attrs = graph.nodes[obj]
    #             role = attrs.get('role', '').strip()
    #             name = attrs.get('name', '').lower().strip()
                
    #             # 只处理role为others且不是门/窗的物体
    #             if role == 'others' and 'door' not in name and 'window' not in name and 'picture' not in name:
    #                 obj_pos = attrs.get('pos_map', (0, 0, 0))[:2]  # 只取(x,y)
                    
    #                 # 寻找最近的容器或可移动物品
    #                 closest_item = None
    #                 min_distance = float('inf')
    #                 closest_area = None
    #                 closest_function = None
                    
    #                 # 检查所有容器和可移动物品
    #                 for other in all_objects:
    #                     if other == obj:
    #                         continue
                            
    #                     other_attrs = graph.nodes[other]
    #                     other_role = other_attrs.get('role', '').strip()
                        
    #                     # 只考虑容器或可移动物品
    #                     if other_role in ['container', 'movable'] and 'area' in other_attrs:
    #                         other_pos = other_attrs.get('pos_map', (0, 0, 0))[:2]
    #                         distance = ((obj_pos[0]-other_pos[0])**2 + (obj_pos[1]-other_pos[1])**2)**0.5
                            
    #                         if distance < min_distance:
    #                             min_distance = distance
    #                             closest_item = other
    #                             closest_area = other_attrs['area']
    #                             closest_function = other_attrs.get('area_function', '')
                    
    #                 # 如果找到最近的物品，则更新区域属性
    #                 if closest_item:
    #                     graph.nodes[obj]['area'] = closest_area
    #                     graph.nodes[obj]['area_function'] = closest_function
    #                     print(
    #                         f"更新: {obj} (others) → 区域 {closest_area}\n"
    #                         f"  最近参考: {closest_item} (距离: {min_distance:.2f})\n"
    #                         f"  功能描述: '{closest_function}'"
    #                     )
    #                 else:
    #                     print(f"警告: {obj} (others) 未找到附近参考物品，保持原区域")

    #         # 确保更新obs中的图结构
    #         obs["room_object_graph"] = graph

    #         # 写入area.txt文件（包含容器和物品）
    #         with open("area.txt", "a") as f:
    #             for obj in graph.successors(room):
    #                 attrs = graph.nodes[obj]
    #                 name = attrs.get('name', "").strip()
    #                 area = attrs.get('area', "").strip()
    #                 if name and area:
    #                     f.write(f"name='{name}' {room} {area}\n")
    #     return obs

    def determine_room_functions(self, obs):
        """
        根据容器数量决定房间划分方式：
        - 容器>5：按容器分组重新划分房间
        - 容器≤5：直接判断整个房间类型
        保持相同的输出格式
        """
        # 获取原始房间对象图
        original_graph = obs["room_object_graph"]
        vor_graph = obs["seperate_voronoi_graph"]
        # 获取所有原始房间节点
        original_rooms = list(original_graph.successors("root"))
        
        # 初始化数据结构收集所有房间信息
        all_containers_data = []
        room_container_counts = {}  # 记录每个原始房间的容器数量
        
        # 收集所有容器及其物品信息
        for room in original_rooms:
            objects = sorted(list(original_graph.successors(room)))
            container_count = 0
            
            for obj in objects:
                attrs = original_graph.nodes[obj]
                role = attrs.get('role', "").strip()
                name = attrs.get('name', "").strip()
                
                if role == "container":
                    container_count += 1
                    pos_map = attrs.get('pos_map')
                    if isinstance(pos_map, tuple) and len(pos_map) == 3:
                        x, y, _ = pos_map
                    else:
                        x, y, _ = 0, 0, 0
                    
                    items_on_container = []
                    for other_obj in objects:
                        other_attrs = original_graph.nodes[other_obj]
                        other_role = other_attrs.get('role', "").strip()
                        other_name = other_attrs.get('name', "").strip()
                        
                        if other_role == "movable":
                            relations = other_attrs.get('relation', [])
                            for relation in relations:
                                if "on" in relation:
                                    _, container = relation.split(" on ")
                                    if container.strip() == name:
                                        items_on_container.append({
                                            "name": other_name,
                                            "attrs": other_attrs.copy()
                                        })
                    
                    container_data = {
                        "name": name,
                        "original_room": room,
                        "position": (x, y),
                        "items": items_on_container,
                        "attrs": attrs.copy()
                    }
                    all_containers_data.append(container_data)
            
            room_container_counts[room] = container_count
        
        # 创建新的图结构
        new_graph = nx.DiGraph()
        new_graph.add_node("root", type="root")
        
        # 根据容器数量决定处理方式
        for room in original_rooms:
            if room_container_counts[room] > 5:
                # 容器数量>5的情况 - 按容器分组
                room_containers = [c for c in all_containers_data if c["original_room"] == room]
                
                # 构建LLM提示信息（多容器分组）
                prompt = f"""Below is the list of all containers in {room} along with their positions and items:

    Containers in {room}:
    """
                for i, container in enumerate(room_containers):
                    items_str = ", ".join([item["name"] for item in container["items"]]) if container["items"] else "None"
                    prompt += f"{i + 1}. Name: {container['name']}, Position: ({container['position'][0]:.2f}, {container['position'][1]:.2f}), Items: {items_str}\n"

                prompt += """
    Please group these containers into distinct rooms based on their spatial proximity and semantic similarity. For each room, provide:
    1. The list of containers included in the room.
    2. The type of room (Only one specific type can be given,"or" shall not appear, e.g., kitchen, living room, bedroom).
    3. Your reasoning for grouping these containers together.

    Output your response in the following strict format:

    Room Type: kitchen
    - Containers: [container1, container2, ...]
    - Reasoning: [your reasoning]

    Room Type: living room
    - Containers: [container3, container4, ...]
    - Reasoning: [your reasoning]

    ...
    """
            else:
                # 容器数量≤5的情况 - 直接判断整个房间类型
                room_containers = [c for c in all_containers_data if c["original_room"] == room]
                
                # 构建LLM提示信息（整体房间判断）
                prompt = f"""Below is the list of all containers in {room} along with their items:

    Room {room} contains:
    """
                for i, container in enumerate(room_containers):
                    items_str = ", ".join([item["name"] for item in container["items"]]) if container["items"] else "None"
                    prompt += f"- {container['name']} with items: {items_str}\n"

                prompt += """
    Please determine what type of room this is based on the containers and items. Provide:
    1. The type of room (e.g., kitchen, living room, bedroom).
    2. Your reasoning for this classification.

    Output your response in the following strict format:

    Room Type: bedroom
    - Reasoning: [your reasoning]
    """
            
            # 发送查询并接收LLM的响应
            conversation_messages = [
                {"role": "system", "content": "You are a helpful assistant that specializes in spatial organization and room classification."},
                {"role": "user", "content": prompt}
            ]
            response = self.send_query(conversation=Conversation(conversation_messages), model=self.model)
            
            # 解析LLM的输出
            lines = response.strip().split("\n")
            room_data = {}
            
            if room_container_counts[room] > 5:
                # 多容器分组解析
                current_room_type = None
                for line in lines:
                    if line.startswith("Room Type:"):
                        current_room_type = line.split(":")[1].strip()
                        room_data[current_room_type] = {"containers": [], "reasoning": ""}
                    elif line.startswith("- Containers:"):
                        containers = line.split(":")[1].strip().strip("[]").replace("'", "").split(", ")
                        room_data[current_room_type]["containers"] = [c.strip() for c in containers if c.strip()]
                    elif line.startswith("- Reasoning:"):
                        room_data[current_room_type]["reasoning"] = line.split(":")[1].strip()
                
                # 构建新图结构（多容器分组）
                for room_type, data in room_data.items():
                    room_node = f"{room_type}_{len(list(new_graph.successors('root'))) + 1}"
                    new_graph.add_node(room_node, type="room", room_type=room_type, reasoning=data["reasoning"])
                    new_graph.add_edge("root", room_node)
                    
                    for container_name in data["containers"]:
                        container_data = next((c for c in room_containers if c["name"] == container_name), None)
                        if container_data:
                            container_attrs = container_data["attrs"].copy()
                            container_attrs.update({
                                "type": "container",
                                "position": container_data["position"],
                                "role": "container"
                            })
                            new_graph.add_node(container_name, **container_attrs)
                            new_graph.add_edge(room_node, container_name)
                            
                            for item in container_data["items"]:
                                item_attrs = item["attrs"].copy()
                                item_attrs.update({
                                    "type": "movable",
                                    "role": "movable"
                                })
                                new_graph.add_node(item["name"], **item_attrs)
                                new_graph.add_edge(room_node, item["name"])
            else:
                # 单房间类型解析
                room_type = None
                reasoning = ""
                for line in lines:
                    if line.startswith("Room Type:"):
                        room_type = line.split(":")[1].strip()
                    elif line.startswith("- Reasoning:"):
                        reasoning = line.split(":")[1].strip()
                
                if room_type:
                    room_node = f"{room_type}_{len(list(new_graph.successors('root'))) + 1}"
                    new_graph.add_node(room_node, type="room", room_type=room_type, reasoning=reasoning)
                    new_graph.add_edge("root", room_node)
                    
                    for container_data in room_containers:
                        container_attrs = container_data["attrs"].copy()
                        container_attrs.update({
                            "type": "container",
                            "position": container_data["position"],
                            "role": "container"
                        })
                        new_graph.add_node(container_data["name"], **container_attrs)
                        new_graph.add_edge(room_node, container_data["name"])
                        
                        for item in container_data["items"]:
                            item_attrs = item["attrs"].copy()
                            item_attrs.update({
                                "type": "movable",
                                "role": "movable"
                            })
                            new_graph.add_node(item["name"], **item_attrs)
                            new_graph.add_edge(room_node, item["name"])

        # 更新obs中的图结构
        obs["reorganized_room_graph"] = new_graph
    
        # 新增部分：保存房间-物品信息到文件
        with open("room_seg.txt", "w") as f:
            # 遍历所有房间节点
            for room_node in new_graph.successors("root"):
                room_type = new_graph.nodes[room_node]["room_type"]
                items_info = []
                
                # 遍历房间下的所有节点（容器和物品）
                for node in new_graph.successors(room_node):
                    node_type = new_graph.nodes[node].get("type", "")
                    closest_vor = new_graph.nodes[node].get("closest_vor_node", "unknown")
                    
                    # 记录所有节点（包括容器和可移动物品）
                    items_info.append(f"{node}:{closest_vor}")
                
                # 写入文件
                if items_info:
                    f.write(f"{room_type}: {' '.join(items_info)}\n")
        
        return obs

