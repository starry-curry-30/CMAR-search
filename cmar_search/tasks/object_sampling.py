# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum, IntEnum
from pathlib import Path

import igibson
import numpy as np
import pybullet as p
from bddl.object_taxonomy import ObjectTaxonomy
from cachetools import cached
from cachetools.keys import hashkey
from igibson import object_states
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.object_states.utils import clear_cached_states, sample_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import (get_ig_category_path,
                                        get_ig_model_path, get_ig_scene_path)
from igibson.utils.utils import restoreState
from IPython import embed

from cmar_search.llm.sbert import SentenceBERT
from cmar_search.tasks.patched_scene import REPLACED_CATEGORIES
from cmar_search.utils.constants import PACKAGE_DIR, TEST_SCENES, TRAINING_SCENES


def load_object_distribution_prior():
    # load object placement json
    obj_placement_file = os.path.join(PACKAGE_DIR, "configs", "object_placement.json")
    obj_placement_ig_pt = json.load(open(obj_placement_file, "r"))
    object_distribution_prior = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    # go through all relation keys and replace with object states
    for room_type, furniture_dict in obj_placement_ig_pt.items():
            for furniture, obj_dict in furniture_dict.items():
                for obj, relation_dict in obj_dict.items():
                    for relation, prob in relation_dict.items():
                        if relation == "ON_TOP":
                            object_distribution_prior[room_type][furniture][object_states.OnTop][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        elif relation == "INSIDE":
                            object_distribution_prior[room_type][furniture][object_states.Inside][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        elif relation == "UNDER":
                            object_distribution_prior[room_type][furniture][object_states.Under][obj] = obj_placement_ig_pt[room_type][furniture][obj][relation]
                        else:    
                            raise NotImplementedError(f"Relation {relation} not implemented")
    return object_distribution_prior


OBJECT_DISTRIBUTION_PRIOR = load_object_distribution_prior()


def create_new_object(env, scene, category, obj_number, in_rooms=None, rendering_params=None, scale=None, bounding_box=None):
    """Based on InteractiveIndoorScene.__init__"""
    
    object_name = f"{category}_{obj_number}"
    model = "random"

    # Non-robot object
    # Do not load these object categories (can blacklist building structures as well)
    if scene.not_load_object_categories is not None and category in scene.not_load_object_categories:
        return

    # An object can in multiple rooms, seperated by commas,
    # or None if the object is one of the walls, floors or ceilings
    # in_rooms = link.attrib.get("room", None)
    if in_rooms is not None:
        in_rooms = in_rooms.split(",")

    # This object is not located in one of the selected rooms, skip
    if scene.load_room_instances is not None and len(set(scene.load_room_instances) & set(in_rooms)) == 0:
        return

    category_path = get_ig_category_path(category)
    assert len(os.listdir(category_path)) != 0, "No models in category folder {}".format(category_path)

    if model == "random":
        model = env.np_random.choice(os.listdir(category_path))         

    model_path = get_ig_model_path(category, model)
    filename = os.path.join(model_path, model + ".urdf")

    if (bounding_box is not None) and (scale is not None):
        raise Exception("You cannot define both scale and bounding box size for a URDFObject")
    if scale is not None and isinstance(scale, float):
        scale = np.array([scale, scale, scale])
    elif bounding_box is not None:
        bounding_box = np.array(bounding_box)
    else:
        scale = np.array([1.0, 1.0, 1.0])

    bddl_object_scope = None
    fixed_base = False

    obj = URDFObject(filename,
                     name=object_name,
                     category=category,
                     model_path=model_path,
                     bounding_box=bounding_box,
                     scale=scale,
                     fixed_base=fixed_base,
                     avg_obj_dims=scene.avg_obj_dims[category],
                     in_rooms=in_rooms,
                     texture_randomization=scene.texture_randomization,
                     overwrite_inertial=True,
                     scene_instance_folder=scene.scene_instance_folder,
                     bddl_object_scope=bddl_object_scope,
                     merge_fixed_links=scene.merge_fixed_links,
                     rendering_params=rendering_params,
                     fit_avg_dim_volume=True)
    env.simulator.import_object(obj)
    return obj


def match_furniture(given_furniture, spawnable_object_distribution):
    # match furniture objects from the prior graph to igibson furniture objects - mix of sbert and human judgement
    object_taxonomy = ObjectTaxonomy()
    all_igibson_furniture = [ic for c in object_taxonomy.get_descendants('furniture.n.01') for ic in object_taxonomy.get_igibson_categories(c)]
    all_igibson_furniture += ["shelf", "shelving_unit", "counter_top", "kitchen_counter"]
    for k, v in REPLACED_CATEGORIES.items():
        if isinstance(v, dict):
            v = list(v.values())
        else:
            v = [v]
        all_igibson_furniture += v
    all_igibson_furniture = list(set(all_igibson_furniture))
        
    furniture_mapping = get_sbert_matching(given_furniture, all_igibson_furniture, score_thresh=0.6, closest_k=50)
    furniture_mapping["shelf"].append("shelving_unit")
    furniture_mapping["shelving_unit"].append("shelf")
    furniture_mapping["dresser"].append("bottom_cabinet_no_top")
    furniture_mapping["dresser"].append("bottom_cabinet")
    furniture_mapping["dresser"].append("top_cabinet")
    furniture_mapping["top_cabinet"].append("dresser")

    for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
        for furniture, relation_dict in furniture_dict.items():
            if furniture in furniture_mapping:
                for igibson_furniture in furniture_mapping[furniture]:
                    if (furniture != igibson_furniture):
                        spawnable_object_distribution[room_type][igibson_furniture].update(spawnable_object_distribution[room_type][furniture])
    return spawnable_object_distribution


def match_rooms(given_room_types, spawnable_object_distribution):
    """mapping from igibson gt room label to room types in the prior graph"""
    human_room_mapping = {"bathroom": "bathroom", 
                          "bedroom": "bedroom",
                          "childs_room": "bedroom",
                          "closet": None,
                          "corridor": None,
                          "dining_room": "living_room",
                          "empty_room": None,
                          "exercise_room": None,
                          "garage": None,
                          "home_office": None,
                          "kitchen": "kitchen",
                          "living_room": "living_room",
                          "lobby": None,
                          "pantry_room": None,
                          "playroom": None,
                          "staircase": None,
                          "storage_room": None,
                          "television_room": "living_room",
                          "utility_room": None,
                          "balcony": None,
                          "library": None,
                          "auditorium": None,
                          "undefined": None}
    inverse_room_mapping = {r: [] for r in given_room_types}
    for k, v in human_room_mapping.items():
        if v is not None:
            inverse_room_mapping[v].append(k)
    
    for room_type in given_room_types:
        for matched_room in inverse_room_mapping[room_type]:
            if (matched_room != room_type):
                spawnable_object_distribution[matched_room].update(spawnable_object_distribution[room_type])
    return spawnable_object_distribution


@cached(cache={})
def match_distribution_against_objs():
    # 初始化SentenceBERT模型，用于计算对象之间的相似度
    sbert = SentenceBERT()
    
    # 获取所有可生成对象（这里以"apple"类别为例，实际应用中应更广泛）
    # spawnable_objs = env.task.all_object_categories
    spawnable_objs = [p.name for p in Path(get_ig_category_path("apple")).parent.iterdir()]
    
    # 初始化集合来存储已知的对象、家具和房间类型
    given_objs = set()
    given_furniture = set()
    given_room_types = set()
    
    # 遍历OBJECT_DISTRIBUTION_PRIOR中的每个房间类型及其包含的家具和关系
    for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
        given_room_types.add(room_type)  # 添加当前房间类型到集合
        for furniture, relation_dict in furniture_dict.items():
            given_furniture.add(furniture)  # 添加当前家具到集合
            for relation, relation_obj_dict in relation_dict.items():
                given_objs.update(list(relation_obj_dict.keys()))  # 更新给定对象集合
    
    # 将集合转换为列表以便后续处理
    given_objs = list(given_objs)
    given_furniture = list(given_furniture)
    given_room_types = list(given_room_types)

    # 计算可生成对象与给定对象之间的相似度矩阵
    obj_sim = sbert.compute_cooccurrence_bipartite(["A " + obj for obj in spawnable_objs], ["A " + obj for obj in given_objs])
    
    # 对于每个可生成对象，找到与其最相似的50个给定对象（基于相似度阈值0.7）
    closest_k_given = np.argsort(obj_sim, axis=1)[:,-50:]
    
    # 初始化嵌套字典来存储最终的可生成对象分布
    spawnable_object_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    
    # 遍历每个可生成对象及其最相似的给定对象
    for spawnable_idx, given_idcs in enumerate(closest_k_given):
        spawnable_obj = spawnable_objs[spawnable_idx]  # 当前可生成对象
        for given_idx in given_idcs:
            given_obj = given_objs[given_idx]  # 当前最相似的给定对象
            # 如果对象不相同且相似度大于0.7，则更新分布
            if (given_obj != spawnable_obj) and (obj_sim[spawnable_idx, given_idx] > 0.7):                
                # 将匹配的可生成对象添加到所有相关的房间类型、家具和关系中
                for room_type, furniture_dict in OBJECT_DISTRIBUTION_PRIOR.items():
                    for furniture, relation_dict in furniture_dict.items():
                        for relation, relation_obj_dict in relation_dict.items():
                            if given_obj in relation_obj_dict.keys():
                                # 更新可生成对象分布
                                spawnable_object_distribution[room_type][furniture][relation][spawnable_obj] = relation_obj_dict[given_obj]
    
    # 根据家具进一步匹配并调整可生成对象分布
    spawnable_object_distribution = match_furniture(given_furniture, spawnable_object_distribution)
    
    # 根据房间类型进一步匹配并调整可生成对象分布
    spawnable_object_distribution = match_rooms(given_room_types, spawnable_object_distribution)

    return spawnable_object_distribution  # 返回最终的可生成对象分布

def relation_to_str(relation):
    return str(relation).split('.')[-1].split("'")[0]


def get_sbert_matching(categories, possible_categories, score_thresh: float = 0.0, closest_k: int = 0):
    from cmar_search.llm.sbert import SentenceBERT
    sbert = SentenceBERT()
    sim = sbert.compute_cooccurrence_bipartite(categories, possible_categories)
    closest_categories = np.argsort(sim, axis=1)[:,-closest_k:]
    
    mapping = {c: [] for c in categories}
    
    for i, close_cat_idcs in enumerate(closest_categories):
        for close_cat_idx in close_cat_idcs:
            if sim[i, close_cat_idx] > score_thresh:
                # print(categories[i], possible_categories[close_cat_idx], sim[i, close_cat_idx])
                mapping[categories[i]].append(possible_categories[close_cat_idx])
    return mapping


def fast_set_value(state, other, new_value, use_ray_casting_method=False):
    """
    Based on object_states.Inside._set_value(), which does 10x100 sampling attemps, thereby being super slow if it's not possible to spawn 
    a relation -> reduce to 1x50 sampling attempts
    """
    state_id = p.saveState()
    
    if isinstance(state, object_states.Inside):
        prefix = "inside"
    elif isinstance(state, object_states.OnTop):
        prefix = "onTop"
    else:
        raise NotImplementedError(type(state))

    for _ in range(1):
        sampling_success = sample_kinematics(prefix, state.obj, other, new_value, use_ray_casting_method=use_ray_casting_method, max_trials=50)
        if sampling_success:
            clear_cached_states(state.obj)
            clear_cached_states(other)
            if (other.category != "shelf") and (state.get_value(other) != new_value):
                sampling_success = False
            if igibson.debug_sampling:
                print("Inside checking", sampling_success)
                embed()
        if sampling_success:
            break
        else:
            restoreState(state_id)

    p.removeState(state_id)

    return sampling_success
    
    
def is_receptacle(obj_category):
    if obj_category in ["door", "window"]:
        return False
    elif obj_category in ["shelf", "shelving_unit"]:
        return True
    object_taxonomy = ObjectTaxonomy()
    n = object_taxonomy.get_class_name_from_igibson_category(obj_category)
    return n and "openable" in object_taxonomy.get_abilities(n)
    

def add_objects_from_our_distribution(env):
    env.scene.force_wakeup_scene_objects()
    
    # 添加功能：记录所有物品的名称和ID
    def save_all_object_ids(scene):
        with open("object_id.txt", "w") as f:
            for obj in scene.get_objects():
                body_ids = obj.get_body_ids()
                if body_ids:  # 确保对象有有效的body ID
                    f.write(f"{obj.name} {body_ids[0]}\n")
    
    # 首先保存场景中原有的所有物品ID
    save_all_object_ids(env.scene)
    
    # add all igibson objects that match the objects in the prior graph to the relations
    spawnable_object_distribution_real = match_distribution_against_objs()

    # to make object names unique
    obj_number = 10000
    new_objects = []
    new_relations = {}
    
    # Create a dictionary to store object positions for saving
    object_position_data = {}
    
    for existing_obj in env.scene.get_objects():
        room_type = env.scene.get_room_type_by_point(existing_obj.get_position()[:2])
        prior_relations = spawnable_object_distribution_real.get(room_type, {}).get(existing_obj.category, {})
        
        possible_objects = set()
        for objects in prior_relations.values():
            possible_objects.update(set(objects.keys()))
        possible_objects_wo_metadata = set(possible_objects) - set(env.scene.avg_obj_dims.keys())
        for o in possible_objects_wo_metadata:
            del possible_objects[o]
            
        possible_objects = sorted(possible_objects)

        if possible_objects:
            r = set([object_states.Inside] + list(prior_relations.keys())) if is_receptacle(existing_obj.category) else list(prior_relations.keys())
            relation = env.np_random.choice(sorted(r, key=lambda x: str(x)))

            max_tries = min(5, len(possible_objects))
            i = 0
            while True:
                new_obj_category = env.np_random.choice(list(possible_objects))
                if (np.prod(env.scene.avg_obj_dims[new_obj_category]["size"]) > 0.9 * np.prod(env.scene.avg_obj_dims.get(existing_obj.category, {"size": np.inf})["size"])):
                    possible_objects.remove(new_obj_category)
                    i += 1
                    continue
                
                while f"{new_obj_category}_{obj_number}" in env.scene.objects_by_name.keys():
                    obj_number += 1
                    
                new_obj = create_new_object(env=env, 
                                          scene=env.scene,
                                          category=new_obj_category,
                                          in_rooms=room_type, 
                                          obj_number=obj_number,
                                          scale=None,
                                          bounding_box=None)
                obj_number += 1
                try:
                    if relation in [object_states.OnTop, object_states.Inside]:
                        assert fast_set_value(state=new_obj.states[relation], other=existing_obj, new_value=True, use_ray_casting_method=True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
                    else:
                        assert new_obj.states[relation].set_value(existing_obj, True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
                    
                    new_objects.append(new_obj)
                    new_relations[new_obj.name] = (existing_obj, relation)
                    print(f"Spawned new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}")
                    
                    # Store position data for saving
                    pos = new_obj.get_position()
                    object_position_data[existing_obj.name] = (
                        new_obj_category,
                        new_obj.get_body_ids()[0],  # since we incremented after creating
                        pos[0], pos[1], pos[2]
                    )
                    
                    # 每次成功创建新对象后，更新object_id.txt文件
                    with open("object_id.txt", "a") as f:
                        body_ids = new_obj.get_body_ids()
                        if body_ids:
                            f.write(f"{new_obj.name} {body_ids[0]}\n")
                    
                    break
                except:
                    print(f"Failed to spawn new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}")                 
                    set_base_values_with_z(new_obj.get_body_ids()[0], [200, 200 + (obj_number - 10000), 0.0], z=0.05)
                    env.scene.remove_object(new_obj)
                    
                    possible_objects.remove(new_obj_category)
                    i += 1
                if i >= max_tries:
                    print("Unable to spawn any object for this object/relation combo. Skipping it.")
                    break
        else:
            print(f"{room_type}/{existing_obj.name}: no possible objects to spawn")
    
    # Save the generated objects to a file
    save_object_relations(object_position_data)
    
    return new_objects, new_relations

def save_object_relations(object_data):
    """Save object relations to a file in the specified format."""
    with open("object_relations.txt", "w") as f:
        f.write("### Batch_1 ###\n")
        for container_name, (obj_category, obj_id, x, y, z) in object_data.items():
            f.write(f"{container_name}: {obj_category},{obj_id},{x},{y},{z}\n")



# def add_objects_from_our_distribution(env):
#     env.scene.force_wakeup_scene_objects()
    
#     spawnable_object_distribution_real = match_distribution_against_objs()  # 得到物品集
    
#     obj_number = 10000
#     new_objects = []
#     new_relations = {}
#     all_new_objects_info = []  # 存储所有新生成对象的信息

#     for existing_obj in env.scene.get_objects():
#         room_type = env.scene.get_room_type_by_point(existing_obj.get_position()[:2])
#         prior_relations = spawnable_object_distribution_real.get(room_type, {}).get(existing_obj.category, {})
        
#         possible_objects = set()
#         for objects in prior_relations.values():
#             possible_objects.update(set(objects.keys()))
#         possible_objects_wo_metadata = set(possible_objects) - set(env.scene.avg_obj_dims.keys())
#         for o in possible_objects_wo_metadata:
#             possible_objects.remove(o)
            
#         if not possible_objects:  # 检查是否有任何可能的对象
#             print(f"No possible objects found for {existing_obj.name} in room type {room_type}.")
#             continue
            
#         possible_objects = sorted(possible_objects)

#         successful_additions = 0  # 新增：记录已成功添加的新对象数量
#         possible_objects_set = set(possible_objects)  # 使用集合来方便去重
#         max_new_objects = min(2, len(possible_objects_set))  # 如果可能的对象少于2个，则生成的数量也相应减少
        
#         batch_counter = 0  # 计数器用于决定分配到哪个批次
        
#         while successful_additions < max_new_objects and len(possible_objects_set) > 0:
#             if not possible_objects_set:  # 再次检查集合是否为空
#                 print("No more possible objects to spawn for this existing object.")
#                 break  # 如果集合为空，则跳出循环
            
#             r = set([object_states.Inside] + list(prior_relations.keys())) if is_receptacle(existing_obj.category) else list(prior_relations.keys())
#             relation = env.np_random.choice(sorted(r, key=lambda x: str(x)))
            
#             while True:
#                 if not possible_objects_set:  # 再次检查集合是否为空
#                     print("No more possible objects to choose from in the current iteration.")
#                     break
                
#                 new_obj_category = env.np_random.choice(list(possible_objects_set))  # 从集合中随机选择以保证唯一性
                
#                 while f"{new_obj_category}_{obj_number}" in env.scene.objects_by_name.keys():
#                     obj_number += 1
                
#                 new_obj = create_new_object(env=env, 
#                                             scene=env.scene,
#                                             category=new_obj_category,
#                                             in_rooms=room_type, 
#                                             obj_number=obj_number,
#                                             scale=None,
#                                             bounding_box=None)
#                 obj_number += 1
                
#                 try:
#                     if relation in [object_states.OnTop, object_states.Inside]:
#                         assert fast_set_value(state=new_obj.states[relation], other=existing_obj, new_value=True, use_ray_casting_method=True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
#                     else:
#                         assert new_obj.states[relation].set_value(existing_obj, True), f"{existing_obj.category} > {relation_to_str(relation)} > {new_obj_category}"
                    
#                     new_objects.append(new_obj)
#                     new_relations[new_obj.name] = (existing_obj, relation)

#                     position = new_obj.get_position()
#                     obj_id = new_obj.get_body_ids()[0]  # 获取对象ID
                    
#                     # 根据计数器决定将新生成的对象分配到哪个批次
#                     if batch_counter == 0:
#                         all_new_objects_info.append(('batch_1', existing_obj.name, new_obj_category, obj_id, position))
#                     else:
#                         all_new_objects_info.append(('batch_2', existing_obj.name, new_obj_category, obj_id, position))
                    
#                     print(f"Spawned new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}, ID: {obj_id}")
#                     successful_additions += 1  # 成功添加后增加计数器
                    
#                     possible_objects_set.remove(new_obj_category)  # 添加成功后从集合中移除该对象以确保不会重复添加
#                     batch_counter = (batch_counter + 1) % 2  # 切换批次计数器
#                     break  # 成功后退出内层循环
#                 except Exception as e:
#                     print(f"Failed to spawn new object: {room_type}/{existing_obj.name} > {relation_to_str(relation)} > {new_obj_category}, Error: {e}")
#                     set_base_values_with_z(new_obj.get_body_ids()[0], [200, 200 + (obj_number - 10000), 0.0], z=0.05)
#                     env.scene.remove_object(new_obj)
                    
#                     possible_objects_set.remove(new_obj_category)
            
#             if successful_additions >= max_new_objects or len(possible_objects_set) == 0:
#                 break  # 达到目标或者没有更多可能对象时退出外层循环

#     # 分配到两批
#     batch_1, batch_2 = [], []

#     for item in all_new_objects_info:
#         batch_name, existing_obj_name, new_obj_category, obj_id, position = item
#         if batch_name == 'batch_1':
#             batch_1.append((existing_obj_name, new_obj_category, obj_id, position))
#         elif batch_name == 'batch_2':
#             batch_2.append((existing_obj_name, new_obj_category, obj_id, position))

#     with open('object_relations.txt', 'w') as file:
#         def write_batch(batch, batch_name):
#             file.write(f"### {batch_name.capitalize()} ###\n")
#             for existing_obj_name, new_obj_category, obj_id, position in batch:
#                 line = f"{existing_obj_name}: {new_obj_category},{obj_id},{','.join(map(str, position))}"
#                 file.write(line + '\n')
#             file.write("\n")  # 批次间空行分隔
        
#         write_batch(batch_1, "batch_1")
#         write_batch(batch_2, "batch_2")

#     # 移动batch_2中的物品到临时位置
#     temp_position = [200, 200, 0]
#     temp_z = 0.05
#     for item in batch_2:
#         _, _, obj_id, _ = item
#         set_base_values_with_z(obj_id, temp_position, temp_z)

#     return new_objects, new_relations


def change_environment(filepath = "object_relations.txt"):
    """
    从给定路径的文件中读取对象ID和位置信息，并调用set_base_values_with_z函数来移动对象。
    - batch_1中的物品将被移动到临时位置(200, 200, 0.05)。
    - batch_2中的物品将被移动到它们记录的位置。
    
    参数:
    - filepath: 包含对象ID和位置信息的文件路径。
    - set_base_values_with_z: 函数，用于设置物体的位置和Z轴的高度。
    """
    batch_1 = []
    batch_2 = []
    current_batch = None

    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith("### Batch_1 ###"):
                current_batch = batch_1
                continue
            elif line.startswith("### Batch_2 ###"):
                current_batch = batch_2
                continue
            elif not line or current_batch is None:
                continue
            
            # 解析对象名、类别、ID和位置
            parts = line.split(':')
            if len(parts) != 2:
                print(f"格式错误的行: {line}")
                continue
            
            existing_obj_name, rest_info = parts
            new_obj_category, obj_id, x, y, z = rest_info.split(',')
            
            # 假设obj_id是字符串形式，需要转换为整数或保持为字符串，取决于set_base_values_with_z的要求
            try:
                obj_id = int(obj_id)
            except ValueError:
                pass
            
            # 将x, y, z坐标转换为浮点数
            position = (float(x), float(y))
            z_value = float(z)
            
            current_batch.append((obj_id, position, z_value))

    # 移动batch_1中的物品到临时位置
    temp_position = [200, 200, 0]
    temp_z = 0.05
    for item in batch_1:
        obj_id, _, _ = item
        set_base_values_with_z(obj_id, temp_position, temp_z)

    # 移动batch_2中的物品到指定位置
    for item in batch_2:
        obj_id, position, z_value = item
        set_base_values_with_z(obj_id, [position[0], position[1], 0], z_value)

# 示例调用
# 假设object_relations.txt是你之前生成的包含对象信息的文件
# move_objects_based_on_file('object_relations.txt', set_base_values_with_z)


# 注意：你需要确保env.scene有set_base_values_with_z方法或者适配到现有的移动逻辑
