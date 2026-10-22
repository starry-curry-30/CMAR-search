# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
# NOTE: monkey-patching; needs to be imported before any other file imports it
from moma_llm.tasks.patched_scene import MonkeyPatchedInteractiveIndoorScene
from igibson.scenes import igibson_indoor_scene
igibson_indoor_scene.InteractiveIndoorScene._add_object = MonkeyPatchedInteractiveIndoorScene._add_object
igibson_indoor_scene.InteractiveIndoorScene._orig_add_object = MonkeyPatchedInteractiveIndoorScene._orig_add_object
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from igibson.utils.utils import parse_config
from sklearn.metrics import auc
import wandb
import networkx as nx
from moma_llm.env.baselines import (GreedyBaseline,
                                    RandomBaseline)
from moma_llm.env.env import OurIGibsonEnv, create_igibson_env
from moma_llm.env.llm_env import JsonLLMEnv, LLMEnv
from moma_llm.llm.llm import LLM
from moma_llm.utils.constants import TEST_SCENES, TRAINING_SCENES
from moma_llm.utils.utils import get_config
from moma_llm.tasks.object_sampling import add_objects_from_our_distribution, change_environment

def create_env(cfg, agent: str, config_file: str, scene_id: str, control_freq: float, cheap: bool, seed: int) -> LLMEnv:
    if agent == "moma_llm":
        env_fn = LLMEnv
    elif agent == "json_llm":
        env_fn = JsonLLMEnv
    elif agent == "greedy":
        env_fn = GreedyBaseline
    elif agent == "random":
        env_fn = RandomBaseline
    else:
        raise ValueError(f"Unknown agent {agent}")

    llm_variant = "deepseek-chat"   # "gpt-4？"
    llm = LLM(debug=True, model=llm_variant, room_classification_model="deepseek-chat",
              open_set_rooms=cfg["open_set_room_categories"])

    low_level_env = create_igibson_env(config_file=config_file,
                                       control_freq=control_freq,
                                       scene_id=scene_id,
                                       seed=seed)
    high_level_env = env_fn(env=low_level_env, llm=llm, seed=seed)
    return high_level_env


def calc_area_under_curve(x, y, max_x):
    if max(x) > max_x:
        idx = (x <= max_x)
        x = x[idx]
        y = y[idx]
    if max(x) < max_x:
        x = np.concatenate([x, [max_x]])
        y = np.concatenate([y, [y[-1]]])

    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    return auc(x, y) / max_x


def plot_efficiency_curves(episode_infos, max_hl_steps: int):
    ll_steps = []
    ll_steps_gtDone = []
    hl_steps = []
    task_success = []
    task_success_gtDone = []
    for scene_id in sorted(episode_infos.keys()):
        for e in episode_infos[scene_id]:
            ll_steps.append(e["num_low_level_steps_with_open_cost"])
            ll_steps_gtDone.append(e["num_low_level_steps_with_open_cost_gtDone"])
            hl_steps.append(e["num_high_level_steps"])
            task_success.append(e["task_success"])
            task_success_gtDone.append(e["task_success_gtDone"])
    task_success = np.array(task_success)
    task_success_gtDone = np.array(task_success_gtDone)
    ll_steps = np.array(ll_steps)
    hl_steps = np.array(hl_steps)

    def _plot(steps, task_success):
        df = pd.DataFrame({"steps": steps, "task_success": task_success})
        df = df.sort_values("steps")

        values = [np.logical_and(df["task_success"].values, df["steps"].values <= max_steps).mean() for max_steps in
                  df["steps"]]
        df2 = pd.DataFrame({"steps": df["steps"].values, "success": values})
        return wandb.Table(dataframe=df2)

    def _get_auc(steps, task_success, max_x: int, title: str):
        table = _plot(steps=steps, task_success=task_success)
        auc = calc_area_under_curve(table.get_dataframe()["steps"].values, table.get_dataframe()["success"].values,
                                    max_x=max_x)
        wandb_plot = wandb.plot_table("wandb/area-under-curve/v0",
                                      table,
                                      {"x": "steps", "y": "success"},
                                      {"title": title,
                                       "x-axis-title": "Steps",
                                       "y-axis-title": "Success rate", }, )
        return auc, wandb_plot

    hl_auc, hl_plot = _get_auc(steps=hl_steps, task_success=task_success, max_x=max_hl_steps,
                               title="High-level-step-curve")
    wandb.log({"efficiency_curve_high_level_steps": hl_plot, "hl_auc": hl_auc})

    ll_auc_max_steps = 5000
    ll_auc, ll_plot = _get_auc(steps=ll_steps, task_success=task_success, max_x=ll_auc_max_steps,
                               title="Low-level-step-curve")
    wandb.log({"efficiency_curve_low_level_steps": ll_plot, "ll_auc": ll_auc})

    ll_auc_gtDone, ll_plot_gtDone = _get_auc(steps=ll_steps_gtDone, task_success=task_success_gtDone,
                                             max_x=ll_auc_max_steps, title="Low-level-step-curve-gtDone")
    wandb.log({"efficiency_curve_low_level_steps_gtDone": ll_plot_gtDone, "ll_auc_gtDone": ll_auc_gtDone})


def calculcate_metric_means(episode_infos):
    columns = sorted(list(episode_infos.values())[0][0].keys())
    scene_logs = defaultdict(dict)
    for scene_id in sorted(episode_infos.keys()):
        for column in columns:
            if isinstance(episode_infos[scene_id][0].get(column, None), str):
                continue
            elif isinstance(episode_infos[scene_id][0].get(column, None), (np.ScalarType)):
                d = np.nanmean([e.get(column, np.nan) for e in episode_infos[scene_id]])
                # NOTE: if value is not defined for an episode, we take the mean over those that have the value!
                #   e,g, [steps]_gtDone are only defined for successful episodes
                scene_logs[scene_id][column] = d
            else:
                continue
            print(column, d)
        # 计算总运行时长
        total_runtime = sum(e["runtime"] for e in episode_infos[scene_id])
        scene_logs[scene_id]["total_runtime"] = total_runtime
    return scene_logs



def log_summary_table(episode_infos):
    def _check_float(v):
        try:
            float(v)
            return True
        except:
            return False

    scene_logs = calculcate_metric_means(episode_infos)
    data = []
    scenes = list(scene_logs.keys())
    columns = ["scene_id"]
    for k in scene_logs[scenes[0]].keys():
        # 仅添加所有场景都有的指标
        if all([k in scene_logs[s] for s in scenes]):
            columns.append(k)

    for scene_id in sorted(scenes):
        data.append([scene_id] + [scene_logs[scene_id][c] for c in columns[1:]])

    avg_row = ["Overall avg"]
    avg_dict = {}
    for i, column_values in enumerate(np.array(data).T):
        if _check_float(column_values[0]):
            d = np.mean(column_values.astype(float))
            avg_dict[f"avg_{columns[i]}"] = d
            avg_row.append(str(d))
    data.append(avg_row)
    avg_dict["overview_table"] = wandb.Table(columns=columns, data=np.array(data).astype(str))
    wandb.log(avg_dict)

def delete_folders_if_exist(*folder_paths): ###############liaokaiyao  delete folders if exist
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Folder {folder_path} has been deleted.")
            except Exception as e:
                print(f"Failed to delete {folder_path}. Reason: {e}")
        else:
            print(f"Folder {folder_path} does not exist.")
################################################################################

def evaluate_scene(config_file: str, cfg, scene_id: str, tot_ep: int) -> list:
    # folder_path1 = 'save_rgb' #LKY
    # folder_path2 = 'output_rgb'
    # delete_folders_if_exist(folder_path1, folder_path2)#LKY
    episode_infos = []
    with open ("execute_action.txt", "w") as f:
        f.write(" ")
    
    with open('blacklist.txt', 'w') as f:  # 'w'模式会创建文件或覆盖已有内容
        f.write('')

    with open("check_find_target.txt", "w") as f:
        f.write('')

    with open("rgb_find_target.txt", "w") as f:
        f.write('')
    
    with open("all_objects.txt", "w") as f:
        f.write('')

    high_level_env = create_env(cfg, agent=cfg["agent"], config_file=config_file, scene_id=scene_id,          ########这里是创建环境，看看怎么修改可以换物品
                                control_freq=cfg["control_freq"], cheap=cfg["cheap"], seed=cfg["seed"])     ########high_level_env对象 的方法都在llm_env 在LLMENV里面       

    n = cfg["num_episodes_per_scene"]
    i = 0
    retry_count = 0
    while i < n:

        with open ("execute_action.txt", "w") as f:
            f.write(" ")
    
        with open('blacklist.txt', 'w') as f:  # 'w'模式会创建文件或覆盖已有内容
            f.write('')

        with open("check_find_target.txt", "w") as f:
            f.write('')

        with open("rgb_find_target.txt", "w") as f:
            f.write('')
    
        with open("all_objects.txt", "w") as f:
            f.write('')

        with open ("stuck.txt", "w") as f:
            f.write('')
            

        start_time = time.time()  # 记录开始时间
        done = False

        
        obs = high_level_env.reset(config_file=config_file, scene_id=scene_id, episode_num=i)  ############初始环境

          
        print("########################################")
        print(   
            f"{scene_id} - Starting episode {i + 1} in scene {scene_id}, {tot_ep + 1} overall. Task: {high_level_env.unwrapped.task.task_description}")                                                       
        print("########################################")

        while not done:
            # high_level_env.visualize(obs)  

            done, task_success, episode_info = high_level_env.take_action(obs=obs,    ###########做任务
                                                                          task_description=high_level_env.unwrapped.task.task_description)
            

            # env adds last action to the figure title, that's why we log it after the env step
            wandb.log({"bev_maps": high_level_env.unwrapped.f})
            obs = high_level_env.get_state(compute_scene_graph=True)

            pprint(episode_info)

   #     save_graph_as_pkl(obs["room_object_graph"], base_directory="./room_object_graph_lky", base_filename="room_object_graph", extension=".pkl") #######LKY

        # delete_folders_if_exist(folder_path1, folder_path2)####LKY
        
    #    save_clipped_map(obs["clipped_map"]) ###LKY

        try:
            with open('stuck.txt', 'r') as f:
                stuck = f.read().strip().lower()
                if stuck == '1' and task_success == False:
                    retry_count += 1  # Increment the counter
                    i += 1
                    n += 1
                    print("机器人卡住，重新开始任务")
                    if retry_count > 3:  # Check if we've exceeded the limit
                        print("超过最大重试次数(3次)，跳出循环")
                        break  # Break out of the outer loop
                    continue  # 跳过本次循环，且 i 不递增
        except FileNotFoundError:
            pass  # 文件不存在时继续执行
        
        # Reset the counter if we successfully complete an episode without retry
        retry_count = 0
        i += 1


        labeling_results = high_level_env.evaluate_room_labeling(obs)

        # 分别添加两个评估指标到episode_info
        episode_info["room_semantic_accuracy"] = labeling_results["semantic_accuracy"]
        episode_info["room_segmentation_purity"] = labeling_results["purity"]

        episode_runtime = time.time() - start_time  # 计算运行时长
        episode_info["runtime"] = episode_runtime  # 将运行时长添加到 episode_info 中



        # high_level_env.visualize(obs)
        
        # if "failure_reason" in episode_info:
        #     # high_level_env.env.f.suptitle(
        #     #     f"{high_level_env.env.f._suptitle.get_text()}, {episode_info['failure_reason']}")
        episode_info["bev_maps"] = high_level_env.unwrapped.f
        episode_info["num_low_level_steps_with_open_cost"] = episode_info["num_low_level_steps"] + \
                                                             high_level_env.env.config["magic_open_cost"] * \
                                                             episode_info["magic_open_actions"]
        if episode_info.get("num_low_level_steps_gtDone", None) is not None:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = episode_info["num_low_level_steps_gtDone"] + \
                                                                        high_level_env.env.config["magic_open_cost"] * \
                                                                        episode_info["magic_open_actions_gtDone"]
            episode_info["task_success_gtDone"] = True
        else:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = episode_info[
                "num_low_level_steps_with_open_cost"]
            episode_info["task_success_gtDone"] = task_success
        episode_info["episode_step"] = tot_ep
        episode_info["spl"] = episode_info["task_success"] * (
                    episode_info["shortest_dist"] / max(episode_info["shortest_dist"], episode_info["dist_travelled"]))
        pprint(episode_info)
        # episode_info["rgb"] = wandb.Video((255 * np.transpose(np.stack(high_level_env.env.rgb_frames, axis=0), (0, 3, 1, 2))).astype(np.uint8), fps=6)
        wandb.log({k: float(v) if isinstance(v, bool) else v for k, v in episode_info.items()})

        episode_infos.append(episode_info)
        successes = [e["task_success"] for e in episode_infos] 
        tot_ep += 1
        print(
            f"Task success: {task_success} (wandb_step: {wandb.run.step}). Current successes: {sum(successes)}/{len(successes)}")
    
    scene_logs = calculcate_metric_means({scene_id: episode_infos})
    wandb.log({f"{scene_id}_{k}": v for k, v in scene_logs[scene_id].items()}) 

    high_level_env.close()
    return episode_infos, tot_ep

def main(): 
    np.set_printoptions(precision=3, suppress=True)

    config_file = get_config("moma_llm.yaml")
    # NOTE: igibson will reload the config file, so changes here won't be relfected! Just for wandb logging
    cfg = parse_config(config_file)
    if cfg["seed"] > 0: 
        np.random.seed(cfg["seed"])

    if cfg["datasplit"] == "train":
        scene_ids = TRAINING_SCENES 
    elif cfg["datasplit"] == "test": 
        scene_ids = TEST_SCENES
    else:
        raise ValueError(f"Unknown datasplit {cfg['datasplit']}")

    cfg.update({"scene_ids": scene_ids, "agent": cfg["agent"]})
    wandb.init(
               project="cmar-search",
               entity=os.environ.get("WANDB_ENTITY"),
               config=cfg,
               #mode="online" if cfg["wandb"] else "disabled",
                mode="offline",
               name="task1"
               )
    # copy config file to wandb run dir, so modifications to the main config file won't affect current runs
    new_config_file = Path(wandb.run.dir) / Path(config_file).name
    shutil.copy(config_file, new_config_file)
    config_file = str(new_config_file)

    episode_infos = defaultdict(list)

    tot_ep = 0
    all_episode_data = []
    if isinstance(scene_ids, str):
        scene_ids = [scene_ids]
    for scene_id in scene_ids:
        infos, tot_ep = evaluate_scene(config_file=config_file, cfg=cfg, scene_id=scene_id, tot_ep=tot_ep)   ######这里开始运行
        episode_infos[scene_id] = infos
        for info in infos:
            all_episode_data.append([
                info["num_low_level_steps_with_open_cost"],
                info["num_high_level_steps"],
                info["task_success"]
            ])

    # 创建并记录 wandb 表格
    table = wandb.Table(columns=["Low Level Steps with Open Cost", "High Level Steps", "Task Success"], data=all_episode_data)
    wandb.log({"All Episodes Info Table": table})

    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos, max_hl_steps=cfg["max_high_level_steps"])

    wandb.run.finish()
    print("Done!")


if __name__ == "__main__":
    main()
