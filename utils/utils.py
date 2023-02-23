import json
import pandas as pd
from tqdm import tqdm
import random
import os
import numpy as np
import torch

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_topic_tree(input_topic_df):
    df = pd.DataFrame()

    for channel in tqdm(input_topic_df['channel'].unique()):
        channel_df = input_topic_df[(input_topic_df['channel'] == channel)].reset_index(drop=True)
        for level in sorted(channel_df.level.unique()):
            # For level 0, it first creates a topic tree column which is the title of that topic.
            if level == 0:
                topic_tree = channel_df[channel_df['level'] == level]['title'].astype(str)
                topic_tree_df = pd.DataFrame([channel_df[channel_df['level'] == level][['id']], topic_tree.values]).T
                topic_tree_df.columns = ['child_id', 'topic_tree']
                channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(
                    ['child_id'], axis=1)

            # Once the topic tree column has been created, the parent node and child node is merged on parent_id = child_id
            topic_df_parent = channel_df[channel_df['level'] == level][['id', 'title', 'parent', 'topic_tree']]
            topic_df_parent.columns = 'parent_' + topic_df_parent.columns

            topic_df_child = channel_df[channel_df['level'] == level + 1][['id', 'title', 'parent', 'topic_tree']]
            topic_df_child.columns = 'child_' + topic_df_child.columns

            topic_df_merged = topic_df_parent.merge(topic_df_child, left_on='parent_id', right_on='child_parent')[
                ['child_id', 'parent_id', 'parent_title', 'child_title', 'parent_topic_tree']]

            # Topic tree is parent topic tree + title of the current child on that level
            topic_tree = topic_df_merged['parent_topic_tree'].astype(str) + ' > ' + topic_df_merged[
                'child_title'].astype(str)

            topic_tree_df = pd.DataFrame([topic_df_merged['child_id'].values, topic_tree.values]).T
            topic_tree_df.columns = ['child_id', 'topic_tree']

            channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(
                ['child_id'], axis=1)
            if 'topic_tree_y' in list(channel_df.columns):
                channel_df['topic_tree'] = channel_df['topic_tree_x'].combine_first(channel_df['topic_tree_y'])
                channel_df = channel_df.drop(['topic_tree_x', 'topic_tree_y'], axis=1)

        df = pd.concat([df, channel_df], ignore_index=True)
    return df[["id", "topic_tree"]]

def read_config():
    f = open('config.json')
    config = json.load(f)
    config["supervised_model"]["betas"] = eval(config["supervised_model"]["betas"])
    return config