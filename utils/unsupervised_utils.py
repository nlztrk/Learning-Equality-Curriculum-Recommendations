import pandas as pd
from tqdm import tqdm
import gc

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from cuml.neighbors import NearestNeighbors
import cupy as cp
import torch

from .utils import generate_topic_tree

def get_neighbors(topic_df,
                  content_df,
                  config_obj):
    # Create unsupervised model to extract embeddings
    model = SentenceTransformer(config_obj["unsupervised_model"]["save_name"])
    model = model.to("cuda")

    # Predict
    topics_preds = model.encode(topic_df["model_input"],
                                show_progress_bar=True)
    content_preds = model.encode(content_df["model_input"],
                                 show_progress_bar=True,
                                 batch_size=100)

    # Transfer predictions to gpu
    topics_preds_gpu = cp.array(topics_preds)
    content_preds_gpu = cp.array(content_preds)

    # Release memory
    torch.cuda.empty_cache()
    gc.collect()

    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors=config_obj["unsupervised_model"]["top_n"],
                                       metric='cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance=False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content_df.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    topic_df['predictions'] = predictions

    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    return topic_df, content_df


def build_training_set(topic_df,
                       content_df,
                       mode="local"):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    # Iterate over each topic
    for k in tqdm(range(len(topic_df))):
        row = topic_df.iloc[k]
        topics_id = row['id']
        topics_title = row['model_input']
        predictions = row['predictions'].split(' ')

        if mode == "local":
            ground_truth = row['content_ids'].split(' ')

        for pred in predictions:
            content_title = content_df.loc[pred, 'model_input']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            # If pred is in ground truth, 1 else 0
            if mode == "local":
                if pred in ground_truth:
                    targets.append(1)
                else:
                    targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids,
         'content_ids': content_ids,
         'model_input1': title1,
         'model_input2': title2
         }
    )
    if mode == "local":
        train["target"] = targets

    return train

def read_data(data_path,
              config_obj,
              read_mode="all"):
    topics = pd.read_csv(data_path + 'topics.csv')
    content = pd.read_csv(data_path + 'content.csv')
    if read_mode != "all":
        correlations = pd.read_csv(data_path + 'correlations.csv')
    else:
        correlations = None
    topic_trees = generate_topic_tree(topics)

    if read_mode != "all":
        splits = pd.read_csv("train_test_splits.csv")
        topics = topics[topics.id.isin(splits[splits.fold == read_mode].id)].reset_index(drop=True)

    topics = topics.merge(topic_trees, how="left", on="id")

    topics = generate_topic_model_input(input_df=topics,
                                        seq_len=config_obj["unsupervised_model"]["seq_len"])
    content = generate_content_model_input(input_df=content,
                                           seq_len=config_obj["unsupervised_model"]["seq_len"])

    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace=True)
    content.sort_values('length', inplace=True)

    # Drop cols
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length'], axis=1,
                inplace=True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis=1,
                 inplace=True)
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    if read_mode != "all":
        print(f"correlations.shape: {correlations.shape}")

    return topics, content, correlations


def generate_topic_model_input(input_df,
                               seq_len=128):
    """
    :param input_df: Input topic dataframe
    :return: Dataframe with additional model input column
    """

    input_df = input_df.fillna("")
    input_df = input_df.astype(str)

    whole_input = "" + input_df["language"] + \
                  " " + input_df["level"] + \
                  " [SEP] " + input_df["title"] +\
                  " " + input_df["description"] + \
                  " " + input_df["topic_tree"]

    input_df["model_input"] = whole_input.str.split().apply(lambda x: " ".join(x[:seq_len]))

    return input_df


def generate_content_model_input(input_df,
                                 seq_len=128):
    """
    :param input_df: Input content dataframe
    :return: Dataframe with additional model input column
    """

    input_df = input_df.fillna("")
    input_df = input_df.astype(str)

    whole_input = "" + input_df["title"] +\
                  " " + input_df["description"] + \
                  " " + input_df["text"]

    input_df["model_input"] = whole_input.str.split().apply(lambda x: " ".join(x[:seq_len]))

    return input_df