#!/usr/bin/env python
# coding: utf-8
# %%
## 1. Training Unsupervised SentenceTransformer

# %%

import faulthandler
faulthandler.enable()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers import datasets


from datasets import Dataset
from utils.evaluators import InformationRetrievalEvaluator

import warnings
warnings.filterwarnings('ignore')


# %%
# Custom libraries
from utils.unsupervised_utils import generate_topic_model_input, generate_content_model_input, read_data
from utils.utils import read_config

# %%
os.environ["TOKENIZERS_PARALLELISM"]="true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="false"


# %%
config = read_config()


# %%
DATA_PATH = "../raw_data/"


# %%
topics, content, correlations, _ = read_data(data_path=DATA_PATH,
                                             config_obj=config,
                                             read_mode="train")

topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])

corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")

corr["set"] = corr[["topic_model_input", "content_model_input"]].values.tolist()
train_df = pd.DataFrame(corr["set"])

dataset = Dataset.from_pandas(train_df)

train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    if example[0] == None:
        continue        
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))

# %%
# Setting-up the Evaluation

test_topics, test_content, test_correlations, _ = read_data(data_path=DATA_PATH,
                                                            config_obj=config,
                                                            read_mode="test")

test_correlations["content_id"] = test_correlations["content_ids"].str.split(" ")
test_correlations = test_correlations[test_correlations.topic_id.isin(test_topics.id)].reset_index(drop=True)
test_correlations["content_id"] = test_correlations["content_id"].apply(set)
test_correlations = test_correlations[["topic_id", "content_id"]]


# %%
ir_relevant_docs = {
    row['topic_id']: row['content_id'] for i, row in tqdm(test_correlations.iterrows())
}


# %%
unq_test_topics = test_correlations.explode("topic_id")[["topic_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
unq_test_topics = unq_test_topics.merge(test_topics[["id", "model_input"]], how="left", left_on="topic_id",
                       right_on="id").drop("id", 1)

ir_queries = {
    row['topic_id']: row['model_input'] for i, row in tqdm(unq_test_topics.iterrows())
}


# %%
all_topics, all_content, _, special_tokens = read_data(data_path=DATA_PATH,
                                                         config_obj=config,
                                                         read_mode="all")

unq_contents = correlations.explode("content_id")[["content_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
unq_contents = unq_contents.merge(all_content[["id", "model_input"]], how="left", left_on="content_id",
                       right_on="id").drop("id", 1)

ir_corpus = {
    row['content_id']: row['model_input'] for i, row in tqdm(unq_contents.iterrows())
}

# %%
evaluator = InformationRetrievalEvaluator(
    ir_queries,
    ir_corpus,
    ir_relevant_docs,
    show_progress_bar=True,
    main_score_function="cos_sim",
    precision_recall_at_k=[5, 10, 25, 50, 100],
    name='K12-local-test-unsupervised'
)

# %%
# Training

train_dataloader = datasets.NoDuplicatesDataLoader(train_examples,
                                                   batch_size=config["unsupervised_model"]["batch_size"])


# %%
TARGET_MODEL = config["unsupervised_model"]["base_name"]
OUT_MODEL = config["unsupervised_model"]["save_name"]
TARGET_MODEL, OUT_MODEL


# %%
model = SentenceTransformer(TARGET_MODEL)
model.max_seq_length = config["unsupervised_model"]["seq_len"]

word_embedding_model = model._first_module()
word_embedding_model.tokenizer.add_tokens(list(special_tokens),
                                          special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


# %%
train_loss = losses.MultipleNegativesRankingLoss(model=model)

#k% of train data
num_epochs = config["unsupervised_model"]["epochs"]
warmup_steps = int(len(train_dataloader) * config["unsupervised_model"]["warmup_ratio"])


# %%
model.fit(train_objectives=[(train_dataloader, train_loss)],
#           scheduler="constantlr",
#           optimizer_class=Lion,
#           optimizer_params={'lr': 2e-5},
          evaluator=evaluator,
#           evaluation_steps=400,
          
          checkpoint_path=f"checkpoints/unsupervised/{OUT_MODEL.split('/')[-1]}",
          checkpoint_save_steps=len(train_dataloader),
    
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=OUT_MODEL,
          save_best_model=True,
         use_amp=True)

