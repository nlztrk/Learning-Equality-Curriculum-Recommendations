#!/usr/bin/env python
# coding: utf-8
# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %%
# =========================================================================================
# Libraries
# =========================================================================================
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

# Custom libraries
from utils.unsupervised_utils import read_data
from utils.utils import read_config
from utils.metrics import get_pos_score, get_f2_score

os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="true"

# %%
config = read_config()
DATA_PATH = "../raw_data/"
GENERATED_DATA_PATH = "./generated_files/"


# %%
train_df = pd.read_parquet(GENERATED_DATA_PATH + "unsupervised_train.parquet")
test_df = pd.read_parquet(GENERATED_DATA_PATH + "unsupervised_test.parquet")

correlation_df = pd.read_csv(DATA_PATH + "correlations.csv")

# %%
train_samples = [InputExample(texts=[row.model_input1,
                                     row.model_input2],
                              label=int(row.target)) for row in tqdm(train_df.itertuples())]

test_samples = [InputExample(texts=[row.model_input1,
                                     row.model_input2],
                              label=int(row.target)) for row in tqdm(test_df.itertuples())]


# %%
model = CrossEncoder(config["supervised_model"]["base_name"],
                     num_labels=1,
                    max_length=config["supervised_model"]["seq_len"])

num_epochs = config["supervised_model"]["epochs"]

train_dataloader = DataLoader(train_samples,
                              shuffle=True,
                              batch_size=config["supervised_model"]["batch_size"],
                              num_workers=0,
                             pin_memory=False)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples,
                                                                name='K12-local-test',
                                                               show_progress_bar=True
                                                               )

warmup_steps = math.ceil(len(train_dataloader) * config["supervised_model"]["warmup_ratio"])


# %%
model.fit(train_dataloader=train_dataloader,
          show_progress_bar=True,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
           save_best_model=True,
          output_path=config["supervised_model"]["save_name"],
         use_amp=True)


# %%
model

# %%
# ### Load Model & Tune Threshold

model = CrossEncoder(config["supervised_model"]["save_name"],
                    num_labels=1,
                    max_length=config["supervised_model"]["seq_len"])

preds = model.predict(test_df[["model_input1", "model_input2"]].values,
                      show_progress_bar=True,
                      batch_size=96)

test_df["pred_score"] = preds


# %%
for thr in np.arange(0., 0.3, 0.0025):
    preds_thr_df = test_df[test_df.pred_score >= thr].sort_values(by="pred_score",
                                                    ascending=False)[["topics_ids",
                                                                      "content_ids"]].\
                                    groupby("topics_ids")["content_ids"].apply(lambda x: " ".join(x)).rename("pred_content_ids").reset_index()

    preds_thr_df = preds_thr_df.merge(correlation_df[correlation_df.topic_id.isin(test_df.topics_ids)],
                                      how="outer", right_on="topic_id", left_on="topics_ids")
    preds_thr_df.fillna("None", inplace=True)
    f2score_for_threshold = get_f2_score(preds_thr_df['content_ids'],
                                         preds_thr_df['pred_content_ids'])

    print(f"Threshold: {thr} | Score: {f2score_for_threshold}")

# %%
# Threshold: 0.0175 | Score: 0.6395 @100
# Threshold: 0.0175 | Score: 0.6424 @75
# Threshold: 0.0150 | Score: 0.6461 @50
# Threshold: 0.0050 | Score: 0.6464 @25
