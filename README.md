## Solution for "Learning Equality - Curriculum Recommendations" @Kaggle

![architecture](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)

I used [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library and the models from [HuggingFace](https://huggingface.co/). I tried to implement the shared architecture [here](https://www.sbert.net/examples/applications/retrieve_rerank/README.html).

The pipeline consists of:
- [Splitting the Data as Train/Val](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/0.%20Generating%20Splits.ipynb)
- [Text Processing](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/utils/unsupervised_utils.py#L154)
- [Training Sentence-Transformer (Stage 1)](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/1.%20Unsupervised%20Training.py)
- [Retrieve with kNN using Stage 1 Embeddings](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/2.%20Unsupervised%20Sampling.ipynb)
- [Training Cross-Encoder (Stage 2)](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/3.%20Supervised%20Training.py)
- [Inference](https://github.com/nlztrk/Learning-Equality-Curriculum-Recommendations/blob/main/4.%20Inference.ipynb)

### Splitting the Data as Train/Val
I've seen a lot of different approaches on the forum. I also wanted to use the imbalance in language distribution in my approach. I set all the data coming from **source** as **train**. For the remaining, I used:

- **CV Scheme:** Grouped Stratified K-Fold
- **Folds:** 5 (Used only the first)
- **Group:** Topic ID
- **Stratifier Label:** Language

### Text Processing
- Created topic tree
- Created special tokens for each value **language** and **content kind** can take.
- Created identifier separators for **topic title**, **topic tree**, **topic description**, **content title**, **content description** and **content text**.

My final input for the model was like:
- **Topic:** `[<[language_en]>] [<[topic_title]>] videos [<[topic_tree]>] maths g3 to g10 > maths > g6 > 17. geometrical constructions > perpendicular and perpendicular bisector > videos [<[topic_desc]>] nan`
- **Content:** `[<[language_en]>] [<[kind_exercise]>] [<[cntnt_title]>] level 3: identify elements of simple machine(axle,wheel,pulley and inclined plane etc [<[cntnt_desc]>] nan [<[cntnt_text]>] nan`

### Training Sentence-Transformer (Stage 1)
- **Base Model:** [AIDA-UPM/mstsb-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/AIDA-UPM/mstsb-paraphrase-multilingual-mpnet-base-v2)
- **Sequence Length:** 128
- **Epochs:** 50
- **Batch Size:** 128
- **Warm-Up Ratio:** 0.03

### Retrieve with kNN using Stage 1 Embeddings
I used **kNN** from [RAPIDS](https://rapids.ai/) and get closest **100** content embedding for each topic embedding using **cosine-similarity**.

### Training Cross-Encoder (Stage 2)
- **Base Model:** Trained model from Stage 1
- **Output:** Sigmoid
- **Sequence Length:** 128
- **Epochs:** 15
- **Batch Size:** 256
- **Warm-Up Ratio:** 0.05

### Inference
- Ran all the steps above sequentially in a single script.
- Tuned classification threshold on the hold-out validation set to maximize F2-Score.
- Imputed empty topic rows with the highest scoring content IDs.

## Didn't Work & Improve

- Language specific kNN
- Smaller models
- Lower sequence length
- Lower batch-size
- Union submission blending
