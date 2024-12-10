# Utilizing LLMs to Adjust NFM Recommendation Lists

## Acknowledgement
The data pre-processing codes, NFM model and training codes are adapted from [MicroLens](https://github.com/westlake-repl/MicroLens).

Building on the NFM model from [MicroLens](https://github.com/westlake-repl/MicroLens), **inference** functionality is explicitly added to enable the output of top-N recommended items given user history.

## Purpose
This is an experiment on the performance of large language models (LLMs) modifying
the per-user top recommendation lists produced by Neural Factorization Machines (NFM),
an ID-based recommendation system model.

In the process, each user is represented by a recent subset of videos they have interacted with (commented on),
while the most recently interacted videos are hidden from models for validation and test.

Titles are used to generate text attributes for items -- short-form videos in this case.
They are then fed to LLMs via Hugging Face [serverless inference API](https://huggingface.co/docs/api-inference/en/index) calls for vectorized embeddings or together with recent user history and NFM predictions for
more direct re-ranked recommendation lists.

Two methods of LLM usage are applied:
- Use LLM encoders to generate item embeddings, and re-rank recommendations via similarity comparison.
- Prompt decoder-only LLMs to get re-ranked recommendations.

## Usage
- Required data files from [MicroLens-100k-Dataset](https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/): `MicroLens-100k_pairs.tsv` for model training; `MicroLens-100k_pairs.csv`, `MicroLens-100k_title_en.csv` for model inference and LLM re-ranking.

- Currently, LLM models have to be accessible via Hugging Face [serverless inference API](https://huggingface.co/docs/api-inference/en/index) calls.

```
pip install -r requirements.txt
python run_pipeline.py --encoder_model <encoder_model_name> --llama_model <llama_model_name>
```

## Outcome
### NFM Model Training

Current hyperparameter configurations can train a NFM model with decent hit ratio statistics for a recommendation
system model that only utilizes item IDs.

- **Metric: Hit Ratio at N (HR@N)**, the proportion of users for which the top N recommended items include the ground truth (the most recent item they have interacted, not visible to the model during training).

|  Total Number of Users | HR@5 | HR@10 | HR@20|
|---------------------------|------|-------|-------|
|      100,000              |  0.01747 | 0.02939 |  0.04618 |

### LLM Processing

However, apparently LLMs are not good at re-ranking NFM recommendation results on the sample
that includes 100 users whose original top-20 recommendation lists include the ground truth (the most recently interacted item).

- **Metric: Number of Hits at N (Hits@N)**, the number of users for which the top N recommended items include the ground truth (the most recent item they have interacted, not visible to the LLM during processing).

|                                   |  Hits@5  | Hits@10 |
|----------------------------------|----------|------ |
| NFM (baseline)|  38 | 64 |  
| `e5-small-v2` (encoding + similarity comparison)  | 32 (-16%) | 56 (-13%) |
| `Meta-Llama-3-8B-Instruct` (decoding) | 29 (-24%) | 49 (-23%) |