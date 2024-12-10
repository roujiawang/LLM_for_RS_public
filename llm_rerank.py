from dotenv import load_dotenv
import os
import pandas as pd
from typing import List
import re
from huggingface_hub import InferenceClient
import requests
import numpy as np
import json
import random


def parse_sampled_results_from_path(file_path, sample_size=100):
    '''
    Parse the data from complete inference results
    '''
    predictions_dict, full_history_dict = {}, {}

    # Read the file
    with open(file_path, "r") as file:
        lines = file.read().strip().split("\n\n")  # Split into blocks by empty lines

    '''
    sample block format:
    User [27985]:
    Ground Truth: [9163]
    Predictions: [17284,9163,17972,11021,5915,11386,19531,4883,16079,450,4431,17632,15092,2515,15483,1267,6361,12725,17194,6897]
    History: [1291,13248,10139,11302,10283,17206]
    '''

    # Filter out blocks with multiple users
    single_user_blocks = []
    for block in lines:
        user_matches = re.search(r"User \[([^\]]+)\]", block)
        if user_matches:
            users = list(map(int, user_matches.group(1).split(',')))
            if len(users) == 1:  # Include only blocks with a single user
                single_user_blocks.append(block)
    
    # Sample 200 blocks randomly from single-user blocks
    sampled_blocks = random.sample(single_user_blocks, min(len(single_user_blocks), sample_size))

    for block in sampled_blocks:
        # Extract all users, ground truth, predictions, and history using regex
        user_matches = re.search(r"User \[([^\]]+)\]", block)
        gt_match = re.search(r"Ground Truth: \[(\d+)\]", block)
        predictions_match = re.search(r"Predictions: \[([^\]]+)\]", block)
        history_match = re.search(r"History: \[([^\]]+)\]", block)

        if user_matches and gt_match and predictions_match and history_match:
            user = int(user_matches.group(1))  # Single user guaranteed
            ground_truth = int(gt_match.group(1))
            predictions_list = list(map(int, predictions_match.group(1).split(',')))
            history_list = list(map(int, history_match.group(1).split(',')))

            # Populate the dictionaries
            if user in predictions_dict:
                print(f"User {user} already recorded.")  # sanity check
            else:
                predictions_dict[user] = predictions_list
                full_history_dict[user] = [history_list, [ground_truth]]
        else:
            print(f"Skipped block: {block}")  # sanity check
    
    # Save results to JSON files
    predictions_file = "sampled_predictions_dict.json"
    with open(predictions_file, "w") as pred_file:
        json.dump(predictions_dict, pred_file, indent=4)
    print(f"Saved predictions to {predictions_file}")

    history_file = "sampled_full_history_dict.json"
    with open(history_file, "w") as hist_file:
        json.dump(full_history_dict, hist_file, indent=4)
    print(f"Saved history to {history_file}")

    return predictions_dict, full_history_dict


def get_all_items(pairs) -> List[int]:
    return pairs["item"].unique().tolist()


def filter_items(item_list: List[int], true_item_list: List[int]) -> List[int]:
    '''Remove the invalid item from a given item list.'''
    return [item for item in item_list if item in true_item_list]


def extract_title(item_id, titles):
    if item_id <= 0:
        raise ValueError(f"Item ID {item_id} is not valid.")

    title_row = titles[titles["item"] == item_id]
    if title_row.empty:
        raise ValueError(f"No title found for Item ID {item_id}.")
    return title_row["title"].iloc[0]


def reranker_llama_stream(predictions: List[int], user_history: List[int],model_name="meta-llama/Meta-Llama-3-8B-Instruct", titles=None) -> List[int]:

    prompt="You are a expert in recommending items to user.\n"

    previous_item_list = user_history[0]  # leave out the last interaction
    if titles is not None and not titles.empty:
        for item in previous_item_list:
            item_title = extract_title(item, titles)
            prompt += f"User has previously interacted with this item with title{item_title}.\n"

    item_list = predictions
    prompt += f"These are the item_ids that you need to rerank:{item_list}./n"
    prompt += "Below are the information of the items./n"
    if titles is not None and not titles.empty:
        titles_dict = {item: extract_title(item, titles) for item in item_list}
        for item, title in titles_dict.items():
            prompt += f"Item_Id:{item}.Title: {title}\n"

    prompt += "Please rerank these items for the user. Return the list of item_id in the order of ranking with each separated by a ','.\n"
    prompt += """
    {reranked_items_list}
    """
    client = InferenceClient(api_key=HF_API_TOKEN)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=7000,
        stream=True
    )
    # Initialize an empty string to store the accumulated result
    result_string = ""

    # Iterate through the stream and accumulate content into the result string
    for chunk in stream:
        result_string += chunk.choices[0].delta.content

    print(f"result string:\n{result_string}")

    match = re.search(r'(?:\d+\s*,\s*)+\d+\b(?!(?:.*\d+\s*,\s*)+\d+)', result_string)
    if match:
        reranked_items=list(map(int, match.group().split(',')))

        print("> llma re-ranker:", predictions, "re-ranked to", reranked_items)
        return reranked_items
    else:
        print("> llama re-ranker: invalid response")
        # do nothing
        return item_list


def reranker_sentence_transformers(predictions, user_history, model_name, titles):

    previous_item_list = user_history[0]  # leave out the last item
    previous_item_info=""
    if titles is not None and not titles.empty:
        for item in previous_item_list:
            item_title = extract_title(item, titles)
            previous_item_info += f"{item_title}.\n"

    item_list = predictions
    items_info = {item: "" for item in item_list}
    if titles is not None and not titles.empty:
        titles_dict = {item: extract_title(item, titles) for item in item_list}
        for item, title in titles_dict.items():
            items_info[item] += f"Title: {title}\n"

    payload = {"inputs":{"source_sentence":previous_item_info,"sentences":[items_info[item] for item in item_list]}}
    result = requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=payload).json()
    print(f"API response: {result}")

    # result is a list of scores for each item in the item_list
    # return id of the ranked items from highest score to lowest score
    ranked_items = [item_list[i] for i in np.argsort(result)]

    return ranked_items[::-1]


def get_hit_rate_at_n(N, results_dict, history_dict, num_items=2000):
    '''
    Returns the proportion of users whose top N recommendations in the results_dict
    contains the ground truth item in history_dict.

    results_dict[user] = [item1, item2, ...]
    history_dict[user] = [[recent_item1, recent_item2, ...], [most_recent_item]]
    '''

    num_hits = 0
    for user in results_dict.keys():
        # check if the latest interaction is predicted
        if history_dict[user][-1][0] in results_dict[user][:N]:
            num_hits += 1
    return num_hits / num_items


def evaluate_reranker(llm_model_name, predictions_dict, user_history_dict, nfm_reranker_dict, is_encoder=False,
                      titles=None, pairs=None):

    nfm_reranker_dict[llm_model_name] = {}

    true_item_list = get_all_items(pairs)

    # for each user, get re-ranked predictions from llm
    results_dict = {}
    for user_id in predictions_dict.keys():
        predictions = filter_items(predictions_dict[user_id], true_item_list)  # exclude invalid items
        user_history = user_history_dict[user_id]  # leave out the ground truth

        if(is_encoder):  # (sentence) transformer encoder
            reranked_predictions = reranker_sentence_transformers(predictions=predictions, user_history=user_history, model_name=llm_model_name,
                                        titles=titles)
        else:  # llama
            reranked_predictions = reranker_llama_stream(predictions=predictions, user_history=user_history, model_name=llm_model_name,
                                        titles=titles)

        # store predictions
        results_dict[user_id] = reranked_predictions

    nfm_reranker_dict[llm_model_name]["results"] = results_dict

    # evaluate re-ranker results by hit ratio
    evaluate_dict = {}
    for N in [5, 10, 15]:
        original_hit_ratio = get_hit_rate_at_n(N, predictions_dict, user_history_dict)

        new_hit_ratio = get_hit_rate_at_n(N, nfm_reranker_dict[llm_model_name]["results"], user_history_dict)

        # assume original hit ratio is not 0 (or it is pointless to re-rank)
        change = (new_hit_ratio - original_hit_ratio) / original_hit_ratio
        print(f"HR@{N} before:", original_hit_ratio, f"after:", new_hit_ratio, f"({'{:.2}'.format(change)})")

        # store evaluation results
        evaluate_dict[f"HR@{N}"] = [original_hit_ratio, new_hit_ratio, change]

    # store result
    nfm_reranker_dict[llm_model_name]["evaulate"] = evaluate_dict
    llm_model_name_short = llm_model_name.split("/")[-1]
    with open(f"reranker_results_{llm_model_name_short}.json", "w") as result_json_file:
        json.dump(nfm_reranker_dict[llm_model_name], result_json_file)


if __name__ == "__main__":
    # load Hugging Face access token from environment file
    load_dotenv()
    HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # load data files
    pairs=pd.read_csv("MicroLens-100k_pairs.csv")
    titles=pd.read_csv("MicroLens-100k_title_en.csv",header=None)
    titles.columns = ['item', 'title']

    # load inference results
    fname = "rec-nfm_NFM-Nov-16-2024_18-02-45.pth.txt"  # TODO: CHANGE THIS
    nfm_pre_dict, nfm_his_dict = parse_sampled_results_from_path(fname)  # nfm_his_dict is acquired here

    # result storage
    nfm_reranker_dict={}

    # re-rank: encoder + similarity comparison
    encoder_model_name = "intfloat/e5-small-v2"
    evaluate_reranker(llm_model_name=encoder_model_name, nfm_reranker_dict=nfm_reranker_dict, is_encoder=True, predictions_dict=nfm_pre_dict, user_history_dict=nfm_his_dict, titles=titles, pairs=pairs)

    # re-rank: prompt engineering
    llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    evaluate_reranker(llm_model_name=llama_model_name, nfm_reranker_dict=nfm_reranker_dict, is_encoder=False, predictions_dict=nfm_pre_dict, user_history_dict=nfm_his_dict, titles=titles, pairs=pairs)