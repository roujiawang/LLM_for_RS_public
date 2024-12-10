import torch
import pandas as pd
from dotenv import load_dotenv

import os
import random

import argparse

from data_preprocess import *
from nfm_inference import *
from llm_rerank import *

def preprocess_data():
    tsv_to_csv()
    generate_popnpy()


def train_nfm():
    '''NFM/main.py'''

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = '0'  

    master_port = random.randint(1002,9999)

    nproc_per_node = len(device.split(','))
    
    run_yaml = f"CUDA_VISIBLE_DEVICES='{device}'  python  -m torch.distributed.run --nproc_per_node {nproc_per_node} \
--master_port {master_port} run.py --config_file YAML/nfm.yaml" 
    os.system(run_yaml)


def nfm_rs_inference(device):
    model_file = "NFM-Nov-16-2024_18-02-45.pth"  # TODO: CHANGE THIS

    output_file_raw = f"inference_correct_raw_{model_file}.txt"  # items are pseudo ids
    output_file_refined = f"inference_correct_{model_file}.txt"  # items are original ids
    output_file_complete = f"rec-nfm_{model_file}.txt"  # users are also original ids

    config_file = "YAML/nfm.yaml"
    csv_filename = "ks/ks.inter"

    dataload = get_dataload(config_file)  # need this to be synced throughout inference processing

    get_raw_inference_results(config_file=config_file, model_file=model_file, output_file=output_file_raw, dataload=dataload, device=device)
    refine_raw_output(raw_filename=output_file_raw, refined_filename=output_file_refined, dataload=dataload)
    complete_refined_output(refined_filename=output_file_refined, complete_filename=output_file_complete, csv_filename=csv_filename)


def llm_rerank_list(encoder_model_name="intfloat/e5-small-v2", llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"):
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
    if encoder_model_name:
        evaluate_reranker(llm_model_name=encoder_model_name, nfm_reranker_dict=nfm_reranker_dict, is_encoder=True, predictions_dict=nfm_pre_dict, user_history_dict=nfm_his_dict, titles=titles, pairs=pairs)

    # re-rank: prompt engineering
    if llama_model_name:
        evaluate_reranker(llm_model_name=llama_model_name, nfm_reranker_dict=nfm_reranker_dict, is_encoder=False, predictions_dict=nfm_pre_dict, user_history_dict=nfm_his_dict, titles=titles, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description="Run the LLM models of interest.")
    parser.add_argument("--encoder_model", type=str, default="intfloat/e5-small-v2",
                        help="Name of the Hugging Face encoder model (default: intfloat/e5-small-v2)")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Name of the Hugging Face LLaMA model (default: meta-llama/Meta-Llama-3-8B-Instruct)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model_name= args.encoder_model
    llama_model_name = args.llama_model

    preprocess_data()
    train_nfm()
    nfm_rs_inference(device)
    llm_rerank_list(encoder_model_name, llama_model_name)


if __name__ == "__main__":
    main()