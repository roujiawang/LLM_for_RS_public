from cProfile import run
from logging import getLogger
import torch
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import sys
import re
import pandas as pd


def get_dataload(config_file="YAML/nfm.yaml"):
    config = Config(config_file_list=[config_file])
    dataload = load_data(config)
    return dataload


def get_raw_inference_results(config_file, model_file, output_file, dataload, device):
    """
    Writes to output_file the user history for user in each batch, 
    plus ground truth (the latest interaction), top predictions, and user history for users whose top predictions
    include the ground truth.
    Example excerpt:

    User 0 History: [12948, 8124, 287, 5398, 2628]
    User 1 History: [3844, 4543, 5519, 5291]
    User 2 History: [301, 7558, 3613, 11006]
    User 3 History: [14362, 4365, 5660, 6345]
    User 4 History: [3244, 12199, 40, 4541]
    User 5 History: [5893, 1110, 12329, 644, 324, 5326]
    User 6 History: [107, 108, 109, 110]
    User 7 History: [112, 113, 114, 115, 116, 117]
    User 8 History: [2208, 7151, 10569, 13769, 6382]
    User 9 History: [21, 22, 23, 24, 25]
    User 10 History: [1354, 3002, 509, 4527, 827, 1893, 11211, 297]
    User 11 History: [49, 50, 51, 52]
    User 12 History: [11, 12, 13, 14, 15, 16, 17, 18, 19]
    User 13 History: [18751, 4587, 10011, 15367]
    User 14 History: [14392, 5052, 16034, 171, 4850]
    User 15 History: [7169, 3080, 1868, 382]
    User 16 History: [13055, 7205, 2972, 9396]
    User 17 History: [6108, 2014, 34, 8656, 12783, 2678, 13052]
    User 18 History: [3892, 2228, 9434, 354, 1643, 40, 7944, 3402]
    User 19 History: [2135, 4585, 3806, 2137, 1165, 6460, 1196, 977, 505]
    User 6:
      Ground Truth: [111]
      Predictions: [297, 4275, 8110, 15396, 7617, 10707, 711, 2601, 2125, 1323, 9185, 3637, 2266, 1722, 111, 4280, 2413, 5153, 3, 4038]
      Matches:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
      History:     [0, 107, 108, 109, 110]
    """

    config = Config(config_file_list=[config_file])
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    model = get_model(config['model'])(config, dataload).to(device)

    trainer = Trainer(config, model)
    trainer.device = device

    with open(output_file, 'w') as outfile:
        default_stdout = sys.stdout
        try:
            # redirect output to txt file
            sys.stdout = outfile
            test_result = trainer.evaluate(test_loader, model_file= f"NFM/saved/{model_file}", load_best_model=True, show_progress=True)
        finally:
            sys.stdout = default_stdout


def refine_raw_output(raw_filename, refined_filename, dataload):
    """
    From the raw inference results, extract only lines about
    users whose top predictions include the ground truth.
    Example excerpt:

    User 6:
    Ground Truth: [11737]
    Predictions: [16748,14067,7177,5620,8984,8092,13929,17521,2489,339,16088,11833,14925,10310,11737,6952,54,13079,15223,8773]
    History: [9419,16951,12596,7566]
    """

    num_matched_user = 0  # record number of users for which the predictions contain the ground truth

    with open(raw_filename, 'r') as infile, open(refined_filename, 'w') as outfile:
        lines = infile.readlines()
        user_data = []
        is_user_section = False

        for line in lines:
            line = line.strip()  # Remove leading and trailing spaces

            # Start a new user section
            match_user = re.match(r"User (\d+):", line)
            if match_user:
                num_matched_user += 1

                if user_data:
                    # Write the previous user data if any
                    outfile.write('\n'.join(user_data) + '\n\n')
                    user_data = []  # Clear the previous data
                is_user_section = True
                user_data.append(f"User {match_user.group(1)}:")
                continue

            if is_user_section:
                # Capture Ground Truth line
                match_ground_truth = re.match(r"Ground Truth: \[(.*)\]", line)
                if match_ground_truth:
                    ground_truth_raw = [int(x.strip()) for x in match_ground_truth.group(1).split(',')]
                    ground_truth_orig = dataload.lookup_original_ids(ground_truth_raw)  # map back to original item ids
                    ground_truth_orig_str = ','.join(map(str, ground_truth_orig))
                    user_data.append(f"Ground Truth: [{ground_truth_orig_str}]")
                    continue

                # Capture Predictions line
                match_predictions = re.match(r"Predictions: \[(.*)\]", line)
                if match_predictions:
                    predictions_raw = [int(x.strip()) for x in match_predictions.group(1).split(',')]
                    predictions_orig = dataload.lookup_original_ids(predictions_raw)  # map back to original item ids
                    predictions_orig_str = ','.join(map(str, predictions_orig))
                    user_data.append(f"Predictions: [{predictions_orig_str}]")
                    continue

                # Capture History line (accounting for extra spaces)
                match_history = re.match(r"History:\s*\[(.*?)\]", line)
                if match_history:
                    history_raw = [int(x) for x in match_history.group(1).split(',') if int(x) != 0]
                    history_orig = dataload.lookup_original_ids(history_raw)  # map back to original item ids
                    history_orig_str = ','.join(map(str, history_orig))
                    user_data.append(f"History: [{history_orig_str}]")
                    continue

        # If there is any remaining user data at the end of the file
        if user_data:
            outfile.write('\n'.join(user_data) + '\n')

    print(num_matched_user)  # sanity check


def complete_refined_output(refined_filename, complete_filename, csv_filename='ks/ks.inter'):
    """
    From the refined inference results, for users whose top predictions include the ground truth,
    find corresponding true user ids given their latest history.
    """

    df = pd.read_csv(csv_filename, header=0, names=['item_id', 'user_id', 'timestamp'])
    user_item_sets = df.groupby('user_id')['item_id'].apply(set).to_dict()

    def get_item_with_largest_timestamp(user_id):
        user_data = df[df['user_id'] == user_id]  # Filter for the specific user
        if not user_data.empty:
            return user_data.loc[user_data['timestamp'].idxmax(), 'item_id']  # Get the item with the max timestamp
        return None

    def find_user_list_given_item_subset(item_ids_to_check):
        '''Assumption: the last item is the most recent.'''

        # Check if each user_id's set of item_ids contains all item_ids in item_ids_to_check
        subset_results = {
            user_id: user_item_sets[user_id].issubset(item_ids_to_check)
            for user_id in user_item_sets if user_id in user_item_sets
        }

        # Find the user_ids whose item_ids contain all of the item_ids in item_ids_to_check
        subset_results = {
            user_id: all(item_id in user_item_sets[user_id] for item_id in item_ids_to_check)
            for user_id in user_item_sets
        }

        # Filter and make a list of only the user_ids where the result is True
        true_subset_results = [user_id for user_id, result in subset_results.items() if result]

        # Check if the most recent item aligns
        true_subset_results = [user_id for user_id in true_subset_results if get_item_with_largest_timestamp(user_id) == item_ids_to_check[-1]]

        return true_subset_results

    def process_block(block, outfile):
        ground_truth_line, predictions_line, history_line = block[1], block[2], block[3]

        # Extract ground truth and history values
        ground_truth = eval(ground_truth_line.split(":")[1].strip())
        history = eval(history_line.split(":")[1].strip())

        # Combine ground truth and history
        input_items = history + ground_truth

        # Find real user IDs
        real_user_ids = find_user_list_given_item_subset(input_items)
        if not real_user_ids:
            raise ValueError(f"Error: No user IDs found for input items: {input_items}")

        # Modify the first line with real user IDs
        new_user_line = f"User {real_user_ids}:"

        # Write the modified block to the output file
        outfile.write(new_user_line + '\n')
        outfile.write(predictions_line + '\n')
        outfile.write(history_line + '\n')
        outfile.write(ground_truth_line + '\n\n')

    def process_file(input_file, output_file):
        num_matched_users = 0  # count the number of processed users

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            block = []
            for line in infile:
                if line.strip():
                    block.append(line.strip())
                else:  # End of a block or empty line
                    if block:
                        process_block(block, outfile)
                        num_matched_users += 1
                    block = []

            # Process the last block
            if block:
                process_block(block, outfile)
                num_matched_users += 1

        print(num_matched_users)  # sanity check

    try:
        process_file(refined_filename, complete_filename)
        print(f"Processing completed. Modified content written to {complete_filename}.")
    except ValueError as e:
        print(str(e))


if __name__ == "__main__":
    model_file = "NFM-Nov-16-2024_18-02-45.pth"  # TODO: CHANGE THIS

    output_file_raw = f"inference_correct_raw_{model_file}.txt"  # items are pseudo ids
    output_file_refined = f"inference_correct_{model_file}.txt"  # items are original ids
    output_file_complete = f"rec-nfm_{model_file}.txt"  # users are also original ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_file = "YAML/nfm.yaml"
    csv_filename = "ks/ks.inter"

    dataload = get_dataload(config_file)  # need this to be synced throughout inference processing

    get_raw_inference_results(config_file=config_file, model_file=model_file, output_file=output_file_raw, dataload=dataload, device=device)
    refine_raw_output(raw_filename=output_file_raw, refined_filename=output_file_refined, dataload=dataload)
    complete_refined_output(refined_filename=output_file_refined, complete_filename=output_file_complete, csv_filename=csv_filename)