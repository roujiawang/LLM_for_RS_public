from cProfile import run
import logging
from logging import getLogger
import torch
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import os
import random
import numpy as np
import argparse


def run_loop(config_file=None, saved=True):
    config = Config(config_file_list=[config_file])

    # Set device to single GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # Initialize seed and logger
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    # Load data and model
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    model = get_model(config['model'])(config, dataload).to(device)

    # Trainer initialization
    trainer = Trainer(config, model)

    logger.info(config)
    logger.info(dataload)
    logger.info(model)

    # Train and evaluate
    best_valid_score, best_valid_result = trainer.fit(
        train_loader, valid_loader, saved=saved, show_progress=False
    )

    test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=False)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=-1, type=str)
    args = parser.parse_args()
    config_file = args.config_file

    # Run training loop
    run_loop(config_file=config_file)
