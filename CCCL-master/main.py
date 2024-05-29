import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
import torch
from CCCL import CCCL
from trainer import CCCLTrainer


def run_single_model(args):
    config = Config(
        model=CCCL,
        dataset=args.dataset, 
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # 选择第一个可用的GPU设备



    model = CCCL(config, train_data.dataset).to(config[device])
    logger.info(model)

    trainer = CCCLTrainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='.')
    parser.add_argument('--config', type=str, default='', help='.')
    args, _ = parser.parse_known_args()

    args.config_file_list = [
        'properties/overall.yaml',
        'properties/CCCL.yaml'
    ]
    if args.dataset in ['ml-1m']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)

    run_single_model(args)
