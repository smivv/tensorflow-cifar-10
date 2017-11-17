import argparse
import logging
from classes import ModelCifar10

logging.basicConfig(level=logging.INFO)


def serve(config):
    model = ModelCifar10.ModelCifar10(config.input_size, config.hidden_size, config.output_size)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', default='32', help='Input layer size of CNN.')

    logging.info('Input layer size argument passed..')

    parser.add_argument('--hidden_size', default='32', help='Hidden layer size of CNN.')

    logging.info('Hidden layer size argument passed..')

    parser.add_argument('--output_size', default='32', help='Output layer size of CNN.')

    logging.info('Output layer size argument passed..')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    logging.info('Processing started..')

    # Run service
    serve(config=args)
