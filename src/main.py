"""Main file for testing the results and generating submissions."""
import logging
import os
import argparse

from src.models import train_model
from src.models import predict_model
from src import DATA_DIR

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Poverty classification module')
    parser.add_argument('country', help='A or B or C')
    parser.add_argument('module', default='train',
                        help='train or predict. Note that only train is supported now, which logs a local cv score.')
    parser.add_argument('--out',
                        help='Output filename for predictions if predict mode is chosen. Stores in interim directory')
    args = parser.parse_args()
    assert args.country in ['A', 'B', 'C']
    assert args.module in ['train', 'predict']
    if args.module == 'predict':
        if not args.out:
            raise AttributeError('output filename is empty')
            logging.error('Please enter a output filename')
        else:
            fp = os.path.join(DATA_DIR, 'interim', args.out + '.csv')
            predict_model.main(args.country).to_csv(fp, index=False, header=False)
    else:
        _, _, _ = train_model.cv_setup(country=args.country)
