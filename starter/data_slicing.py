from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent
import os
import pandas as pd
import logging
from .starter.train_model import clean_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_slicing(df):
    class_labels = df['salary'].unique()

    for class_label in class_labels:
        logging.info("descriptive stats for {}:\n{}\n".format(
        class_label, df.loc[df['salary'] == class_label].describe()))

    logging.info("OK - data_slicing.py finished")
    return


if __name__ == '__main__':
    # load test data
    full_input_path = os.path.join('starter', 'data', 'census.csv')
    df = pd.read_csv(full_input_path)
    df = clean_data(df)
    data_slicing(df)
