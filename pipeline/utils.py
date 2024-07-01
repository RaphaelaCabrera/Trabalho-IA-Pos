import logging

import pandas as pd
from sklearn.model_selection import train_test_split


class Utils:
    @staticmethod
    def read_csv(file_name: str) -> pd.DataFrame:
        return pd.read_csv('../datasets/' + file_name)

    @staticmethod
    def set_train_test_sets(x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def logger():
        console_handler = logging.StreamHandler()
        lineformat = '[%(asctime)s] | %(levelname)s | [%(process)d - %(processName)s]: %(message)s'
        dateformat = '%d-%m-%Y %H:%M:%S'
        logging.basicConfig(format=lineformat, datefmt=dateformat, level=20, handlers=[console_handler])
        return logging.getLogger()

