import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.utils import Utils


class Preprocess:
    def __init__(self):
        self.utils = Utils()
        self.scaler = StandardScaler()

    def execute(self, dataframe: pd.DataFrame, dataset: str):
        if dataset == 'dataset1':
            dataframe = dataframe.dropna()
        categorical_columns = self.get_categorical_columns(dataframe)
        dataframe = self.process_categorical_columns(categorical_columns, dataframe)
        coluna_alvo = self.get_coluna_alvo(dataset)
        x = dataframe.drop(coluna_alvo, axis=1)
        y = dataframe[coluna_alvo]
        X_train, X_test, y_train, y_test = self.utils.set_train_test_sets(x, y)
        numerical_columns_x = self.get_numerical_columns(x)
        X_train[numerical_columns_x] = self.scaler.fit_transform(X_train[numerical_columns_x])
        X_test[numerical_columns_x] = self.scaler.transform(X_test[numerical_columns_x])
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_coluna_alvo(dataset):
        if dataset == 'dataset1':
            return 'num'
        elif dataset == 'dataset2':
            return 'stroke'
        elif dataset == 'dataset3':
            return 'survived'
        elif dataset == 'dataset4':
            return 'NObeyesdad'
        elif dataset == 'dataset5':
            return 'Recurred'

    @staticmethod
    def get_categorical_columns(dataframe):
        return dataframe.select_dtypes(include=['object', 'category']).columns

    @staticmethod
    def get_numerical_columns(dataframe):
        return dataframe.select_dtypes(include=['int64', 'float64']).columns

    @staticmethod
    def process_categorical_columns(categorical_columns: list, dataframe: pd.DataFrame):
        for col in categorical_columns:
            dataframe = pd.get_dummies(dataframe, columns=[col])
        return dataframe
