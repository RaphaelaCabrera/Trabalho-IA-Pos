import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.utils import Utils


class Preprocess:
    def __init__(self):
        self.utils = Utils()
        self.scaler = StandardScaler()

    def execute(self, dataframe: pd.DataFrame, dataset: str):
        coluna_alvo = ''
        if dataset == 'dataset1':
            dataframe = dataframe.dropna()
        categorical_columns = self.get_categorical_columns(dataframe)
        dataframe = self.process_categorical_columns(categorical_columns, dataframe)
        if dataset == 'dataset1':
            print("Colunas do dataframe:", dataframe.columns)
            coluna_alvo = 'num'
        elif dataset == 'dataset2':
            print("Colunas do dataframe:", dataframe.columns)
            coluna_alvo = 'stroke'
        x = dataframe.drop(coluna_alvo, axis=1)
        y = dataframe[coluna_alvo]
        X_train, X_test, y_train, y_test = self.utils.set_train_test_sets(x, y)
        numerical_columns = self.get_numerical_columns(dataframe)
        X_train[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        return X_train, X_test, y_train, y_test

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
