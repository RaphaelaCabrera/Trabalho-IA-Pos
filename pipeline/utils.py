import logging
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Utils:
    @staticmethod
    def read_csv(file_name: str) -> pd.DataFrame:
        return pd.read_csv('datasets/' + file_name)

    @staticmethod
    def set_train_test_sets(x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def logger(dataset: str):
        console_handler = logging.StreamHandler()
        lineformat = '[%(asctime)s] | %(levelname)s | [%(process)d - %(processName)s]: %(message)s'
        file_handler = logging.FileHandler('resultados/metricas_' + dataset)
        dateformat = '%d-%m-%Y %H:%M:%S'
        logging.basicConfig(format=lineformat, datefmt=dateformat, level=20, handlers=[console_handler, file_handler])
        return logging.getLogger()

    def set_best_model(self, X_train, y_train, algorithm: str, param_grid):
        grid_search = self.get_grid_search_by_algorithm(algorithm, param_grid)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    @staticmethod
    def get_grid_search_by_algorithm(algorithm, param_grid):
        if algorithm == 'LogisticRegression':
            return GridSearchCV(LogisticRegression(), param_grid, cv=5)
        elif algorithm == 'KNN':
            return GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        elif algorithm == 'SVM':
            return GridSearchCV(SVC(), param_grid, cv=5)
        elif algorithm == 'DecisionTree':
            return GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        elif algorithm == 'RandomForest':
            return GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

    def get_metrics(self, y_pred, y_test, algorithm: str, dataset: str):
        self.logger(dataset).info(f"Resultados {algorithm}")
        metrics = self.calculate_metrics(y_test, y_pred)
        for metric, value in metrics.items():
            if metric != 'classification_report' and metric != 'confusion_matrix':
                self.logger(dataset).info(f"{metric}: {value}")
            else:
                self.logger(dataset).info(f"\n{metric}:\n{value}")

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return metrics

