import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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

    def get_metrics(self, y_pred, y_test, algorithm: str):
        self.logger().info(f'Métricas {algorithm}')
        self.logger().info(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        self.logger().info(f'Precision: {precision_score(y_test, y_pred)}')
        self.logger().info(f'Recall: {recall_score(y_test, y_pred)}')
        self.logger().info(f'F1 Score: {f1_score(y_test, y_pred)}')
        self.logger().info(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')

