import pandas as pd

from pipeline.utils import Utils
from pipeline.preprocess import Preprocess
from pipeline.LogisticRegression import LogisticRegression
from pipeline.KNN import KNN
from pipeline.SVM import SVM
from pipeline.DecisionTrees import DecisionTrees
from pipeline.RandomForest import RandomForest


class Pipeline:
    def __init__(self):
        self.utils = Utils()
        self.preprocess = Preprocess()
        self.logistic_regression = LogisticRegression()
        self.knn = KNN()
        self.svm = SVM()
        self.decision_trees = DecisionTrees()
        self.random_forest = RandomForest()

    def execute(self, dataset: str):
        dataframe = pd.DataFrame()
        self.utils.logger(dataset).info('Início da execução do pipeline')
        dataframe = self.read_dataset(dataframe, dataset)
        self.utils.logger(dataset).info('Início do preprocessamento do dataset')
        X_train, X_test, y_train, y_test = self.preprocess.execute(dataframe, dataset)
        self.utils.logger(dataset).info('Executando Regressão Logística')
        self.logistic_regression.execute(X_train, y_train, X_test, y_test, dataset)
        self.utils.logger(dataset).info('Executando KNN')
        self.knn.execute(X_train, y_train, X_test, y_test, dataset)
        self.utils.logger(dataset).info('Executando SVM')
        self.svm.execute(X_train, y_train, X_test, y_test, dataset)
        self.utils.logger(dataset).info('Executando Árvores de Decisão')
        self.decision_trees.execute(X_train, y_train, X_test, y_test, dataset)
        self.utils.logger(dataset).info('Executando Random Forest')
        self.random_forest.execute(X_train, y_train, X_test, y_test, dataset)
        self.utils.logger(dataset).info('Finalizada execução do pipeline')

    def read_dataset(self, dataframe, dataset):
        if dataset == 'dataset1':
            self.utils.logger(dataset).info('Leitura do dataset sobre doenças cardíacas')
            dataframe = self.utils.read_csv('heart_disease_uci.csv')
        elif dataset == 'dataset2':
            self.utils.logger(dataset).info('Leitura do dataset sobre derrame cerebral')
            dataframe = self.utils.read_csv('brain_stroke.csv')
        elif dataset == 'dataset3':
            self.utils.logger(dataset).info('Leitura do dataset sobre mortalidade de câncer de pulmão')
            dataframe = self.utils.read_csv('lung_cancer_mortality_data_test_v2.csv')
        elif dataset == 'dataset4':
            self.utils.logger(dataset).info('Leitura do dataset sobre estimativa dos níveis de obesidade')
            dataframe = self.utils.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        elif dataset == 'dataset5':
            self.utils.logger(dataset).info('Leitura do dataset sobre Recorrência de câncer de tireoide')
            dataframe = self.utils.read_csv('Thyroid_Diff.csv')
        return dataframe





