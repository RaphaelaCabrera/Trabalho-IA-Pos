from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pipeline.utils import Utils


class LogisticRegression:
    def __init__(self):
        self.utils = Utils()
        self.param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }

    def execute(self, X_train, y_train, X_test, y_test, dataset: str):
        best_model = self.utils.set_best_model(X_train, y_train, 'LogisticRegression', self.param_grid)
        y_pred = best_model.predict(X_test)
        self.utils.get_metrics(y_pred, y_test, 'Regressão Logística', dataset)
