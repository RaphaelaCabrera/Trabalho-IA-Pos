from pipeline.utils import Utils


class SVM:
    def __init__(self):
        self.utils = Utils()
        self.param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4],
            'coef0': [0.0, 0.1, 0.5, 1.0]
        }

    def execute(self, X_train, y_train, X_test, y_test, dataset: str):
        best_model = self.utils.set_best_model(X_train, y_train, 'SVM', self.param_grid)
        y_pred = best_model.predict(X_test)
        self.utils.get_metrics(y_pred, y_test, 'SVM', dataset)
