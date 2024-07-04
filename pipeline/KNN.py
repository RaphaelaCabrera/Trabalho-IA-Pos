from pipeline.utils import Utils


class KNN:
    def __init__(self):
        self.utils = Utils()
        self.param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

    def execute(self, X_train, y_train, X_test, y_test, dataset: str):
        best_model = self.utils.set_best_model(X_train, y_train, 'KNN', self.param_grid)
        y_pred = best_model.predict(X_test)
        self.utils.get_metrics(y_pred, y_test, 'KNN', dataset)
