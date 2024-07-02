from pipeline.utils import Utils


class DecisionTrees:
    def __init__(self):
        self.utils = Utils()
        self.param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }

    def execute(self, X_train, y_train, X_test, y_test):
        best_model = self.utils.set_best_model(X_train, y_train, 'KNN', self.param_grid)
        y_pred = best_model.predict(X_test)
        self.utils.get_metrics(y_pred, y_test, 'Árvores de Decisão')
