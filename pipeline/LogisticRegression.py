from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pipeline.utils import Utils


class LogisticRegression:
    def __init__(self):
        self.logger = Utils.logger()
        self.param_grid = {
                            'C': [0.1, 1, 10],
                            'solver': ['lbfgs', 'liblinear']
                          }

    def execute(self, X_train, y_train, X_test, y_test):
        best_model = self.set_best_model(X_train, y_train)
        y_pred = best_model.predict(X_test)
        self.logger().info(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    def set_best_model(self, X_train, y_train):
        grid_search = GridSearchCV(LogisticRegression(), self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
