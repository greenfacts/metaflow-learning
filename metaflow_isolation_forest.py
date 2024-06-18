from metaflow import FlowSpec, step, Parameter, card


class IsolationForestFlow(FlowSpec):
    n_estimators = Parameter(
        "n_estimators", default=100, help="Number of trees in the forest"
    )
    max_samples = Parameter(
        "max_samples",
        default=256,
        help="Number of samples to draw from the data to train each tree",
    )
    contamination = Parameter(
        "contamination", default=0.1, help="Proportion of outliers in the data"
    )
    random_state = Parameter("random_state", default=42, help="Random seed")

    @step
    def start(self):
        # get data
        import numpy as np

        np.random.seed(self.random_state)
        normal_data = np.random.normal(loc=0, scale=1, size=(5000, 10))
        anomalies = np.random.uniform(low=-5, high=5, size=(100, 10))
        self.data = np.vstack((normal_data, anomalies))
        self.labels = np.zeros(len(self.data))
        self.labels[len(normal_data) :] = 1

        self.next(self.train)

    @card
    @step
    def train(self):
        # train the isolation forest
        from sklearn.ensemble import IsolationForest
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd

        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=self.random_state
        )
        self.X_test = X_test
        self.y_test = y_test

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self.next(self.end)

    @step
    def end(self):
        # test the model
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            recall_score,
            precision_score,
            f1_score,
        )

        y_pred = self.model.predict(self.X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        print(f"Recall: {self.recall}")
        print(f"Precision: {self.precision}")
        print(f"F1: {self.f1}")


if __name__ == "__main__":
    IsolationForestFlow()
