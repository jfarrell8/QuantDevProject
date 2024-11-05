from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class TimeSeriesModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

    # @abstractmethod
    # def evaluate(self, X_test, y_test):
    #     y_pred = self.predict(X_test)
    #     return mean_squared_error(y_test, y_pred)
    

class LinearRegressionModel(TimeSeriesModel):
    def __init__(self, **params):
        if params:
            self.model = LinearRegression(**params)
        else:
            self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class RandomForestModel(TimeSeriesModel):
    def __init__(self, **params):
        if params:
            self.model = RandomForestRegressor(**params)
        else:
            self.model = RandomForestRegressor()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)