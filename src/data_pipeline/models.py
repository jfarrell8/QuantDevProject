from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.tsa.statespace.sarimax as sm

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
    

class SARIMAXModel(TimeSeriesModel):
    def __init__(self, order, seasonal_order, method='powell', maxiter=100, trend='n', **kwargs):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.kwargs = kwargs
        self.model = None
        self.last_train_index = None
    
    def fit(self, X_train, y_train):
        self.train_index = y_train.index
        self.last_train_index = len(self.train_index) - 1

        self.model = sm.SARIMAX(
            endog=y_train,
            exog=X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            **self.kwargs
        ).fit()

    def predict(self, X_test):
        start = self.last_train_index + 1
        end = start + len(X_test) - 1
        # if not SARIMAX:
        #     exog=None
        # else:
        #     exog=X_test
        return self.model.predict(exog=X_test, start=start, end=end)