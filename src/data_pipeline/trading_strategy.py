import pandas as pd
import numpy as np
import joblib
from ml_prediction import LinearRegressionModel, RandomForestModel

class PriceDiffStrategy:
    def __init__(self, threshold=0.05, max_position=100, transaction_cost=0.001):
        self.threshold = threshold
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.position = 0

    def generate_signals(self, forecasts, day_ahead_prices):
        signals = pd.DataFrame(index=forecasts.index)
        signals['forecast'] = forecasts
        signals['day_ahead'] = day_ahead_prices
        signals['price_diff'] = signals['forecast'] - signals['day_ahead']

        # generate buy/sell signals
        signals['signal'] = np.where(signals['price_diff'] > self.threshold, 1,
                                     np.where(signals['price_diff'] < -self.threshold, -1, 0)) # <-- double check this
        
        
        return signals
    

if __name__ == "__main__":
    filename = r"C:\Users\Admin\Desktop\GitHubPortfolio\QuantDevProject\src\results\model_results.pkl"
    j = joblib.load(filename)
    df = pd.DataFrame(j['Linear Regression']['cv_results'])
    df.to_excel(r"C:\Users\Admin\Desktop\GitHubPortfolio\QuantDevProject\src\results\model_results_linearRegression.xlsx")