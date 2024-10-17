import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
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
    load_dotenv()
    input_dir = os.getenv("LOCAL_DATA_DIR")
    logging_dir = os.getenv("LOGGING_DIR")
    results_dir = os.getenv("RESULTS_DIR")

    filename = os.path.join(results_dir, "model_results.pkl")
    j = joblib.load(filename)
    # for model_name, model_info in j.items():
    #     print(model_name)
    #     df = pd.DataFrame(j[model_name]['cv_results'])
    #     df.to_excel(os.path.join(results_dir, f"{model_name.lower().replace(' ', '')}_model_results.xlsx"), index=False)

    # optimal model - hard code for testing right now
    df = pd.DataFrame(j['Linear Regression']['cv_results'])
    df_s = df[df.ErrorType !='MAPE']
    cv_dfs = []
    for idx, row in df_s.iterrows():
        test_index = row.TestIndex
        preds = row.yPred
        y_true = row.yTrue
        preds_df = pd.DataFrame(index=test_index, data={'yPred': preds, 'yTrue': y_true})
        cv_dfs.append(preds_df)

    cv_df = pd.concat(cv_dfs, axis=0)
    cv_df.to_excel(os.path.join(results_dir, f"linear_regression_IS_preds.xlsx"))