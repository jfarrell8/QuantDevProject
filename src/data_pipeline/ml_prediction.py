import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from typing import Type, Any, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict

import logging
from utils import setup_logging, save_pickle_file
from dotenv import load_dotenv
import os
import sys




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
    






class CustomTimeSeriesSplit:
    def __init__(self, train_period, test_period, step_size=None, gap=0, expanding_window=False):
        self.train_period = train_period
        self.test_period = test_period
        # self.step_size = step_size if step_size is not None else test_period
        self.step_size = test_period
        self.gap = gap
        self.expanding = expanding_window

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # # Determine the number of splits based on available data
        # max_train_start = n_samples - self.train_period - self.test_period - self.gap
        # n_splits = (max_train_start // self.step_size) + 1

        max_splits = (n_samples - self.train_period - self.test_period - self.gap) // self.test_period + 1


        for i in range(max_splits):
            
            if self.expanding:
                train_start = 0
                train_end = self.train_period + i * self.test_period
            else:
                train_start = i*self.test_period
                train_end = train_start + self.train_period
            test_start = train_end + self.gap
            test_end = test_start + self.test_period

            if test_end > n_samples:
                break

            yield indices[train_start:train_end], indices[test_start:test_end]


class TimeSeriesModelRegistry:
    def __init__(self, logging_path: str, data: pd.DataFrame, final_test_size: int, YEAR: int = 252) -> None:
        self.models = {}
        self.YEAR = YEAR # tradeable days in a year
        self.logging_path = logging_path
        self.final_test_size = final_test_size
        self.data = data

    def register(self, name: str, model_class: Type[TimeSeriesModel], param_grid: dict = None) -> None:
        if param_grid is None:
            param_grid = {}
        
        self.models[name] = {
            'model_class': model_class,
            'param_grid': param_grid
            # 'best_params': None,
            # 'best_model': None,
            # 'cv_results': None
        }

    def perform_walk_forward_cv(self, train_period_length: int, test_period_length: int, \
                                n_jobs: int = -1, expanding_window: bool = True) -> dict:

        # Split out the final portion of data for out-of-sample evaluation
        train_data = self.data[:-self.final_test_size]
        
        # Initialize the custom time series splitter
        splitter = CustomTimeSeriesSplit(train_period_length, test_period_length, expanding_window=expanding_window)

        results = {}

        def process_model(model_name: str, model_info: dict) -> Tuple[str, ]:
            setup_logging(model_name, self.logging_path)
            logging.info(f"Processing {model_name}")

            model_class = model_info['model_class']
            param_grid = model_info['param_grid']
            best_error = np.inf
            best_params = None
            best_model_instance = None
            cv_results = {
                            'Params': [],
                            'Fold': [],
                            'TrainStartDate': [],
                            'TrainEndDate': [],
                            'TestStartDate': [],
                            'TestEndDate': [],
                            'ErrorType': [],
                            'ErrorVal': [],
                            'yTrue': [],
                            'yPred': []
                            }

            for params in self.parameter_grid(param_grid):
                errors = []
                model = model_class(**params)

                # Walk-forward cross-validation with linear regression
                for fold, (train_idx, test_idx) in enumerate(splitter.split(X=train_data)):
                    # Train/test split
                    X_train, y_train = train_data.iloc[train_idx, :-1], train_data.iloc[train_idx, -1]
                    X_test, y_test = train_data.iloc[test_idx, :-1], train_data.iloc[test_idx, -1]
                    # print(f'TRAIN start date: {X_train.index[0]} ... end date: {X_train.index[-1]}, length: {len(X_train)}')
                    # print(f'TEST start date: {X_test.index[0]} ... end date: {X_test.index[-1]}, length: {len(X_test)}')

                    # Train the model
                    model.fit(X_train, y_train)
                    # error = model.evaluate(X_test, y_test)
                    y_pred = model.predict(X_test)
                    rmse_error = root_mean_squared_error(y_pred, y_test)
                    mape_error = mean_absolute_percentage_error(y_pred, y_test)
                    errors.append(rmse_error)
                    
                    metadata_str = ['Params', 'Fold', 'TrainStartDate', 'TrainEndDate', 'TestStartDate', 'TestEndDate',\
                                    'ErrorType', 'ErrorVal', 'yTrue', 'yPred']
                    metadata_rmse_vals = [params, fold, X_train.index[0], X_train.index[-1], X_test.index[0], X_test.index[-1], \
                                     'RMSE', rmse_error, y_test.values, y_pred]
                    metadata_mape_vals = [params, fold, X_train.index[0], X_train.index[-1], X_test.index[0], X_test.index[-1], \
                                     'MAPE', mape_error, y_test.values, y_pred]
                    for key, rmse_val, mape_val in list(zip(metadata_str, metadata_rmse_vals, metadata_mape_vals)):
                        cv_results[key].append(rmse_val)
                        cv_results[key].append(mape_val)


                    print(f"    Fold {fold + 1} RMSE: {rmse_error}, MAPE: {mape_error}")

                avg_error = np.mean(errors)
                if avg_error < best_error:
                    best_error = avg_error
                    best_params = params
                    best_model_instance = model

            return model_name, best_params, best_model_instance, best_error, cv_results
        
        # parallel processing each model type
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(process_model)(model_name, model_info) for model_name, model_info in self.models.items()
        )
        
        for model_name, best_params, best_model_instance, best_error, cv_results in parallel_results:
            self.models[model_name]['best_params'] = best_params
            self.models[model_name]['best_model'] = best_model_instance
            self.models[model_name]['best_error'] = best_error
            self.models[model_name]['cv_results'] = cv_results
            results[model_name] = {'avg_error': best_error, 'best_params': best_params}
            print(f"Model: {model_name}, Best Params: {best_params}, CV RMSE: {best_error}")

        return results

    def out_of_sample_predict(self) -> dict:
        # perform out of sample forecast for best models
        # Final out-of-sample evaluation (train on the full training data, test on the hold-out set)
        train_data = self.data[:-self.final_test_size]
        out_of_sample_data = self.data[-self.final_test_size:]
        final_out_of_sample_results = {}

        X_train_full, y_train_full = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_out_of_sample, y_out_of_sample = out_of_sample_data.iloc[:, :-1], out_of_sample_data.iloc[:, -1]

        
        for model_name, model_info in self.models.items():
            model_class = model_info['model_class']
            best_params = model_info['best_params']
            if best_params is None:
                best_params = {}
            best_model = model_class(**best_params)
            best_model.fit(X_train_full, y_train_full)

            y_pred_out_sample = best_model.predict(X_out_of_sample)
            out_sample_rmse_error = root_mean_squared_error(y_out_of_sample, y_pred_out_sample)
            out_sample_mape_error = mean_absolute_percentage_error(y_out_of_sample, y_pred_out_sample)

            final_out_of_sample_results[model_name] = {'out_sample_rmse_error': out_sample_rmse_error, 
                                                       'out_sample_mape_error': out_sample_mape_error,
                                                       'y_pred_out_sample': y_pred_out_sample,
                                                       'y_out_sample': y_out_of_sample}
            print(f"Model: {model_name}, Out-of-Sample RMSE: {out_sample_rmse_error}, MAPE: {out_sample_mape_error}")

        return final_out_of_sample_results

    @staticmethod
    def parameter_grid(param_grid):
        if not param_grid:
            yield {}
        else:
            keys, values = zip(*param_grid.items())
            for v in product(*values):
                yield dict(zip(keys, v))

            

# Example usage
if __name__ == "__main__":
    load_dotenv()
    input_dir = os.getenv("LOCAL_DATA_DIR")
    logging_dir = os.getenv("LOGGING_DIR")
    results_dir = os.getenv("RESULTS_DIR")

    # From data_pipeline
    data = pd.read_csv(os.path.join(input_dir, "input_dataset.csv"), index_col='period')

    # Parameters
    train_period_length = 63
    test_period_length = 10
    # n_splits = int(3 * 252 / test_period_length)
    # n_splits=5
    # lookahead = 1

    registry = TimeSeriesModelRegistry(logging_path=logging_dir, data=data, final_test_size=test_period_length)
    
    # Linear regression model with no hyperparameters
    registry.register('Linear Regression', LinearRegressionModel, {})

    # RandomForest model with hyperparams
    rf_param_grid = {
        'n_estimators': [50],
        'max_depth': [5, 10]        
    }
    registry.register('Random Forest', RandomForestModel, rf_param_grid)
    
    
    cv_results = registry.perform_walk_forward_cv(train_period_length, test_period_length, n_jobs=-1, expanding_window=True)
    print("Final Walk-Forward Cross-Validation Results: ", cv_results)

    out_sample_results = registry.out_of_sample_predict()


    preds_df = pd.DataFrame(data=data.iloc[-test_period_length:, -1].values, columns=['y_true']) # start with the true value
    model_oos_errors = pd.DataFrame(columns=['oos_error'])
    
    for model_name in out_sample_results.keys():
        # compile out of sample errors for each model
        out_sample_error = out_sample_results[model_name]['out_sample_rmse_error']
        model_oos_errors = pd.concat([model_oos_errors, pd.DataFrame(index=[model_name], data=[out_sample_error], columns=['oos_error'])], axis=0)
        print(f"{model_name} Out-of-Sample error: ", out_sample_error)

        # look at preds vs true
        out_sample_preds = out_sample_results[model_name]['y_pred_out_sample']
        preds_df = pd.concat([preds_df, pd.DataFrame(data=out_sample_preds, columns=[model_name.replace(" ", "") + '_preds'])], axis=1)

    preds_df.index = data.index[-test_period_length:]
    preds_df.to_csv(os.path.join(results_dir, f'preds_df.csv'))
    
    print('##########################')
    print('##########################')
    print(model_oos_errors)
    print('##########################')
    print('##########################')

    model_oos_errors.to_csv(os.path.join(results_dir, f'model_out_of_sample_errors.csv'))


    model_results_filepath = os.path.join(results_dir, 'model_results.pkl')
    save_pickle_file(registry.models, model_results_filepath)