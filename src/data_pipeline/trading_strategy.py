import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from src.data_pipeline.ml_prediction import LinearRegressionModel, RandomForestModel
from src.data_pipeline.utils import get_most_recent_folder
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_pnl(price_data: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PnL from price data and trade log
    
    Parameters:
    price_data: DataFrame with columns ['date' (int), 'close']
    trade_log: DataFrame with columns ['date' (int), 'old_position', 'new_position']
    
    Returns:
    DataFrame with prices, positions and PnL
    """
    # Make copies to avoid modifying original data
    price_data = price_data.copy()
    trade_log = trade_log.copy()
    
    # Sort by integer dates
    price_data = price_data.sort_values('date')
    trade_log = trade_log.sort_values('date')
    
    # Merge trade log with prices using integer dates
    merged_data = pd.merge_asof(
        price_data,
        trade_log,
        on='date',
        direction='forward'
    )
    
    # Forward fill new_position and fill any remaining NAs with 0
    merged_data['position'] = merged_data['new_position'].ffill().fillna(0)
    
    # Calculate daily returns
    merged_data['returns'] = merged_data['Adj Close'].pct_change()
    
    # Calculate strategy returns (position * next day's return)
    merged_data['strategy_returns'] = merged_data['position'].shift(1) * merged_data['returns']
    
    # Calculate cumulative returns
    merged_data['cumulative_returns'] = (1 + merged_data['strategy_returns']).cumprod()
    
    return merged_data

# def plot_trades(price_data: pd.DataFrame, trade_log: pd.DataFrame):
#     """
#     Plot price chart with entry and exit points
    
#     Parameters:
#     price_data: DataFrame with columns ['date' (int), 'Adj Close']
#     trade_log: DataFrame with columns ['date' (int), 'old_position', 'new_position']
#     """
#     plt.figure(figsize=(15, 8))
    
#     # Plot price
#     # plt.plot(price_data['date'], price_data['Adj Close'], label='Price', alpha=0.7)
#     plt.plot(price_data.index, price_data['Adj Close'], label='Price', alpha=0.7)

    
#     # Long entries (from 0 to 1)
#     long_entries = trade_log[
#         (trade_log['old_position'] == 0) & (trade_log['new_position'] == 1)
#     ]
    
#     # Long exits (from 1 to 0)
#     long_exits = trade_log[
#         (trade_log['old_position'] == 1) & (trade_log['new_position'] == 0)
#     ]
    
#     # Short entries (from 0 to -1)
#     short_entries = trade_log[
#         (trade_log['old_position'] == 0) & (trade_log['new_position'] == -1)
#     ]
    
#     # Short exits (from -1 to 0)
#     short_exits = trade_log[
#         (trade_log['old_position'] == -1) & (trade_log['new_position'] == 0)
#     ]
    
#     # Plot entry and exit points
#     # For long trades
#     # plt.scatter(long_entries['date'],
#     #              price_data.loc[price_data['date'].isin(long_entries['date']), 'Adj Close'],
#     plt.scatter(long_entries.index, 
#                price_data.loc[price_data.index.isin(long_entries.index), 'Adj Close'],
#                marker='^', color='green', s=100, label='Long Entry')
#     # plt.scatter(long_exits['date'], 
#     #            price_data.loc[price_data['date'].isin(long_exits['date']), 'Adj Close'],
#     #            marker='v', color='red', s=100, label='Long Exit')
#     plt.scatter(long_exits.index, 
#                price_data.loc[price_data.index.isin(long_exits.index), 'Adj Close'],
#                marker='v', color='red', s=100, label='Long Exit')
    
#     # For short trades
#     # plt.scatter(short_entries['date'], 
#     #            price_data.loc[price_data['date'].isin(short_entries['date']), 'Adj Close'],
#     #            marker='v', color='purple', s=100, label='Short Entry')
#     plt.scatter(short_entries.index, 
#                price_data.loc[price_data.index.isin(short_entries.index), 'Adj Close'],
#                marker='v', color='purple', s=100, label='Short Entry')
#     # plt.scatter(short_exits['date'], 
#     #            price_data.loc[price_data['date'].isin(short_exits['date']), 'Adj Close'],
#     #            marker='^', color='orange', s=100, label='Short Exit')
#     plt.scatter(short_exits.index, 
#                price_data.loc[price_data.index.isin(short_exits.index), 'Adj Close'],
#                marker='^', color='orange', s=100, label='Short Exit')
    
#     plt.title('Price Chart with Trade Points')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def print_performance_summary(results: pd.DataFrame, trade_log: pd.DataFrame):
    """Print key performance metrics"""
    total_return = (results['cumulative_returns'].iloc[-1] - 1) * 100
    n_trades = len(trade_log[trade_log['old_position'] != trade_log['new_position']])
    win_rate = (results['strategy_returns'] > 0).mean() * 100
    
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {n_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    if n_trades > 0:
        print(f"Average Return per Trade: {total_return/n_trades:.2f}%")

class EnhancedTradingStrategy:
    def __init__(
        self,
        lambda_turnover: float = 0.001,    # Turnover penalty parameter
        half_life: int = 20,               # Half-life for exponential decay of turnover
        base_threshold: float = 0.005,     # Base threshold for trades
        stop_loss: float = 0.05,           # Stop loss threshold
        pred_window: int = 5,              # Window for averaging predictions
        price_window: int = 20,            # Window for price moving average
        momentum_window: int = 10          # Window for momentum calculation
    ):
        self.lambda_turnover = lambda_turnover
        self.half_life = half_life
        self.base_threshold = base_threshold
        self.stop_loss = stop_loss
        self.pred_window = pred_window
        self.price_window = price_window
        self.momentum_window = momentum_window
        
        # Internal state
        self.current_position = 0
        self.entry_price = None
        self.position_history = []
        
        # Store historical data
        self.prediction_history = [] # these are the predictions from the ML forecast
        self.price_history = [] # these are just the prices
        
    def update_history(self, prediction: float, price: float):
        """Builds a prediction and price history of the most max_window
            recent days
        """

        """Update historical data arrays"""
        self.prediction_history.append(prediction)
        self.price_history.append(price)
        
        # Keep only necessary history
        max_window = max(self.pred_window, self.price_window, self.momentum_window)
        if len(self.prediction_history) > max_window:
            self.prediction_history = self.prediction_history[-max_window:]
            self.price_history = self.price_history[-max_window:]
    
    def calculate_signals(self) -> dict:
        """Calculate various trading signals based on historical data"""
        if len(self.price_history) < max(self.pred_window, self.price_window, self.momentum_window):
            return {
                'pred_ma': None,
                'price_trend': None,
                'momentum': None
            }
        
        # 1. Moving average of predictions - most recent 5 day window in this example
        pred_ma = np.mean(self.prediction_history[-self.pred_window:]) 
        
        # 2. Price trend (current price relative to moving average) - most recent 20 day window in this example
        price_ma = np.mean(self.price_history[-self.price_window:])
        current_price = self.price_history[-1]
        price_trend = (current_price - price_ma) / price_ma
        
        # 3. Momentum (rate of change over momentum window)
        momentum_start = self.price_history[-self.momentum_window]
        momentum = (current_price - momentum_start) / momentum_start # price pct change from momentum_window days ago
        
        return {
            'pred_ma': pred_ma,
            'price_trend': price_trend,
            'momentum': momentum
        }
    
    def get_position_signal(
        self,
        predicted_return: float,
        current_price: float,
        current_date: int
    ) -> tuple[int, str, dict]:
        """
        Determine position action based on multiple signals
        Returns: (new_position, reason, signal_values)
        """
        # Update historical data
        self.update_history(predicted_return, current_price)
        
        # Check stop loss first
        if self.current_position != 0 and self.entry_price is not None: # False at least until we start building positions
            pnl = (current_price - self.entry_price) / self.entry_price * self.current_position
            if pnl <= -self.stop_loss:
                return 0, "stop_loss", {}
        
        # Calculate all signals
        signals = self.calculate_signals()
        if None in signals.values():  # Not enough history
            return self.current_position, "insufficient_history", signals
        
        # Dynamic threshold based on turnover
        dynamic_threshold = self.base_threshold * (1 + self.calculate_turnover_penalty(current_date)) # no position history --> turnover_penalty = 0; dynamic_threshold = base_threshold
        
        # Combined signal logic
        pred_signal = signals['pred_ma']
        # trend_confirmation = signals['price_trend'] * signals['momentum'] # if same sign, this is positive
        
        if self.current_position == 0:  # Currently flat (and starting point)
            # Long entry: all components are positive
            if pred_signal > dynamic_threshold and signals['price_trend'] > 0 and signals['momentum'] > 0:
                return 1, "long_entry", signals
            # Short entry: all components are negative
            elif pred_signal < -dynamic_threshold and signals['price_trend'] < 0 and signals['momentum'] < 0:
                return -1, "short_entry", signals
            return 0, "stay_flat", signals
            
        elif self.current_position == 1:  # Currently long
            if pred_signal < -dynamic_threshold or signals['price_trend'] < 0 or signals['momentum'] < 0:
                return 0, "close_long", signals
            return 1, "maintain_long", signals

        else:  # Currently short
            if pred_signal > dynamic_threshold or signals['price_trend'] > 0 or signals['momentum'] > 0:
                return 0, "close_short", signals
            return -1, "maintain_short", signals


    def calculate_turnover_penalty(self, date: int) -> float:
        """Calculate turnover penalty using exponential decay"""
        if not self.position_history:
            return 0
        
        penalties = []
        for trade_date, _ in self.position_history:
            time_diff = date - trade_date
            decay = np.exp(-np.log(2) * time_diff / self.half_life)
            penalties.append(decay)
            
        return sum(penalties) * self.lambda_turnover

    def update_position(self, new_position: int, date: int, price: float):
        """Update strategy state with new position"""
        if new_position != self.current_position:
            self.position_history.append((date, abs(new_position - self.current_position)))
            cutoff_date = date - 2 * self.half_life
            self.position_history = [(d, p) for d, p in self.position_history if d > cutoff_date]
            self.entry_price = price if new_position != 0 else None
        
        self.current_position = new_position

def backtest_strategy(
    predictions: pd.Series,
    prices: pd.Series,
    strategy: EnhancedTradingStrategy
) -> tuple[pd.DataFrame, pd.Series]:
    """Backtest the strategy and return trade log"""
    trades = []
    turnover = 0
    
    for date, (pred, price) in enumerate(zip(predictions, prices)):
        new_position, reason, signals = strategy.get_position_signal(
            predicted_return=pred,
            current_price=price,
            current_date=date
        )
        
        if new_position != strategy.current_position:
            position_change = abs(new_position - strategy.current_position)
            turnover += position_change
            
            trades.append({
                'date': date,
                'price': price,
                'old_position': strategy.current_position,
                'new_position': new_position,
                'predicted_return': pred,
                'pred_ma': signals.get('pred_ma'),
                'price_trend': signals.get('price_trend'),
                'momentum': signals.get('momentum'),
                'turnover_penalty': strategy.calculate_turnover_penalty(date),
                'cumulative_turnover': turnover,
                'reason': reason
            })
            
            strategy.update_position(new_position, date, price)
        else:
            strategy.current_position = new_position
    
    trade_df = pd.DataFrame(trades)
    
    if len(trade_df) > 0:
        trade_df['time_between_trades'] = trade_df['date'].diff()
        
        metrics = {
            'total_trades': len(trade_df),
            'total_turnover': turnover,
            'avg_time_between_trades': trade_df['time_between_trades'].mean()
        }
        
        return trade_df, pd.Series(metrics)
    
    return pd.DataFrame(), pd.Series()




def visualize_trading_strategy(prices: pd.Series, trades_df: pd.DataFrame):
    """
    Create basic visualizations for trading strategy performance
    
    Parameters:
    prices: pd.Series - Price series indexed by date
    trades_df: pd.DataFrame - DataFrame containing trade information from backtest_strategy()
    """
    # Set style
    plt.style.use('ggplot')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    
    # Plot 1: Price chart with entry/exit points
    ax1.plot(prices.index, prices.values, label='Price', color='gray', alpha=0.7)
    
    # Plot trade points
    for _, trade in trades_df.iterrows():
        if trade['old_position'] == 0:  # Entry
            if trade['new_position'] == 1:  # Long entry
                ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=100, label='Long Entry')
            elif trade['new_position'] == -1:  # Short entry
                ax1.scatter(trade['date'], trade['price'], color='red', marker='v', s=100, label='Short Entry')
        else:  # Exit
            ax1.scatter(trade['date'], trade['price'], color='black', marker='x', s=100, label='Exit')
    
    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    ax1.set_title('Trading Strategy: Price and Signals')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Calculate PnL
    pnl = []
    current_position = 0
    cumulative_pnl = 0
    
    for i in range(len(prices)-1):
        # Update position if there was a trade
        if i in trades_df['date'].values:
            trade = trades_df[trades_df['date'] == i].iloc[0]
            current_position = trade['new_position']
        
        # Calculate daily PnL
        daily_return = (prices.iloc[i+1] - prices.iloc[i]) / prices.iloc[i]
        daily_pnl = current_position * daily_return
        cumulative_pnl += daily_pnl
        pnl.append(cumulative_pnl)
    
    # Plot 2: Cumulative PnL
    ax2.plot(prices.index[1:], np.array(pnl) * 100, label='Cumulative PnL %', color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Cumulative PnL (%)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Print summary statistics
    print("\nStrategy Summary:")
    print(f"Total number of trades: {len(trades_df)}")
    print(f"Final PnL: {pnl[-1]*100:.2f}%")
    print(f"Average trade duration: {trades_df['time_between_trades'].mean():.1f} days")
    print(f"Total turnover: {trades_df['cumulative_turnover'].iloc[-1]:.1f}")
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Distribution of trade reasons
    plt.figure(figsize=(10, 5))
    reason_counts = trades_df['reason'].value_counts()
    sns.barplot(x=reason_counts.index, y=reason_counts.values)
    plt.title('Distribution of Trade Reasons')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_dotenv()
    input_dir = os.getenv("LOCAL_DATA_DIR")
    logging_dir = get_most_recent_folder(os.getenv("LOGGING_DIR"))
    results_dir = get_most_recent_folder(os.getenv("RESULTS_DIR"))

    filename = os.path.join(results_dir, "model_results.pkl")
    j = joblib.load(filename)

    # optimal model - hard code for testing right now
    df = pd.DataFrame(j['Random Forest']['cv_results'])
    df_s = df[df.ErrorType !='MAPE']
    cv_dfs = []
    for idx, row in df_s.iterrows():
        test_index = row.TestIndex
        preds = row.yPred
        y_true = row.yTrue
        preds_df = pd.DataFrame(index=test_index, data={'yPred': preds, 'yTrue': y_true})
        cv_dfs.append(preds_df)

    cv_df = pd.concat(cv_dfs, axis=0)
    

    # get price data
    price_data = yf.download('DUK')
    close_prices = price_data['Adj Close']
    close_prices = close_prices.loc[cv_df.index]
    

    # print(cv_df[['yPred']].describe())

    # Initialize strategy
    strategy = EnhancedTradingStrategy(
        lambda_turnover=0.001,      # Turnover penalty parameter
        half_life=20,               # Half-life for exponential decay of turnover
        base_threshold=0.005,       # Base threshold for trades
        stop_loss=0.05,             # Stop loss threshold
        pred_window=5,              # Window for averaging predictions
        price_window=60,            # Window for price moving average
        momentum_window=30          # Window for momentum calculation
    )

    # Backtest
    trade_log, metrics = backtest_strategy(
        predictions=cv_df['yPred'],
        prices=close_prices,
        strategy=strategy
    )
    
    trade_log.to_csv(os.path.join(results_dir, 'trade_log.csv'), index=False)
    trade_log.set_index('date', inplace=True)
    close_prices = close_prices.to_frame()
    close_prices.reset_index(inplace=True)
    close_prices.index.name = 'date'
    close_prices.to_csv(os.path.join(results_dir, 'close_prices.csv'), index=True)

    print('Trading results finished processing!')