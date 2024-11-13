# Quantitative Development Strategy Project

## Project Description
A machine learning-based quantitative trading strategy implementation focused on Duke Energy's stock returns. The project combines fundamental analysis, economic indicators, sentiment analysis, and weather data to predict 1-day-ahead returns and execute trading decisions.

## 🏗️ Architecture

### Data Pipeline
1. **Database Layer**: Custom Postgres schema for time series data storage
2. **Data Sources**:
   - Fundamental Factors: AlphaVantage API
   - Economic Indicators: U.S. Energy Information Administration (EIA)
   - Financial Sentiment: SEC EDGAR API (10-Q and 10-K filings)
   - Weather Data: Open-Meteo API

### Model Components
- Factor Model incorporating:
  - Fundamental analysis
  - Economic indicators
  - Sentiment scoring (using FinBERT)
  - Weather-based factors

### Trading Strategy
- Custom long/short implementation
- Entry/exit signals based on:
  - Return trends
  - Short-term price movements
  - Long-term price averages

### Visualization
- Custom Dash application for model and strategy performance monitoring

## 📁 Project Structure

### `data_pipeline.py`
- Implements Postgres database schema construction
- Handles data extraction and loading from multiple sources:
  - AlphaVantage
  - EIA
  - EDGAR
  - Open-Meteo

### `cache_system.py`
- Implements caching mechanism for data downloads
- Optimizes performance by avoiding redundant API calls
- Manages local storage of recently downloaded data

### `models.py`
- Contains abstract implementations of time series models
- Focuses on return forecasting capabilities

### `ml_prediction.py`
- Executes walk-forward hyperparameter optimization
- Evaluates different time series models
- Determines optimal model configuration

### `trading_strategy.py`
- Implements custom long/short trading strategy
- Utilizes predictions from the best-performing time series model
- Manages position entry and exit logic

## 🚀 Features
- End-to-end ML-based quantitative trading pipeline
- Multi-factor model incorporating diverse data sources
- Sentiment analysis using FinBERT
- Custom caching system for efficient data management
- Interactive visualization dashboard

## 🔄 Future Improvements
The project serves as an introductory implementation of an ML-based quantitative trading strategy. Potential areas for enhancement include:
- Additional factor incorporation
- Strategy optimization
- Enhanced risk management
- Extended backtesting capabilities
- Real-time trading integration

## ⚙️ Setup and Installation
[To be added: Installation steps, dependencies, and configuration instructions]

## 📊 Usage
[To be added: Example usage, configuration options, and running instructions]

## 📈 Performance
[To be added: Strategy performance metrics and visualization examples]

## 📝 License
[To be added: License information]