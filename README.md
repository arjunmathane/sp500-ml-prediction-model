# sp500-ml-prediction-model
ML-driven S&amp;P 500 market prediction model with technical indicators and FinBERT sentiment analysis
# ML-Driven S&P 500 Market Prediction Model

A quantitative trading model that combines machine learning with 
NLP-based sentiment analysis to predict S&P 500 market direction.

## What This Project Does
- Pulls 10 years of S&P 500 historical data (2014–2024)
- Engineers 6 technical indicators: RSI, MACD, moving averages, volatility
- Trains a Random Forest classifier on 1,973 trading days
- Achieves 53.85% directional accuracy on unseen test data
- Integrates ProsusAI/FinBERT transformer model for live news sentiment
- Backtests strategy performance against buy-and-hold benchmark

## Key Results
| Metric | Value |
|--------|-------|
| Model Accuracy | 53.85% |
| Training Days | 1,973 |
| Test Days | 494 |
| Top Feature | RSI (18.6%) |
| Buy & Hold Return (test) | +1.28% |
| ML Strategy Return (test) | -9.09% |

## Key Finding
The model underperformed during the 2022–2024 bull market due to 
excessive caution on up days — a known limitation of models trained 
on mixed market regimes. Future work includes regime detection to 
activate the model selectively during volatile market conditions.

## Tech Stack
- Python, pandas, numpy
- yfinance (data)
- scikit-learn (Random Forest)
- HuggingFace Transformers + PyTorch (FinBERT)
- matplotlib (visualization)

## Project Structure
The entire project is contained in one end-to-end Jupyter notebook:
1. Data Collection & Cleaning
2. Feature Engineering (technical indicators)
3. ML Model Training & Evaluation
4. Backtesting vs Benchmark
5. NLP Sentiment Analysis Pipeline
6. Project Summary

## Limitations & Future Work
1. Add regime detection to activate model only in volatile markets
2. Collect historical daily sentiment for full feature integration
3. Expand features: VIX, earnings data, sector rotation signals
4. Test alternative models: LSTM, XGBoost, ensemble methods
