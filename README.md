# LSTM-Based Stock Price Predictor: Market Trend Analysis Tool

## Overview
This project aims to implement a Recurrent Neural Network (RNN) using TensorFlow, specifically an LSTM (Long Short-Term Memory) model, for stock price prediction using historical stock data. The model will use past Open, High, Low prices, and volume data to predict the next day's opening price. The project includes data preprocessing, model development, training, evaluation, and deployment aspects. The performance of the model was measured by Mean Squared Error (MSE) and achieved less than 2.5\% loss. 

## Background and Motivation
Financial markets are highly volatile and unpredictable, making stock price prediction a challenging yet valuable task. Machine learning techniques, particularly deep learning models like RNNs, have shown promising results in analyzing and forecasting time series data such as stock prices. 

The motivation behind this project is to leverage the power of RNNs to build a regression model that can assist investors and traders in making informed decisions.

## Goals
1. **Data Collection and Preprocessing:** Gather historical stock price data and preprocess it by normalizing and structuring it for training.
2. **Model Development:** Design an LSTM-based RNN model to learn from historical data and make predictions.
3. **Training and Evaluation:** Train the model using the prepared dataset and evaluate its performance using appropriate metrics such as Mean Squared Error (MSE).
4. **Deployment and Visualization:** Deploy the trained model for real-time predictions and visualize the predicted vs. actual stock prices to assess model accuracy.
5. **Documentation and Sharing:** Create comprehensive documentation including code explanations, project setup instructions, and results analysis. Share the project on Git for collaboration and feedback.

## Datasets
The project will utilize historical stock price datasets spanning multiple years, with daily Open, High, Low prices, and volume information. Dataset can be downloaded from data/ folder.

## Libraries Used
- [Pandas](https://pypi.org/project/pandas/): Data manipulation and preprocessing.
- [NumPy](https://pypi.org/project/numpy/): Numerical operations and array handling.
- [Scikit-Learn](https://scikit-learn.org/stable/install.html): Data preprocessing (MinMaxScaler) and evaluation metrics.
- [TensorFlow](https://www.tensorflow.org/install/pip): Deep learning framework for building and training RNN models.
- [Keras](https://keras.io/getting_started/): High-level API for building neural networks (used with TensorFlow).
- [Matplotlib](https://pypi.org/project/matplotlib/): Data visualization for plotting stock price predictions and actual prices.

## Practical Applications
- **Investment Decision Support:** Provide insights to investors and traders for making buy/sell decisions based on predicted price trends.
- **Risk Management:** Assist in risk assessment by forecasting potential price fluctuations and market trends.
- **Algorithmic Trading:** Integrate the predictive model into algorithmic trading systems for automated decision-making.

## Milestones

1. **Data Pre-processing:** Collect, clean, and preprocess historical stock price data, create train-test splits using **Pandas** and **Scikit-Learn** libraries.
2. **Model Development:** Implement and optimize the LSTM-based RNN model architecture using **TensorFlow** and **Keras**.
3. **Training and Evaluation:** Train the model using training data, evaluate performance on test data using MSE from **sklearn**, and fine-tune hyperparameters.
4. **Deployment:** Deploy the trained model for real-time predictions or batch predictions.
5. **Documentation and Sharing:** Create detailed documentation covering project overview, methodology, results, and instructions for others to use and contribute.

## References

- https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.DataFrame.to_numpy.html
- https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
- https://www.tensorflow.org/guide/keras/save_and_serialize
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
- https://www.tensorflow.org/guide/keras/rnn
- https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html










