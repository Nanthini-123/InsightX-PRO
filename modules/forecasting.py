# modules/forecasting.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Prophet Forecasting
def prophet_forecast(df, date_col, target_col, periods=30):
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            raise ImportError("Prophet is not installed. Run: pip install prophet")

    data = df[[date_col, target_col]].dropna().rename(columns={date_col: "ds", target_col: "y"})
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# LSTM Forecasting
def lstm_forecast(df, date_col, target_col, lookback=10, epochs=10):
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    data = df[[date_col, target_col]].dropna()
    data = data.sort_values(by=date_col)
    values = data[target_col].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_seq = scaled[-lookback:]
    forecast = []
    current_seq = last_seq.copy()

    for _ in range(lookback):
        pred = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
        forecast.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred).reshape(lookback, 1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=data[date_col].iloc[-1], periods=lookback + 1, freq="D")[1:]
    forecast_df = pd.DataFrame({date_col: future_dates, "Forecast": forecast})

    return model, forecast_df
