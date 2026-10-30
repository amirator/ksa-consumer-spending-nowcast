import numpy as np


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100)
