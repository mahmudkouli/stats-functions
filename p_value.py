
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def t_stat(model, X, y, feature_name):
    
    # computing sigma squared
    y_pred = model.predict(X)
    data['y_pred'] = y_pred
    sigma_sq = np.sum((y - data['y_pred'])**2) / (len(data[feature_name]) - 2)
    
    # computign SST_x
    data['diff'] = data[feature_name] - np.mean(data[feature_name])
    data['diff'] = data['diff'] ** 2
    sst_x = np.sum(data['diff'])
    
    # computing the denominator
    R_sq = round(model.score(X, y),2)
    denom = sst_x * (1-R_sq)
    
    # computing t-stat
    st_dev = np.sqrt(sigma_sq / denom)
    coef = model.coef_[0]
    t_stat = coef/st_dev
    
    return {'Variance of residuals': sigma_sq, 'Variance in the '+feature_name: sst_x, 'Standard Error approx.':st_dev, 'Coefficient of '+feature_name: coef, 't-stat': t_stat}
