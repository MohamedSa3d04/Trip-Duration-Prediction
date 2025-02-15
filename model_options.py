from sklearn.linear_model import Ridge
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error as MAPE , mean_absolute_error

# Standrization function to decrease the variance
def standrize (Xtrain, Xval=None):
    sc = MinMaxScaler()
    x_train_std = sc.fit_transform(Xtrain)
    if type(Xval) != None:
        x_val_std = sc.transform(Xval)
        return x_train_std, x_val_std, sc
    else:
        return x_train_std

# A function to evaluate the model
def model_evaluation(mod, test_x, test_y, type='Testing', approach1 = True):
    pred = mod.predict(test_x)
    mape = MAPE(test_y, pred) * 100
    r2 = r2_score(test_y, pred)
    rmse = np.sqrt(mean_absolute_error(test_y, pred))
    if approach1:
        print(f'{type + 'For '+ 'approach1 '}\n- R2 : {r2}\nMAPE : {mape}\nRMSE : {rmse}\n')
    else:
        print(f'{type + 'For '+ 'approach2 '}\n- R2 : {r2}\nMAPE : {mape}\nRMSE : {rmse}\n')

    return r2, mape

# A function to rain ridge model
def train_ridge(X, y, alpha=1):
    mod = Ridge(alpha)
    mod.fit(X, y)
    return mod
    
