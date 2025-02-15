# Usage-Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import model_options
import numpy as np
from geopy.distance import geodesic
import joblib

# Pre-Paring the data for the each mode
def prepare_data(data, approach1 = True):
    #Approach1 : Using Co-Ordinates Points
    if approach1:
        x = data[['passenger_count','pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude']]
        y = data['trip_duration'].values
        X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=0.98, shuffle=True)

    #Approach2 : Using Theorm
    else:
        x = data[['passenger_count','distances']].values
        y = data['trip_duration'].values

        X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=0.98, shuffle=True)
    return X_train, X_val, y_train, y_val



# Training Ridge Model in the data (With Validating it)
def train_model(X_train, X_val, y_train, y_val, alpha, approach1=True):
    X_train_std, X_val_std, std = model_options.standrize(X_train, X_val)
    model = model_options.train_ridge(X_train_std, y_train, alpha)

    model_options.model_evaluation(model, X_train_std, y_train, '\nTraining', approach1)
    r2, mape = model_options.model_evaluation(model, X_val_std, y_val, '\nValidating', approach1)
    return model, std



def prepare_for_test(data, approach1=True):
        if approach1: #Just Get The Numerical Points
            x = data[['passenger_count','pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude']]
            y = data['trip_duration'].values
            # data[['passenger_count','pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']].to_csv('./TestingData/approach1Test.csv')



        else: #Get the new feature (distances) Using 4-Co-Ordinates Columns
            data.drop(columns=['id', 'vendor_id','store_and_fwd_flag','pickup_datetime'], inplace=True)
            distances = []
            for i in range(data.shape[0]):
                pickup = (data.iloc[i,2], data.iloc[i,1])
                dropoff = (data.iloc[i,4], data.iloc[i,3])
                distance_meters = geodesic(pickup, dropoff).meters
                distances.append(distance_meters)
            data['distances'] = distances
            data['distances'] = data['distances'].astype('int')
            data.drop(columns=['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude'], inplace=True)
        
            # data.to_csv('./cat.csv')
            x = data[['passenger_count','distances']].values
            y = data['trip_duration'].values
            # data[['passenger_count','distances','trip_duration']].to_csv('./TestingData/approach2Test.csv')


        return x, y

# Testing the model (each_approach) with unseen data
def testModel(mod, std, test_x, target, approach1=True):
    x_std = std.transform(test_x)
    y_log = np.log1p(target)

    model_options.model_evaluation(mod, x_std, y_log, '\nTesting', approach1)





if __name__ == '__main__':
    df_1 = pd.read_csv('./data/train_approach1.csv') #data_model_1 (Normal Data)
    df_2 = pd.read_csv('./data/train_approach2.csv')#data_model_2 (Feature Distanced Added)
    df_tst = pd.read_csv('./data/test.csv') #Test data (model never saw it)

    print(df_1.shape)
    #Grapping data and training model for approach 1
    X_train, X_val, y_train, y_val = prepare_data(df_1)
    model_1, std_1 = train_model(X_train, X_val, y_train, y_val, 1)

    # joblib.dump(model_1, './models/model_app1.pkl')
    # joblib.dump(std_1, './models/std_app1.pkl')

    #Grapping data and training model for approach 2
    X_train, X_val, y_train, y_val = prepare_data(df_2, approach1=False)
    model_2, std_2 = train_model(X_train, X_val, y_train, y_val, 1, False)
    # joblib.dump(model_2, './models/model_app2-V1.pkl')
    # joblib.dump(std_2, './models/std_app2.pkl')


    #Prapering data For each test approach
    x_test_1, y_test_1 = prepare_for_test(df_tst, True)
    x_test_2, y_test_2 = prepare_for_test(df_tst, False)

    #Testing the two models
    testModel(model_1, std_1, x_test_1, y_test_1, True)
    testModel(model_2, std_2, x_test_2, y_test_2, False)



    
