#Geolocation
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime
from meteostat import Point, Daily
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pickle
import pandas as pd
import os
def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="bwi")
    try:
        location = geolocator.geocode(city_name)
        if location:
            print(location.latitude, location.longitude)
            lat=location.latitude
            long=location.longitude
            return lat,long
        else:
            return False, False
    except GeocoderTimedOut as e:
        return False, False


def get_data(latitude, longitude):
    # Set time period
    start = datetime(2008, 1, 1)
    end = datetime(2018, 12, 31)

    # Create Point for Vancouver, BC
    city = Point(latitude, longitude)

    # Get daily data for 2018
    data = Daily(city, start, end)
    data = data.fetch()
    print(type(data))
    print(data.columns)
    data_t=data.tavg
    data_n=data.prcp
    data_listX=[]
    data_listY=[]
    print(data)
    for i in range(len(data)-4):
        dataX=[]
        dataY=[]
        dataX.append(data_t[i:i+3])
        dataX.append(data_n[i:i+3])
        dataY.append(data_t[i+4])
        dataY.append(data_n[i+4])
        data_listX.append(dataX)
        data_listY.append(dataY)
    #print(data_listX)
    #print(len(data_listY))
    trainX=data_listX[:int(0.8*len(data_listX))]
    trainY=data_listY[:int(0.8*len(data_listY))]
    testX=data_listX[int(0.8*len(data_listX)):]
    testY=data_listY[int(0.8*len(data_listY)):]

    #Nach Monaten aufteilen
    print(data_listX[0])
    print(data_listX[1])
    print(data_listX[2])
    print(data_listX[3])
    return trainX,trainY,testX,testY

def train(trainX, trainY,filename):
    X=np.array(trainX)
    print(X)
    Y=np.array(trainY)
    print(Y)
    X = [np.concatenate(i) for i in X]

    
    x = np.where(np.isnan(X), 0, X)
    y = np.where(np.isnan(Y), 0, Y)
    # Define model. Specify a number for random_state to ensure same results each run
    model = DecisionTreeRegressor(random_state=1)

    # Fit model
    model.fit(x, y)
    
    pickle.dump(model, open(filename, 'wb'))
def transfer_learning(model_load_path, model_save_path,trainX, trainY):
    loaded_model = pickle.load(open(model_load_path, 'rb'))
    X=np.array(trainX)
    print(X)
    Y=np.array(trainY)
    print(Y)
    X = [np.concatenate(i) for i in X]
    loaded_model.fit(X, Y)
    pickle.dump(loaded_model, open(model_save_path, 'wb'))
def predict(testX, filename):

    p=[[testX[0][0][0],testX[0][0][1],testX[0][0][2],testX[0][1][0],testX[0][1][1],testX[0][1][2]]]

    pred=np.array(p)
    pred2 = np.where(np.isnan(pred), 0, pred)
    loaded_model = pickle.load(open(filename, 'rb'))
    print("The predictions are")
    #print(melbourne_model.predict(testX))
    predictions=loaded_model.predict(pred2)
    print((predictions))
    print(type(predictions))
    print(predictions.tolist())
    temp=predictions[0][0]
    nied=predictions[0][1]
    print('temp',predictions[0][0])
    print('niederschlag',predictions[0][1])
    return temp,nied

def read_csv_file(filename):
    data=pd.read_csv(filename)
    data_t=data.tavg
    data_n=data.prcp
    data_listX=[]
    data_listY=[]
    print(data)
    for i in range(len(data)-4):
        dataX=[]
        dataY=[]
        dataX.append(data_t[i:i+3])
        dataX.append(data_n[i:i+3])
        dataY.append(data_t[i+4])
        dataY.append(data_n[i+4])
        data_listX.append(dataX)
        data_listY.append(dataY)
    #print(data_listX)
    #print(len(data_listY))
    trainX=data_listX
    trainY=data_listY
    return trainX, trainY


def main(city_name):
    latitude,longitude=get_coordinates(city_name)
    if latitude!=False:
        trainX,trainY,testX,testY=get_data(latitude, longitude)
        if os.path.isfile(city_name+'.sav'):        
            pass
        else:
            train(trainX,trainY,city_name+'.sav')
        if os.path.isfile(city_name+'.csv'):
            read_csv_file(city_name+'.csv')
            transfer_learning(city_name+'.sav',city_name+'_transfer.sav',trainX,trainY)
            temp,nied=predict(testX,city_name+'_transfer.sav')
        else:
            temp,nied=predict(testX,city_name+'.sav')
        return temp,nied
    else:
        return False, False
