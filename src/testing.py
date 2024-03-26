import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error

def testing(sc):
    regressor_Savedmodel=load_model("../models/")
    test= pd.read_csv("../data/test_data_RNN.csv")
    scaled_testDataset = sc.fit_transform(test)

    X_test_data =scaled_testDataset[:,:12]
    Y_test_data =scaled_testDataset[:,12]

    X_test_data = X_test_data.reshape(-1,3,4)
    Y_test_data = Y_test_data.reshape(-1,1)

    predicted_stockPrice = regressor_Savedmodel.predict(X_test_data)
    loss= mean_squared_error(Y_test_data,predicted_stockPrice)
    print("The loss on test data is ",loss)

    return loss