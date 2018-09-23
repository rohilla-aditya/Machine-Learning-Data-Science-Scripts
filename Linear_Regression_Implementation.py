import numpy as np 
import pandas as pd 

#Importing data using pandas
train_data = pd.read_csv("training_data.csv")
train_labels = pd.read_csv("training_labels.csv")
test_data = pd.read_csv("testing_data.csv")
test_labels = pd.read_csv("testing_labels.csv")

# Adding a row of 1 to our data for adding intercept
num1= np.ones(shape = train_labels.shape)
train_data = np.concatenate((num1, train_data), 1)


'''
Calculating W vector from : 
    W = (X(Transpose) . X)−1 . X(Transpose).Y
''' 
W_coef = np.linalg.inv(train_data.transpose().dot(train_data)).dot(train_data.transpose()).dot(train_labels)

#Adding a row of 1 for intercept
test_data = np.concatenate((num1, test_data), 1)


#Predicting labels for our test_data
test_data_predicted = test_data.dot(W_coef)

#print(test_data_predicted)

# Comparing predicted data with original labels and calculating the Root mean square error 
Rmse = np.sqrt(np.mean((test_data - test_data_predicted)**2))
print(Rmse)