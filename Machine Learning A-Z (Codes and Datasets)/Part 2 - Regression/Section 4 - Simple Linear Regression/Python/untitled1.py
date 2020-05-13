# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)



# plot the linear regression


plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,y_prid,color = 'blue')
plt.show()


plt.scatter(X_test,y_test,color = 'blue')
plt.plot(X_train,regressor.predict(y_train),color = 'red')
plt.show()




