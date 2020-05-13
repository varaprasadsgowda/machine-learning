import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1:].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressen = LinearRegression()
regressen.fit(X_train,Y_train)


Y_prid = regressen.predict(X_test)



plt.scatter(X_train,Y_train,color = 'red',edgecolors='red')
plt.plot(X_train,regressen.predict(X_train),color = 'blue')
plt.title('salary vs experiance(training set)')
plt.xlabel = 'expereience'
plt.ylabel = 'salary'
plt.show()




