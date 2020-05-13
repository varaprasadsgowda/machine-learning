import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1:].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)




from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train) ,color = 'blue')
plt.title('age vs sallary')
plt.xlabel('age')
plt.ylabel('sallary')
plt.show()

plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.show()


