from statsmodels.api import datasets
iris = datasets.get_rdataset("iris")
iris.data.columns = ["sapel_len","sepal_wid","petel_len","petel_wid","species"]
iris.data.head()

iris.data.dtypes


from sklearn.preprocessing import scale
import pandas as pd
num_cols = ["sapel_len","sepal_wid","petel_len","petel_wid","species"]

iris_scaled = scale(iris.data[num_cols])