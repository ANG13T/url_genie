import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


df = pandas.read_csv('IRIS.csv')
model = KNeighborsClassifier(n_neighbors=3)

features = list(zip(df["sepal_length"], df["sepal_width"]))

model.fit(features,df["species"])

pickle.dump(model, open('model.pkl','wb'))