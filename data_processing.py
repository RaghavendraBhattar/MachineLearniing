import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NAN, strategy='most_frequent')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# Encoding catogorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()
labelencoder_X = LabelEncoder()
Y = labelencoder_X.fit_transform(Y)


# Splitting the datatset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train, X_test)
# print(Y_train, Y_test)
