# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

# Importing datasets
df=pd.read_csv("Wine quality-data.csv")
print(df.to_string())

# Data preprocessing...
df.info()
df.describe()
df.shape
df.isnull().sum()
df.isnull().sum().sum()
df.drop(columns="Id",inplace=True)

# Extracting Independent and Dependent variable:
x=df.iloc[:,0:11]
y=df.iloc[:,-1]

# splitting the dataset into training and test set:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25,
random_state=0)

# Feature scaling:
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

# Fitting Random forest classifier to the training set:
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10,
criterion="entropy")
classifier.fit(x_train, y_train)

# predicting the test set result:
y_pred= classifier.predict(x_test)

# To compare the actual output values for x _test with the predicted value:
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())

# Evaluating the algorithm:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,
y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# predicting the accuracy score:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))