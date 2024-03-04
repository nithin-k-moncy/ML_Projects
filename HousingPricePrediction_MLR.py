# Data import,structuring and cleaning
#import libraries
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
# import dataset
df=pd.read_csv("house price prediction-data.csv")
print(df.to_string())
# Data preprocessing...

df.info()
df.describe()
df.shape
df.drop(columns=["date","street","city","country","statezip","yr_renovated","yr_built"],inplace=True)
df.isnull().sum().sum()

# Split dataset in feature and target variable:
x=df.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
y=df.iloc[:,0]

# split x and y into training and testing sets:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2,random_state=0)

# Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set result:
y_pred= regressor.predict(x_test)

# To compare the actual output values for x _test with the predicted value:
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())

# Evaluating the Algorithm:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,
y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Predicting the accuracy score:
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 score is ",score*100,"%")



