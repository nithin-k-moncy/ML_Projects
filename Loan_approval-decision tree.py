# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

# importing dataset
df=pd.read_csv("loan_approval_data.csv")
print(df.to_string())

# data preprocessing...

df.shape
df.info()
df.describe()
df.isnull().sum()
df.isnull().sum().sum()
df.dtypes
# Remove leading spaces from column names...
df.rename(columns=lambda x: x.strip(),inplace=True)
# Dropping unwanted columns...
df.drop(columns=["loan_id","education","self_employed"],inplace=True)
# applying label encoding to the column loan_status
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df["loan_status"]=label_encoder.fit_transform(df["loan_status"])

# extracting Independent and Dependent variable:
x=df.iloc[:,0:9]
y=df.iloc[:,-1]

# splitting the dataset into training and test set:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25,
random_state=0)

# feature scaling:
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

# fitting Decision Tree classifier to the training set:
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# predicting the test set result:
y_pred= classifier.predict(x_test)

df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print("Prediction Result")
print(df2.to_string())

# To compare the actual output values for x _test with the predicted value:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,
y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# checking the accuracy:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))