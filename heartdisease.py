import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the dataset
heart_data=pd.read_csv('heart_disease.csv')
#print(heart_data)
print(heart_data['target'].value_counts())
# 1-defective
# 0-normal
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score on training data',training_data_accuracy)

#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data',test_data_accuracy)

#build a prediction system
input_data=(60,1,0,130,206,0,0,132,1,2.4,1,2,3)
inputdata=np.asarray(input_data)
inputdata_reshaped=inputdata.reshape(1,-1)
prediction= model.predict(inputdata_reshaped)
print(prediction)

if prediction[0]==0:
    print("The person does not have a heart disease")
else:
    print("The person has a heart disease")