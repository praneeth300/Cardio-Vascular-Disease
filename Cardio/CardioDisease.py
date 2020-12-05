
import pandas as pd
import numpy as np

data=pd.read_csv('C:/Users/Praneeth/Downloads/Ineuron Hakethon/CardioVascularDisease/CardioVascularDisease/cardio_train.csv',sep=';')

data=data.drop('id',axis=1)

data['Age']=data['age']/365
data=data.drop(columns=['age'],axis=1)
data['height']=data['height']/100
data['BMI']=data['weight']/np.power(data['height'],2)
#Drop the height and weight columns from the dataset
data=data.drop(columns=['height','weight'],axis=1)
data['ap_hi']=data['ap_hi'].apply(lambda x: np.NaN if x <= 0 or x == 1 else x)
data['ap_lo']=data['ap_lo'].apply(lambda x: np.NaN if x <= 0 or x == 1 else x)

#Fill the null values in column 'ap_hi' & ap_lo with median
data['ap_hi']=data['ap_hi'].fillna(data['ap_hi'].median())
data['ap_lo']=data['ap_lo'].fillna(data['ap_lo'].median())

#Split the data in to x and y
x=data.drop('cardio',axis=1)
#Rearranging the columns
x=x[['Age','gender','BMI','ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
       'active']]
y=data['cardio']

x['ap_hi']=np.log(x['ap_hi'])
x['ap_lo']=np.log(x['ap_lo'])
#Splitting the data in to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

import xgboost
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
xgb_pred=xgb.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Accuracy Score: ',accuracy_score(y_test,xgb_pred))
print('Classification_report--->','\n',classification_report(y_test,xgb_pred))
print(confusion_matrix(y_test,xgb_pred))

import pickle
filename = 'xgbmodel_1.pkl'
pickle.dump(xgb, open(filename, 'wb'))