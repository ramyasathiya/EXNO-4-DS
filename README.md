# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
#### STEP 1:Read the given Data.
#### STEP 2:Clean the Data Set using Data Cleaning Process.
#### STEP 3:Apply Feature Scaling for the feature in the data set.
#### STEP 4:Apply Feature Selection for the feature in the data set.
#### STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
```
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME : Ramya S
REG NO : 212222040130
```
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/703e88ee-f8d1-4769-b107-cd331eff38ca)
```
data.isnull().sum()
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/5ffc9c47-d0af-4b1c-a6ce-6042049c3549)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/e4265e5e-1d00-402e-b9b5-54969f7d4bd6)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/e6c40d14-6070-4d57-825d-497937077dd6)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/29fdf9fb-2aa4-4501-9e5b-a09e87a7992f)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/d694ef71-82e8-437a-9380-291d9f655b4c)
```
data2
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/77cdc673-365b-4d88-9ba3-aef0772ad4d1)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/40fee06a-0d12-474c-92a8-a5d8d72ec8f8)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/e3e6dc73-e973-4a93-82b4-0957d392bde3)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/c71d0b2e-467a-45e0-989e-c03d5f448cb0)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/69a45f59-0c62-4a1d-bab4-01fdbfbd5d27)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/45fcc088-5c3c-4952-b02d-1bc816fa242f)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/5d8fe3fb-1384-4b1b-bc7e-d68a36d88f6d)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/79249edd-bdbd-4778-913b-c2e9afaee4ee)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/5a9670ed-8dd6-47c2-b518-e2d6a3026204)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/430a0c4b-cdce-4384-b92b-563d78757040)
```
data.shape
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/faf67c38-239f-43fc-aa33-c69a9907df3c)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/0ef4a0da-64b6-4c01-ab36-bc52c5b50c02)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/cfc24046-f1b0-461c-a2b7-e3bbe92c3107)
```
tips.time.unique()
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/65aad783-c72b-461e-9136-7db0d6609721)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/9fceef68-fffa-4fd3-a8e1-f627e06c0d39)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Kalpanareshma/EXNO-4-DS/assets/122040453/1257bbd5-f921-47a9-b233-89744d3dea1e)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
