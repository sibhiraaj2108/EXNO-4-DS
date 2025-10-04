# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
<img width="1508" height="463" alt="image" src="https://github.com/user-attachments/assets/104b9990-2a47-4c48-b444-e7c0c02057ae" />


```
data.isnull().sum()
```
<img width="272" height="492" alt="image" src="https://github.com/user-attachments/assets/1f198d50-ca98-4259-9e33-0447a547620b" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1507" height="421" alt="image" src="https://github.com/user-attachments/assets/8bc35f86-fd5c-44c4-ac1e-28447b68f679" />


```
data2=data.dropna(axis=0)
data2
```
<img width="1505" height="462" alt="image" src="https://github.com/user-attachments/assets/0947bea2-cdc2-434d-83e2-6e47cc1cf8eb" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="429" height="225" alt="image" src="https://github.com/user-attachments/assets/e9dda964-59f4-49f2-8e7e-429f7646c649" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="518" height="463" alt="image" src="https://github.com/user-attachments/assets/4ccb06d2-949e-45fc-9030-18699c4c40f9" />

```
 data2
```
<img width="1451" height="468" alt="image" src="https://github.com/user-attachments/assets/2c6bc2c9-b104-4aaa-8cbc-238fe139dce1" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1644" height="304" alt="image" src="https://github.com/user-attachments/assets/810c84da-641f-4695-8e18-a2bcf838e3b0" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1510" height="99" alt="image" src="https://github.com/user-attachments/assets/4e397945-c2de-4e9c-8a65-10458d7edc4a" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1507" height="46" alt="image" src="https://github.com/user-attachments/assets/0b7805e3-fc5e-4dd0-a043-3f758a8ed06a" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="457" height="32" alt="image" src="https://github.com/user-attachments/assets/524c65e5-5718-486f-bcfd-df5f2c3ce437" />

```
x=new_data[features].values
print(x)
```
<img width="537" height="139" alt="image" src="https://github.com/user-attachments/assets/5c44fc24-dc68-4205-a4cb-e867c0255ce4" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="569" height="81" alt="image" src="https://github.com/user-attachments/assets/bd7d445f-caa8-4c86-8bfc-07faba9c99c1" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="319" height="54" alt="image" src="https://github.com/user-attachments/assets/2492f3b0-360d-4e26-86a8-caced857a55a" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="345" height="37" alt="image" src="https://github.com/user-attachments/assets/e45020c3-be56-43df-bcd9-17e5a95208f7" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="440" height="33" alt="image" src="https://github.com/user-attachments/assets/a11bf335-6d35-4bab-aa3f-e29edf5c5917" />

```
data.shape
```
<img width="379" height="33" alt="image" src="https://github.com/user-attachments/assets/ac256381-be71-4500-8e5b-dee0f6460fb2" />

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
<img width="624" height="92" alt="image" src="https://github.com/user-attachments/assets/a8d68a05-3e67-4a8e-83d5-a72a1a6e3a96" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="616" height="251" alt="image" src="https://github.com/user-attachments/assets/175536b1-fe78-4c4f-819c-6ba81532dd3a" />

```
tips.time.unique()
```
<img width="488" height="55" alt="image" src="https://github.com/user-attachments/assets/495ca624-6f54-4b67-b680-98788c159210" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="304" height="90" alt="image" src="https://github.com/user-attachments/assets/9df0dbf5-7671-4b41-b0c5-a76b5ba17d52" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="406" height="56" alt="image" src="https://github.com/user-attachments/assets/c9287a3a-c709-4173-bd8b-7fa8c1fcac92" />

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
