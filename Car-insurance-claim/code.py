# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.iloc[:,0:5])
print(df.info())
cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
for col in cols:
    df[col] = df[col].str.replace(r'[^\d.]+', '')

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

count = y.value_counts()
print(count)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here
cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
for col in cols:
    X_train[col] = X_train[col].astype(float)

for col in cols:
    X_test[col] = X_test[col].astype(float)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'], inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

cols = ['AGE', 'CAR_AGE', 'INCOME', 'INCOME']
for col in cols:
    X_train[col] = X_train[col].fillna(X_train[col].mean())

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for col in columns:
    X_train[col] = le.fit_transform(X_train[col].astype(str))

for col in columns:
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('y_pred is ', y_pred)

score = accuracy_score(y_test, y_pred)
print('score is ', score)

precision = precision_score(y_test, y_pred)
print('precision is ', precision)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 6)
X_train, y_train = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('y_pred is ', y_pred)

score = accuracy_score(y_test, y_pred)
print('score is ', score)

# Code ends here


