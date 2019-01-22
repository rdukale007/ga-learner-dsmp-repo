# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
#print(y.head(5))
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
#print(X_train['TotalCharges'].isna().sum())
X_train['TotalCharges'] = X_train['TotalCharges'].replace(' ', np.NaN)
X_test['TotalCharges'] = X_test['TotalCharges'].replace(' ', np.NaN)

X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'])
X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'])

X_train['TotalCharges'].fillna((X_train['TotalCharges'].mean()), inplace=True)
X_test['TotalCharges'].fillna((X_test['TotalCharges'].mean()), inplace=True)

print(X_train['TotalCharges'].isnull().sum())
print(X_test['TotalCharges'].isnull().sum())

le = LabelEncoder()

for i in range(0,X_train.shape[1]):
    if X_train.dtypes[i] == 'object':
        X_train[X_train.columns[i]] = le.fit_transform(X_train[X_train.columns[i]])

for i in range(0,X_test.shape[1]):
    if X_test.dtypes[i] == 'object':
        X_test[X_test.columns[i]] = le.fit_transform(X_test[X_test.columns[i]])


X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

y_train.replace({'No':0, 'Yes':1},inplace=True)
y_test.replace({'No':0, 'Yes':1},inplace=True)


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train)
print(X_test)
print(y_train)
print(y_test)

ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
print('y_pred is ', y_pred)

ada_score = accuracy_score(y_test, y_pred)
print('ada_score is ', ada_score)

ada_cm = confusion_matrix(y_test, y_pred)
print('ada_cm is ', ada_cm)

ada_cr = classification_report(y_test, y_pred)
print('ada_cr is ', ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print('y_pred is ', y_pred)

xgb_score = accuracy_score(y_test, y_pred)
print('xgb_score is ', xgb_score)

xgb_cm = confusion_matrix(y_test, y_pred)
print('xgb_cm is ', xgb_cm)

xgb_cr = classification_report(y_test, y_pred)
print(xgb_cr)

xgb_clf = XGBClassifier(random_state=0)
clf_model = GridSearchCV(estimator=xgb_clf, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)
print('y_pred is ', y_pred)

clf_score = accuracy_score(y_test, y_pred)
print('clf_score is ', clf_score)

clf_cm = confusion_matrix(y_test, y_pred)
print('clf_cm is ', clf_cm)

clf_cr = classification_report(y_test, y_pred)
print(clf_cr)


