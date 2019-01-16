# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
#print(df)
print(df.iloc[:,0:5])
X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]
y = df['list_price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)


# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
#print(X_train)
cols = X_train.columns

fig, ax = plt.subplots(nrows=3, ncols=3)

# code ends here



# --------------
# Code starts here
corr = X_train.corr()
print(corr)

high_corr = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.75).any()
print(high_corr)

X_train = X_train.drop(['play_star_rating','val_star_rating'], axis=1)
X_test = X_test.drop(['play_star_rating','val_star_rating'], axis=1)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error is ', mse)

r2 = r2_score(y_test, y_pred)
print('r^2 score is ', r2)
# Code ends here


# --------------
# Code starts here
residual = y_test - y_pred
print(residual)

plt.hist(residual, normed=True, bins=30)
plt.show()
# Code ends here


