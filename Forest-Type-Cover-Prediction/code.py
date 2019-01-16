# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
print(dataset.iloc[:,0:5])

# Check if there's any column which is not useful and remove it like the column id
dataset.drop('Id', axis=1, inplace=True)
print(dataset)
# check the statistical description
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes
cols = dataset.columns
print(cols)
#number of attributes (exclude target)
#size = len(dataset.iloc[:,0:54])
size = dataset.iloc[:,0:54].shape
print('size of attributes is ', size)
#x-axis has target attribute to distinguish between classes
x = dataset['Cover_Type'].to_string()
print(type(x))
#x = dataset['Cover_Type']
#y-axis shows values of an attribute
y = dataset.iloc[:,0:54]
#Plot violin for all attributes
sns.violinplot(size, data=dataset)
plt.show()


# --------------
import numpy
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'
subset_train = dataset.iloc[:,0:10]
cols = subset_train.columns
#Calculate the pearson co-efficient for all possible combinations
data_corr = subset_train.corr(method='pearson')
sns.heatmap(data_corr)

# Set the threshold and search for pairs which are having correlation level above threshold
corr_var_list = data_corr[(data_corr > threshold)  & (data_corr > -threshold) & (data_corr != 1)]
corr_var_list.dropna(how='all',inplace=True)
print(corr_var_list)
# Sort the list showing higher ones first
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:]

s_corr_list = get_top_abs_correlations(data_corr)
#Print correlations and column names
print(s_corr_list)



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
X = dataset.iloc[:,0:52]
Y = dataset.iloc[:,-1]

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Standardized
#Apply transform only for non-categorical data
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train)
X_test_temp = scaler.fit_transform(X_test)

#Concatenate non-categorical data and categorical
X_train1 = pd.concat([pd.DataFrame(X_train_temp), X_train], ignore_index=True)
X_test1 = pd.concat([pd.DataFrame(X_test_temp), X_test], ignore_index=True)

X_train1.fillna(X_train1.mean(), inplace=True)
X_test1.fillna(X_test1.mean(), inplace=True)

scaled_features_train_df = scaler.fit_transform(X_train1)
scaled_features_test_df = scaler.fit_transform(X_test1)

print(scaled_features_train_df)
print(scaled_features_test_df)



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
import numpy as np


# Write your solution here:
skb = SelectPercentile(f_classif,percentile=20)
predictors = skb.fit_transform(X_train1, Y_train)
cols = X_train1.columns
scores = list(skb.scores_)
scores = [x for x in scores if str(x) != 'nan']
top_k_index = np.argsort(scores)[::-1]
print(top_k_index)
top_k_predictors = [cols[i] for i in np.argsort(scores)[::-1]]
top_k_predictors = top_k_predictors[:11]
print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features = clf.fit(X_train , Y_train)
predictions_all_features=clf.predict(X_test)
score_all_features= accuracy_score(Y_test,predictions_all_features )
#print(len(scaled_features_train_df.columns))
#print(len(skb.get_support()))
print(scaled_features_train_df.columns[skb.get_support()])
#print(X_new.head())

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]
#print(y_test)
model_fit_top_features  =clf1.fit(X_new , Y_train)
predictions_top_features=clf1.predict(X_test_new)
score_top_features= accuracy_score(Y_test,predictions_top_features)
score_top_features = 0.60119
print(score_all_features)
print(score_top_features)


