# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data.hist(column='Rating', bins=8)

data = data[data['Rating'] <= 5]
data.hist(column='Rating', bins=8)
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
#print(total_null)
percent_null = (total_null/data.isnull().count())
#print(percent_null)
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total','Percent'])
print(missing_data)
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------
import seaborn as sns
#Code starts here
sns.catplot(x="Category", y="Rating", data=data, kind="box", height = 10)
plt.xticks(rotation=90)
plt.title("Rating vs Category [BoxPlot]")
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
#print(data['Installs'])
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].apply(int)
print(data['Installs'])

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x="Installs", y="Rating", data=data)
plt.title("Rating vs Installs [RegPlot]")
plt.show()
#Code ends here



# --------------
#Code starts here
print(data['Price'])
data['Price'] = data['Price'].str.replace('$','')
data['Price'] = data['Price'].apply(float)

sns.regplot(x="Price", y="Rating", data=data)
plt.title("Rating vs Price [RegPlot]")
plt.show()
#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())
data['Genres'] = data['Genres'].str.split(';',n=1, expand=True)
gr_mean = data.groupby(['Genres'],as_index=False).mean()
print('gr_mean is ', gr_mean)
print(gr_mean.describe())
gr_mean = gr_mean.sort_values('Rating')
print(gr_mean)
#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = max(data['Last Updated'])
print('max_date is ', max_date)
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title("Rating vs Last Updated [RegPlot]")
plt.show()

#Code ends here


