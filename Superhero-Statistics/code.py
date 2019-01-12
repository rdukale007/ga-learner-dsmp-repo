# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data = pd.read_csv(path)
#print(data.head())
data['Gender'].replace('-', 'Agender', inplace=True)
gender_count = data['Gender'].value_counts()
print(gender_count)
gender_count.plot(kind='bar')



# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
plt.pie(alignment)
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
sc_df = data[['Strength','Combat']].copy()
#print(sc_df)
sc_covariance = (sc_df.cov())['Strength']['Combat']
print('Covariance between two variables is ', sc_covariance)

sc_strength = sc_df['Strength'].std()
print('Standard deviation of column Strength is ', sc_strength)

sc_combat = sc_df['Combat'].std()
print('Standard deviation of column Combat is ', sc_combat)

sc_pearson = sc_covariance / (sc_strength * sc_combat)
print('Correlation coefficient between Strength & Combat is ', sc_pearson)

ic_df = data[['Intelligence','Combat']].copy()

ic_covariance = (ic_df.cov())['Intelligence']['Combat']
print('Covariance between two variables is ', ic_covariance)

ic_intelligence = ic_df['Intelligence'].std()
print('Standard deviation of column Intelligence is ', ic_intelligence)

ic_combat = ic_df['Combat'].std()
print('Standard deviation of column Combat is ', ic_combat)

ic_pearson = ic_covariance / (ic_intelligence * ic_combat)
print('Correlation coefficient between Strength & Combat is ', ic_pearson)


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
#print(total_high)

super_best = data[(data['Total'] > total_high)]
#print(super_best)

super_best_names = [data['Name']]
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3)
data['Intelligence'].plot(kind='box', stacked=True, ax=ax_1)
ax_1.set_title('Intelligence')

data['Speed'].plot(kind='box', stacked=True, ax=ax_2)
ax_1.set_title('Speed')

data['Power'].plot(kind='box', stacked=True, ax=ax_3)
ax_1.set_title('Power')




