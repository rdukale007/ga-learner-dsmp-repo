# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)
data_sample = data.sample(n=sample_size, random_state=0)
sample_mean = data_sample['installment'].mean()
print('Mean of installment is ', sample_mean)
sample_std = data_sample['installment'].std()
print('Standard deviation of installment is ', sample_std)
margin_of_error = z_critical * sample_std / sample_size ** 0.5
print('Margin of error is ', margin_of_error)
confidence_interval = [sample_mean -(margin_of_error), sample_mean + (margin_of_error)]
print('confidence_interval is ', confidence_interval)
true_mean = data['installment'].mean()
print('True_mean is ', true_mean)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
#print(data)
fig ,(axes) = plt.subplots(nrows = 3 , ncols = 1)

for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        sample_data = data.sample(n=sample_size[i], random_state=0)
        sample_mean = sample_data['installment'].mean()
        m.append(sample_mean)
    mean_series = pd.Series(m)

plt.hist(mean_series)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].map(lambda x: x.rstrip('%'))
data['int.rate'] = data['int.rate'].astype('float64')
data['int.rate'] = data['int.rate'] / 100

z_statistic, p_value = ztest(data[data['purpose'] =='small_business']['int.rate'], value = data['int.rate'].mean(),alternative='larger')
print('z_statistic is ', z_statistic)
print('p_value is ', p_value)

if p_value > 0.05:
    inference = 'Accept'
else:
    inference = 'Reject'

print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(data[data['paid.back.loan']=='No']['installment'], x2=data[data['paid.back.loan']=='Yes']['installment'], alternative='two-sided')
print('z_statistic is ', z_statistic)
print('p_value is ', p_value)

if p_value > 0.05:
    inference = 'Accept'
else:
    inference = 'Reject'

print(inference)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']== 'Yes']['purpose'].value_counts()
#print(yes)
no = data[data['paid.back.loan']== 'No']['purpose'].value_counts()
#print(type(no))
observed = pd.concat([yes,no], axis = 1, keys=['Yes','No'])
print(observed)
chi2, p, dof, ex = chi2_contingency(observed)
print('chi2 is ', chi2)
print('critical_value is ', critical_value)

if chi2 > critical_value:
    null_hypo = 'Reject'
else:
    null_hypo = 'Cannot be Rejected'


