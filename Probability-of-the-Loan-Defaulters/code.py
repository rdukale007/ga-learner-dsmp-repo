# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
total = len(df)
print('Total is ', total)
p_a = len(df[df['fico'] > 700]) / total
print('Probability p(A) for the event that fico is ', p_a)
p_b = len(df[df['purpose'] == 'debt_consolidation']) / total
print('Probabilityp(B) for the event that purpose is ', p_b)
df1 = df[df['purpose'] == 'debt_consolidation'].copy()
total_ab = len(df1)
print('Length of new dataframe is ', total_ab)
p_a_b = len(df1[df1['fico'] > 700]) / total_ab
print('Probablity p(B|A) for the event purpose is ', p_a_b)
result = p_a_b == p_a
print(result)
# code ends here


# --------------
# code starts here
#print(df)
prob_lp = len(df[df['paid.back.loan'] == 'Yes']) / len(df)
print('Probability p(A) for the event that paid.back.loan is ', prob_lp)
prob_cs = len(df[df['credit.policy'] == 'Yes']) /len(df)
print('Probability p(B) for the event that credit.policy is ', prob_cs)
new_df = df[df['paid.back.loan'] == 'Yes']
prob_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes']) / len(new_df)
print('Probablity p(B|A) for the event paid.back.loan is ', prob_pd_cs)
bayes = prob_pd_cs * prob_lp / prob_cs
print('Conditional probability of the event P(A|B) is ',bayes)
# code ends here


# --------------
# code starts here
#print(df)
#value_pur = df['purpose'].value_counts()
value_pur = df.groupby(['purpose']).size()
value_pur.plot(kind='bar')
plt.show()

df1 = df[df['paid.back.loan'] == 'No'].copy()
new_value_pur = df1.groupby(['purpose']).size()
new_value_pur.plot(kind='bar')
plt.show()
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
print('Median for installment is ', inst_median)
inst_mean = df['installment'].mean()
print('Mean for installment is ', inst_mean)
plt.hist(df['installment'], normed=True, bins=30)
plt.axvline(inst_median, color='k', linestyle='dashed', linewidth=1)
plt.axvline(inst_mean, color='g', linestyle='dashed', linewidth=1)
plt.show()

incm_median = df['log.annual.inc'].median()
print('Median for log.annual.inc is ', incm_median)
incm_mean = df['log.annual.inc'].mean()
print('Mean for log.annual.inc is ', incm_mean)
plt.hist(df['log.annual.inc'], normed=True, bins=30)
plt.axvline(incm_median, color='k', linestyle='dashed', linewidth=1)
plt.axvline(incm_mean, color='g', linestyle='dashed', linewidth=1)
plt.show()
# code ends here


