# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'],axis=1)
print(banks.isnull().sum())
bank_mode = banks.mode().iloc[0]
print(bank_mode)
banks = banks.fillna(bank_mode)
print(banks)
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc='mean')

print(avg_loan_amount)

# code ends here



# --------------
# code starts here
loan_approved_se = banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')]
print('Count of Self Employed loan : ', len(loan_approved_se))

loan_approved_nse = banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')]
print('Count of No Self Employed loan : ', len(loan_approved_nse))

Loan_Status = 614
print('Count of total Loan status', Loan_Status)

percentage_se = (len(loan_approved_se) * 100) / Loan_Status
print('Percentage of loan approval for self employed people are ', percentage_se)

percentage_nse = (len(loan_approved_nse) * 100) / Loan_Status
print('Percentage of loan approval for people who are not self-employed are ' , percentage_nse)
# code ends here


# --------------
# code starts here


# loan amount term 

loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)

# code ends here


# --------------
# code ends here
#loan_groupby = banks.groupby(['ApplicantIncome', 'Credit_History'])['Loan_Status']
loan_groupby = banks.groupby('Loan_Status')[['ApplicantIncome', 'Credit_History']]
mean_values = loan_groupby.mean()
print(mean_values)

# code ends here


