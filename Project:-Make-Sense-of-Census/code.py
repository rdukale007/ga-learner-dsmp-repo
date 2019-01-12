# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
new_record = np.array(list(new_record))
#Code starts here
data = np.genfromtxt(path,delimiter=",", skip_header = 1)
census = np.concatenate((data, new_record))
print(census)


# --------------
#Code starts here
age = census[:,0]
print(age)
max_age = np.max(age)
print(max_age)
min_age = np.min(age)
print(min_age)
age_mean = np.mean(age)
print(age_mean)
age_std = np.std(age)
print(age_std)


# --------------
#Code starts here
#np.set_printoptions(threshold=np.nan)
#race_0 = np.array([x for x in census[0:, 2] if x == 0],dtype='int32')
#race_1 = np.array([x for x in census[0:, 2] if x == 1],dtype='int32')
#race_2 = np.array([x for x in census[0:, 2] if x == 2],dtype='int32')
#race_3 = np.array([x for x in census[0:, 2] if x == 3],dtype='int32')
#race_4 = np.array([x for x in census[0:, 2] if x == 4],dtype='int32')

race_0 = census[census[:, 2] == 0]
race_1 = census[census[:, 2] == 1]
race_2 = census[census[:, 2] == 2]
race_3 = census[census[:, 2] == 3]
race_4 = census[census[:, 2] == 4]

len_0 = len(race_0)
#print('len_0 :' + str(len_0))
len_1 = len(race_1)
#print('len_1 :' + str(len_1))
len_2 = len(race_2)
#print('len_2 :' + str(len_2))
len_3 = len(race_3)
#print('len_3 :' + str(len_3))
len_4 = len(race_4)
#print('len_4 :' + str(len_4))
if min(len_0,len_1,len_2,len_3,len_4) == len_0:
    minority_race = 0
elif min(len_0,len_1,len_2,len_3,len_4) == len_1:
    minority_race = 1
elif min(len_0,len_1,len_2,len_3,len_4) == len_2:
    minority_race = 2
elif min(len_0,len_1,len_2,len_3,len_4) == len_3:
    minority_race = 3
else:
    minority_race = 4
print(minority_race)



# --------------
#Code starts here
senior_citizens = np.array(census[census[:, 0] > 60],dtype='int32')
#senior_citizens = census[census[:, 0] > 60]

working_hours_sum = senior_citizens[:, 6].sum()

senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum / senior_citizens_len

print('Working Hours Sum : ' + str(working_hours_sum))
print('Senior Citizen Sum : ' +  str(senior_citizens_len))
print('Avg Working Hours : ' + str(avg_working_hours))


# --------------
#Code starts here
high = census[census[:, 1] > 10]
low = census[census[:, 1] <= 10]

avg_pay_high = high[:, 7].mean()
avg_pay_low = low[:, 7].mean()

print(avg_pay_high)
print(avg_pay_low)

if avg_pay_high == avg_pay_low:
    print()


