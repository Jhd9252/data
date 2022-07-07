# -*- coding: utf-8 -*-
"""
Exploration of Salary by Degree/Major

Kaggle Dataset: https://www.kaggle.com/wsj/college-salaries


3/5/2022: Read data, clean data, rename columns
3/7/2022: plotting salary of percentiles (10,25, 50, 75, 90) of all degrees
3/8/2022: plotting salary  of top 10 degrees by mid career median salary as line chart showing range 
"""

# import 
import pandas as pd
import matplotlib.pyplot as plt

# read in csv
degree_df = pd.read_csv('degrees-that-pay-back.csv')

# clean degree data
# clean strings and conver string integers into floats / leave nans for pandas
# rename columns
degree_df.rename(columns = {'Undergraduate Major':'u_major',
                            'Starting Median Salary':'s_median', 
                            'Mid-Career Median Salary':'m_median',
                            'Percent change from Starting to Mid-Career Salary':'pChange_SM',
                            'Mid-Career 10th Percentile Salary':'mid_10p',
                            'Mid-Career 25th Percentile Salary':'mid_25p',
                            'Mid-Career 75th Percentile Salary':'mid_75p',
                            'Mid-Career 90th Percentile Salary':'mid_90p'},
                             inplace = True)

degree_col = [x for x in degree_df.columns if x not in ['u_major','pChange_SM']]
for col in degree_col:
    degree_df[col] = degree_df[col].replace(r'[^0-9.]', '', regex=True).astype(float)

degree_describe = degree_df.describe()

# plotting median salary of (10, 25, median mid, 75, 90) percentiles of all degrees
# sorted by mid-career median salary
# acsending = True
# using default fig parameters and colors
degree_df.sort_values(by = 'm_median', ascending = True, inplace = True)

fig = plt.figure(figsize = (10,15))
plt.grid(True)
plt.xlabel('USD $')
plt.ylabel('Undergraduate Majors')
plt.title('Salary Information by Degree')

x = degree_df['m_median']
y = degree_df['u_major']
plt.scatter(x,y)

x1 = degree_df['mid_90p']
plt.scatter(x1,y)

x2 = degree_df['mid_75p']
plt.scatter(x2,y)

x3 = degree_df['mid_25p']
plt.scatter(x3,y)

x4 = degree_df['mid_10p']
plt.scatter(x4,y)

x5 = degree_df['s_median']
plt.scatter(x5,y)

plt.show()

# plotting top 10 degrees as line chart by m_median
# lines are showing range of salary
# dots are median salary
# x = USD
# y = top 10 degrees
degree_df.sort_values(by = 'm_median', ascending = False, inplace = True)
fig = plt.figure(figsize = (15,10))
plt.grid(True)
plt.xlabel('USD $')
plt.ylabel('Degree')
plt.title('Top Undergrad Degrees by Mid-Career Median Salary')

top10 = degree_df.iloc[:11,[0,1,2,3,4,5,6,7]].sort_values(by='m_median', ascending = True)

for i in range(11):
    x = top10.iloc[i,[1,4,5,2,6,7]]
    y = [top10.iloc[i,0]] * len(x)
    plt.plot(x,y)
    plt.plot(top10.iloc[i,2], [y[0]], marker = 'o', markersize = 15)


