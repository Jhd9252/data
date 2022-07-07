#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:53:32 2022
@ author: Jonathan Dinh

Data Analysis Report D1
Replication of Wallisch & Whritner, 2017.

DataSet desciption:
    This dataset features ratings data of 400 movies from 1097 research participants.
    First row: Headers (Movie titles/questions)
    Row 2-1098: Invididual participants
    Columns 1-400: Contain the ratings for the 400 movies (0 to 4, and missing data) 
    Columns 401-421: Contain self-assessments on sensation seeking behaviors (1-5) 
    Columns 422-464: Contain responses to personality questions (1-5)
    Columns 465-474: Contain self-reported movie experience ratings (1-5)
    Column 475: Gender identity (1 = female, 2 = male, 3 = self-described)
    Column 476: Only child (1 = yes, 0 = no, -1 = no response)
    Column 477: Movies are best enjoyed alone (1 = yes, 0 = no, -1 = no response)
"""

"""
D1 Assignment:
    1) Loads the dataset in ‘movieReplicationSet.csv’. This dataset contains the movie 
    rating data of 400 movies from 1097 research participants on a scale from 0 to 4. 
    Missing values in this dataset represent movies that the participant declined to 
    rate, for whatever reason (usually because they had not seen it).

    2) Computes the mean rating of each movie, across all participants (the output here 
    should be a variable that contains 400 means)

    3) Computes the median rating of each movie, across all participants (the output 
    here should be a variable that contains 400 medians)

    4) Computes the modal rating of each movie, across all participants (the output here 
    should be a variable that contains 400 modes)

    5) Computes the mean of all the mean ratings from 2) to calculate a “mean of means”

    *You can use the pre-existing mean, median and mode functions in Python. 
"""    

# imports
import pandas as pd # data handling
import numpy as np # data handling
import matplotlib.pyplot as plt # plotting
from sklearn.linear_model import LinearRegression # simple linear regression
from sklearn.metrics import mean_squared_error
import pingouin as ping # partial correlation 
from math import sqrt
from scipy.stats import ttest_ind, ttest_rel
import scipy.stats


# load in the data set as a pandas dataframe object with headers
# assert the proper shape of the data frame matches (1097 participants, 477 columns)
movie_df = pd.read_csv('movieReplicationSet.csv')
assert movie_df.shape == (1097, 477)

# basic descruptive statistics methods in pandas are built to exclude missing data
stats_df = movie_df.iloc[:,:400].describe()

# Compute the mean of each movie across all participants -> list of all 400 means
# means = movie_df.iloc[:,:400].mean() -> df option if easier
means = [x for x in stats_df.iloc[1,:]]
assert len(means) == 400

# compute median rating of each movie across all participants -> list of 400 medians.
# medians = movie_df.iloc[:,:400].median() -> df option if easier
medians = [x for x in movie_df.iloc[:,:400].median()]
assert len(medians) == 400

# compute the modal rating of each movie across all participants -> list of 400 modes
# modes_df = movie_df.iloc[:,:400].mode() -> df option if easier
modes = [x for x in movie_df.iloc[:,:400].mode().iloc[0,:]]

# compute the mean of all the mean ratings from step #2 in order to find mean of means
mean_means = pd.DataFrame(means).mean(skipna = True) # 2.63462 with pd skipna

# D1 Quiz
lowest_rating_by_mean = stats_df.iloc[:,means.index(min(means))]
highest_rating_by_median = stats_df.iloc[:, medians.index(max(medians))]
highest_rating_by_mean = stats_df.iloc[:,means.index(max(means))]
index_indendence = stats_df.columns.get_loc('Independence Day (1996)') 
modal_independence_day = movie_df['Independence Day (1996)'].mode()

"""
D2 Assignment:
    1) Built on D1, so the data should already be loaded
    2) Compute standard deviation of ratings of each movie, across all 
        participants -> variable that contains 400 standard deviations
    3) Compute MAD of the ratings of each movie across all participants -> 
        variable that contains 400 MADs                                                                     
    4) computes the mean and median of the SDs resulting from Q2.                                                                    
    5) computes the mean and median of MADs from Q3                                                            
    6) compute pairwise pearson correlation between all ratings of movie -> 
        output should be 400x400 correlation matrix. Okay for movie to 
        correlate with itself. Good way to check. Diagonals should be 1's.
        Length of correlation list should be 79800.                                                  
    * Be sure that all calculations include all 400 movies (check indexing)
    * handle all missing values skipna = True
"""

# compute SD for all movies across all participants
sd = [x for x in stats_df.iloc[2,:400]]

# compute MAD for all movies across all participants
mads = [x for x in movie_df.iloc[:, :400].mad()]

# compute the mean and median of SD from Q2 or sd
mean_sd = pd.DataFrame(sd).mean(skipna=True) 
median_sd = pd.DataFrame(sd).median(skipna=True) 


# compute the mean and median of MADs from Q3
mean_mads = pd.DataFrame(mads).mean(skipna= True)
median_mads = pd.DataFrame(mads).median(skipna=True)

# compute pairwise pearson correlation between ratings of all movies
corr_movies = movie_df.iloc[:, :400].corr(method = 'pearson')
assert corr_movies.shape == (400,400)

# compute the mean and median correlation resulting from Q6 (without self)
# grab all the upper triangular values of 400x400 corr matrix
corrs_without = []
counter = 1
for row in range(len(corr_movies)):
    for column in range(counter, len(corr_movies)):
        corrs_without.append(corr_movies.iloc[row, column])
    counter += 1
assert 1 not in corrs_without and len(corrs_without) == 79800

# compute the mean and median correlation resulting from Q6 (with self)
corrs_with = []
counter = 0
for row in range(len(corr_movies)):
    for column in range(counter, len(corr_movies)):
        corrs_with.append(corr_movies.iloc[row, column])
    counter += 1
assert 1 in corrs_with and len(corrs_with) == 79800 + 400

mean_corr_without = pd.DataFrame(corrs_without).mean(skipna=True)
mean_corr_with = pd.DataFrame(corrs_with).mean(skipna=True)
median_corr_without = pd.DataFrame(corrs_without).mean(skipna=True)
median_corr_with = pd.DataFrame(corrs_with).mean(skipna=True)

"""
D3:
    A. Find ratings of users who've rated both Star Wars I and Star Wars II. 
        - Build simple regression model - predicting the ratings of SWI from the ratings of SWII.
       - Return the betas, the R^2, and RMSE (averaged over all the joint ratings) of this model.
    B. Find the ratings of users who've rated both SW1 and Titanic. 
        - Build simple regression model - predicting ratings of Titanic from the ratings of SW1.
        - Return the betas, the R^2, and the RMSE (averaged over all joint ratings) of this model. 
    ** Keep as simple as possible. So no multiple regression, training, etc. This also means
    that we ignore all other columns of user. 
    Hint:
        Compute the RMSE as follows:
            calculate the deviation (difference) between 2 numbers (pred vs actual)
            square this deviation to get rid of the sign
            do this for all such deviations (all the users)
            sum up all squared deviations
            divide by the number of deviations to get the mean
            take the square root to undo the squaring and get back to original units
"""

# Get index of all star wars columns. Result is SW1 = 5, SW2 = 2
# get rid off nan values - keeping 0 star ratings
star_wars_index = [col for col in movie_df.columns if 'Star' in col]
star_wars_I_index = star_wars_index[5]
star_wars_II_index = star_wars_index[2]
star_wars_df = movie_df[[star_wars_index[5],star_wars_index[2]]].dropna(axis=0)
        
# build simple regression model - predict ratings of SWI from SWII
# keep as simple as possible / no training / meaning don't account for other columns of each user? 
# scikit linear regression has attrs coef_, rank_, singular_, intercept_, n_features_in_, feature_names_in
# scikit linear regression has methods fit(x, y[]), get_params, predict, score, set_params
x_star_wars = star_wars_df.iloc[:,1].values # ratings of sw2
y_star_wars = star_wars_df.iloc[:,0].values # ratings of sw1
x_star_wars = x_star_wars.reshape(len(x_star_wars), 1)
y_star_wars = y_star_wars.reshape(len(y_star_wars), 1)
model_star_wars = LinearRegression().fit(x_star_wars, y_star_wars)

# Evaluate model
# Return the betas, the R^2, and RMSE (averaged over all the joint ratings) of this model.
# yHat = slope * star_wars_df[:,2] + intercept # GLM
r_sq_star_wars = model_star_wars.score(x_star_wars, y_star_wars) # R^2
slope_star_wars = model_star_wars.coef_ # beta
intercept_star_wars = model_star_wars.intercept_ # intercept coeff / beta
pred_star_wars = model_star_wars.predict(x_star_wars) # returns pred sw1 values from sw2
rmse_star_wars = mean_squared_error(y_true = y_star_wars, y_pred = pred_star_wars, squared = False)

##############################################################################
# get index of star wars I and titanic
# get rid of nan values - keeping 0 star ratings
# SW1 is already isolated from previous model
star_wars_titanic_df = movie_df[[star_wars_index[5], 'Titanic (1997)']].dropna(axis = 0)

# build simple regression model - predict ratings of titanic from star wars 1
# keep as simple as possible / no training
x_star_wars1 = star_wars_titanic_df.iloc[:,0].values
y_titanic = star_wars_titanic_df.iloc[:,1].values
x_star_wars1 = x_star_wars1.reshape(len(x_star_wars1), 1)
y_titanic = y_titanic.reshape(len(y_titanic), 1)
model_star_wars_titanic = LinearRegression().fit(x_star_wars1, y_titanic)

# eval the model
# return the betas, R^2 and RMSE, intercept
# yHat = slope * star_wars_df[:,2] + intercept # GLM
r_sq_star_wars_titanic = model_star_wars_titanic.score(x_star_wars1, y_titanic) # R^2
slope_star_wars_titanic = model_star_wars_titanic.coef_ # beta
intercept_star_wars_titanic = model_star_wars_titanic.intercept_ # intercept coeff / beta
pred_titanic = model_star_wars_titanic.predict(x_star_wars1) # return pred titanic values from sw1
rmse_star_wars_titanic = mean_squared_error(y_true = y_titanic, y_pred = pred_titanic, squared = False)

################################################################################
# predict titanic = y from sw1 and sw2 = x

total_df = movie_df[[star_wars_index[5],star_wars_index[2], 'Titanic (1997)']].dropna(axis=0)

# build simple regression model - predict ratings of titanic from star wars 1
# keep as simple as possible / no training
x_total = total_df.iloc[:,[0,1]].values
y_total = total_df.iloc[:,2].values

x_total= x_total.reshape(len(x_total), 2)
y_total = y_total.reshape(len(y_total), 1)
model_total = LinearRegression().fit(x_total, y_total)

# eval the model
# return the betas, R^2 and RMSE, intercept
# yHat = slope * star_wars_df[:,2] + intercept # GLM
r_sq_total = model_total.score(x_total, y_total) # R^2
slope_total = model_total.coef_ # beta
intercept_total = model_total.intercept_ # intercept coeff / beta
pred_total = model_total.predict(x_total) # return pred titanic values from sw1
rmse_total = mean_squared_error(y_true = y_total, y_pred = pred_total, squared = False)


"""
Data Analysis Report 4:
    1. Find users who have rated “Kill Bill Vol. I (2003)”, “Kill Bill Vol. II (2004)” and “Pulp Fiction (1994)”.
    We will only consider ratings from users who have seen all 3 movies for the rest of this assignment. Make 
    sure you have no off-by-one errors, or everything else in this assignment will be wrong.
    2.  Finds the mean and median for all 3 of these movies (using only ratings from users identified in (1),
        who have seen and report ratings for all 3 movies, NOT from everyone)
    3. Does an independent-samples t-test between all 3 movies (all possibilities), using *only* ratings from 
        users identified in 2).
    4. Does a paired-samples t-test between all 3 movies (all possibilities), using using *only* ratings from users identified in 2).
"""

# find users who have rated all three movies from movie_df
# Kill Bill: Vol. 1 (2003)
# Kill Bill: Vol. 2 (2004)
# Pulp Fiction (1994)
d4df = movie_df[['Kill Bill: Vol. 1 (2003)','Kill Bill: Vol. 2 (2004)','Pulp Fiction (1994)']].dropna().reset_index(drop = True)

# calculate the mean and median for all three movies from participants who rated all three movies. 
d4fd_desc = d4df.describe() # means across columns

def median(data): # data being list
    d = sorted(data)
    i = len(d) // 2
    # if even, mean of two middle values
    mean = lambda data: sum(data) / len(data)
    return d[i] if len(d) % 2 == 1 else mean(d[i-1: i+1])

kb1_median = median([x for x in d4df['Kill Bill: Vol. 1 (2003)']])
kb2_median = median([x for x in d4df['Kill Bill: Vol. 2 (2004)']])
pf_median = median([x for x in d4df['Pulp Fiction (1994)']])

# perform independent t-test between all three movies (all possibilities)
    # compares means of 2 independent groups to determine statistical evidence that associated population means are differnt. 
# independent t-test between kb1 and kb2
ttest_i_kb1kb2 = ttest_ind(d4df.iloc[:,0], d4df.iloc[:,1])
# independent t-test between kb2 and pf
ttest_i_kb2pf = ttest_ind(d4df.iloc[:,1], d4df.iloc[:,2])
# independent t-test between kb1 and pf
ttest_i_kb1pf = ttest_ind(d4df.iloc[:,0], d4df.iloc[:,2])

# perform paired-sample t-test between all 3 movies (all possibilities)
    # test whether the mean difference beetween pairs of measurements is zero (correlated) or not. 
    # compare two populations means in the case of two samples that are correlated. 
# independent t-test between kb1 and kb2
ttest_red_kb1kb2 = ttest_rel(d4df.iloc[:,0], d4df.iloc[:,1])
# independent t-test between kb2 and pf
ttest_rel_kb2pf = ttest_rel(d4df.iloc[:,1], d4df.iloc[:,2])
# independent t-test between kb1 and pf
ttest_rel_kb1pf = ttest_rel(d4df.iloc[:,0], d4df.iloc[:,2])

"""
D5: 
    a. compute the mean rating of each movie in the entire dataset (use columnwise removal of missing data)
    b. compute the 95% Confidence interval around each of these means
    c. return array that arranges movies in increasing mean
    d. return an array that arranges the movies in increasing width of confidence interval
"""

# help function
# given 1d array of ratings (representing 1 movie)
# return mean
# return lower, upper bounds
# return distance from mean
# return width
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h, h, (m+h) - (m-h)

# find mean rating of each movie in data set, using columnwise removal
# return array with sorted movies in increasing order of mean
arr1 = []
for x,y in zip(list(movie_df.columns[0:400]), means):
    arr1.append((x, y))
sorted_movies_by_mean = sorted(arr1, key = lambda x: x[1])

# find 95% confidence interval around mean of each movie
# return array with sorted movies ascending by width
arr2 = []
for x in list(movie_df.columns[0:400]):
    holder = [x for x in movie_df[x].dropna()]
    w = mean_confidence_interval(holder)[4]
    arr2.append((x, w))
sorted_movies_by_widthCI = sorted(arr2, key = lambda x:x[1])
