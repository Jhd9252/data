# -*- coding: utf-8 -*-


"""
Dataset description: This dataset features ratings data of 400 movies from 1097 research participants
and is contained in the file “movieReplicationSet.csv”. It is organized as follows:
    
1st row: Headers (Movie titles/questions) – note that the indexing in this list is from 1

Row 2-1098: Responses from individual participants

Columns 1-400: These columns contain the ratings for the 400 movies (0 to 4, and missing)
Columns 401-420: These columns contain self-assessments on sensation seeking behaviors (1-5)
Columns 421-464: These columns contain responses to personality questions (1-5)
Columns 465-474: These columns contain self-reported movie experience ratings (1-5)
Column 475: Gender identity (1 = female, 2 = male, 3 = self-described)
Column 476: Only child (1 = yes, 0 = no, -1 = no response)
Column 477: Social viewing preference – “movies are best enjoyed alone” (1 = y, 0 = n, -1 = nr)
"""


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as ping
from math import sqrt
from scipy.stats import ttest_ind, ttest_rel
import statsmodels.api as sm
import seaborn as sns
from matplotlib import colors
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# import the data and get basic information
df = pd.read_csv('MovieReplicationSet.csv')
assert df.shape == (1097, 477)
df_desc = df.describe()
df_corr = df.iloc[:,401:].corr()

# info on missing entries
missing_data = df.isna().sum().sum()
total_entries = df.shape[0] * df.shape[1]
percentage_missing_entry = missing_data / total_entries

# Pre-Processing the data
    # Combine or remove any unnecesaary rows
    # Choose to either remove data row-wise, or input missing data with justification
    # z-scoring the data
    # dimensionality reduction in three 
    
# Choosing to remove any missing entries row wise. Inputting roughly 62% of missing
# entries by Mean/Median/Mode or regression would reduce the variance of input variables
# too much, or cause high correlation with dependent variables by selection. This also assumes
# the relationship is linear. 

# scaling and PCA of data (relevant to all questions)
# grab the features
df_feat = df.iloc[:,400:474]

# remove and missing data (Justification, each particpant can and will answer most of personal questionaire, 
# versus not seeing movie, therefore there will be little missing data). 
df_feat.dropna(inplace = True)

# scale each of the features (transform, z-scoring)
scaler = StandardScaler()
df_feat_scaled = pd.DataFrame(scaler.fit_transform(df_feat), columns = df_feat.columns)

# separate out features columns
sensation = df_feat_scaled.iloc[:,:20]
personality = df_feat_scaled.iloc[:,20:63]
movie_exp = df_feat_scaled.iloc[:,63:]

# method of PCA for dimensionality reduction, finds lower-dimensional space by preserving variance
# unsupervised method. Linear transformations, finds principal components, which is decomposition 
# of feature matrix into eiganvectors. So PCA is not effective when distribution of dataset is not linear. 
# step 1: calculate the correlation matrix
# step 2: factor extraction
# step 3: determining the number of independent factors (screeplot, showing elbow + kaiser methods)
# step 4: interpreting their meaning
# step 5: determining the factor values of the original data

# perform PCA on sensation seeking
pca = PCA(n_components = sensation.shape[1])
df_sensation_pca = pd.DataFrame(pca.fit_transform(sensation)) 
num_kaiser = len([x for x in pca.explained_variance_ if x>=1])
df_sensation_pca = df_sensation_pca.iloc[:,:num_kaiser]



# perform PCA on personality
pca = PCA(n_components = personality.shape[1])
df_personality_pca = pd.DataFrame(pca.fit_transform(personality))
plt.plot(np.arange(pca.n_components_) + 1, pca.explained_variance_, 'o-', linewidth =2, color='red')
plt.show()
num_kaiser = len([x for x in pca.explained_variance_ if x>=1])
df_personality_pca = df_personality_pca.iloc[:,:num_kaiser]


loadings = np.transpose(pca.components_[:,:7].T * np.sqrt(pca.explained_variance_))
loading_matrix = pd.DataFrame(loadings, columns = ['pca1','pca2','pca3','pca4','pca5','pca6','pca7'], index = personality.columns)
# pca1 = does a thorough job
# pca2 = talkative and careless
# pca3 = shy/inhibited, does things efficiently, worries a lot
# pca4 = is talkative
# pca5 = original, comes up with new ideas
# pca6 = talkative
# pca7 = depressed, generates alot of enthesiam

# perform PCA on movie_experience
pca = PCA(n_components = movie_exp.shape[1])
df_movie_exp_pca = pd.DataFrame(pca.fit_transform(movie_exp))
num_kaiser = len([x for x in pca.explained_variance_ if x>=1])
df_movie_exp_pca = df_movie_exp_pca.iloc[:,:num_kaiser]


################################################################################################
# Q1. What is the relationship between sensation seeking and movie experience?
# Scaled data
# PCA on each category (sensation seeking, movie experience, personality)
# get averages of each partipant in each category
# numpy correlation states that correlation = 0.00742265, basically no relationship
average_sensation = []
for i in range(df_sensation_pca.shape[0]):
    s = 0
    for j in range(df_sensation_pca.shape[1]):
        s += df_sensation_pca.iloc[i,j]
    average_sensation.append(s/df_sensation_pca.shape[1])
    
average_movie_exp = []
for i in range(df_movie_exp_pca.shape[0]):
    s = 0
    for j in range(df_movie_exp_pca.shape[1]):
        s += df_movie_exp_pca.iloc[i,j]
    average_movie_exp.append(s/df_movie_exp_pca.shape[1])

corr_sensation_movie_exp = np.corrcoef(average_sensation, average_movie_exp)

################################################################################################
# Q2. Is there evidence of personality types based on data of participants? If so
# characterize these types both quantitaively and narratively.

# need scaled data of personality
# PCA of personality

# getting optimal k-val using elbow method k = 3
# elbow = calc WCSS (within cliuster sum of square)
elbow = KElbowVisualizer(KMeans(), k = (1,10))
elbow.fit(df_personality_pca)
elbow.show()

# generate clusters, predict participants
kmeans = KMeans(n_clusters = 3)
model = kmeans.fit_predict(df_personality_pca)
df_personality_pca['cluster'] = model

# grab each cluster
personality_cluster0 = df_personality_pca[df_personality_pca['cluster'] == 0]
personality_desc0 = personality_cluster0.describe()
personality_cluster1 = df_personality_pca[df_personality_pca['cluster'] == 1]
personality_desc1 = personality_cluster1.describe()
personality_cluster2 = df_personality_pca[df_personality_pca['cluster'] == 2]
personality_desc2 = personality_cluster2.describe()

# from this we can see that, going with highest and lowest traits

# pca1 = does a thorough job
# pca2 = careless
# pca3 = shy/inhibited, worries a lot, efficient
# pca4 = is talkative
# pca5 = original, comes up with new ideas
# pca6 = talkative
# pca7 = depressed, generates alot of enthesiam

# cluster 0: high level of thoroughness at job, low level of shyness, worry and efficieny. 
# cluster 1: low levels of carelessness, high level of shyness, worries and efficiency
# cluster 2: lowest level of thoroughness at job, high levels of carelessness

# therefore yes, there is evidence of personality types 

#################################################################################################
# Q3. Are movies that are more popular rated higher than movies that are less popular?

# mean of each movie
# operationalize the popularity of movie by how many ratings it has. 
means = np.array(df_desc.iloc[1,:400]).reshape(len(df_desc.iloc[1,:400]), 1)
popularity = np.array(df_desc.iloc[0,:400]).reshape(len(df_desc.iloc[0,:400]), 1)

# there is a high positive linear correlation between popularity and rating, around 0.7
corr_popularity_rating = np.corrcoef(popularity, means)

model = LinearRegression().fit(popularity, means)
r_sq = model.score(popularity, means) # R^2
slope = model.coef_ # beta
intercept = model .intercept_ # intercept coeff / beta
pred_ratings = model.predict(popularity) # returns pred sw1 values from sw2
rmse = mean_squared_error(y_true = means, y_pred = pred_ratings, squared = False)

plt.plot(popularity, popularity * slope + intercept)
for i in range(len(means)):
    plt.plot(popularity[i], means[i], marker = 'o', markersize = 2, color = 'red')
plt.show()
############################################################################################
# Q4. Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
dfq4 = df[['Shrek (2001)','Gender identity (1 = female; 2 = male; 3 = self-described)']].dropna()

# separate self-identified genders
shrek_gender1 = dfq4[dfq4['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]
shrek_gender2= dfq4[dfq4['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]
shrek_gender3= dfq4[dfq4['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 3]

shrek_gender1_desc = shrek_gender1.describe()
shrek_gender2_desc = shrek_gender2.describe()
shrek_gender3_desc = shrek_gender3.describe()

plt.scatter(dfq4.iloc[:,1], dfq4.iloc[:,0])
plt.show()

ttest_ind_12 = ttest_ind(shrek_gender1.iloc[:,0], shrek_gender2.iloc[:,0], axis = 0, nan_policy='omit',equal_var=False)
ttest_ind_13 = ttest_ind(shrek_gender1.iloc[:,0], shrek_gender3.iloc[:,0], axis = 0, nan_policy='omit',equal_var=False)
ttest_ind_23 = ttest_ind(shrek_gender2.iloc[:,0], shrek_gender3.iloc[:,0], axis = 0, nan_policy='omit',equal_var=False)
# the average rating of shrek or self-identified females is = 3.15545
# the std of rating for self-identified females is 0.906547
# the range of ratings for self-identified females is 4

# the average rating of shrek or self-identified males is = 3.08299
# the std of rating for self-identified females is 0.824975
# the range of ratings for self-identified females is 3.5

# the average rating of shrek or self-identified males is = 3.25
# the std of rating for self-identified females is 0.524404
# the range of ratings for self-identified females is 1.5

# generally the participants who identied as gender other had a slightly higher average rating of
# shrek with the smallest amount of variance or spread
# this same pattern is followed by males, then females. 

# in addition, doing an independent ttest for 12, 13, 23 results in pvalues 0.24834907946281018, 0.679928726393761, 0.47941799800855334
# respectively, which are all greater than alpha = 0.05, therefore there is no statistical difference between the means of the three groups. 
# parametric test. There is no statistical evidence that associated population means of ratings are different. 
############################################################################################
#Q5. Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
dfq5 = df[['The Lion King (1994)', 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)']].dropna()

# will be excluding those that did not respond to this question.
# separate out those that are an only child and those that are not
only_child = dfq5[dfq5['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']==1]
not_only_child = dfq5[dfq5['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']==0]
only_child_desc = only_child.describe()
not_only_child_desc = not_only_child.describe()

plt.scatter(dfq5.iloc[:,1], dfq5.iloc[:,0])
plt.show()


# only child mean rating for lion king is 3.34768
# only child std is 0.816483
# only child range is 4

# not only child mean rating for lion king is 3.48196
# not only child std is 0.718194
# not only child range is 4

ttest_ind_only_not_child = ttest_ind(only_child.iloc[:,0], not_only_child.iloc[:,0], nan_policy='omit', equal_var = True)
# ttest ind has pvalue = 0.06102886373552747, we omit nans, and assume unequal variances since sample size is small, and only for those
# that have self-selected/chosen to watch shrek. In addition, sample sizes between only child and not are hugely uneven. 
#######################################################################
# Q6. Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone?

# grab columns
dfq6 = df[['The Wolf of Wall Street (2013)', 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']].dropna()

# separate out those that enjoyed movies alone vs those who do not
# leave out those that did not respond or did not rate wolf of wallstreet.
alone = dfq6[dfq6['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 1]
not_alone = dfq6[dfq6['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 0]
alone_desc = alone.describe()
not_alone_desc = not_alone.describe()

# those that enjoyed movies alone mean rating of 3.14377
# std = 0.869886
# rating range of 4

# those who do not enjoy watching movies alone mean rating of 3.03333
# std = 0.921047
# range of 4
ttest_ind_alone_not = ttest_ind(alone.iloc[:,0], not_alone.iloc[:,0], nan_policy = 'omit',equal_var = False)
# this ttest ind gives us a pvalue of 0.12139103950020742. Not assuming equal variance because of uneven sample size, and self selection into movie watched). This provides evidence that there is no significant difference in population rating of movie from viewing preference. Therefore fail to reject null hypothesis that there is no difference. even though sample sizes are not significanlty uneven. 

####################################################################################################################
#Q7 IS EXTRA CREDIT
####################################################################################################################
# Q8. Build a prediction model of your choice (regression or supervised learning) to predict movie
# ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation
# methods to avoid overfitting and characterize the accuracy of your model.

# multiple linear regression with PCA 1-7 as input, and means of movies as output
# y = a + B1P1 + B2P2 + B3P3 + ..... + B8P8

# scale the df
dfq8 = df.iloc[:,:400]
# scale the data
scaler = StandardScaler()
df_movie_scaled = pd.DataFrame(scaler.fit_transform(dfq8), columns = dfq8.columns)

# match the personality pca length to movie length by dropping rows where null in personality features = 953
df_holder = df.iloc[:,400:474].dropna().reset_index()

# drop the movies with any nan values in pca by grabbing the same indexes
index = list(df_holder.iloc[:,0].values)
df_movie_scaled = df_movie_scaled.iloc[index, :].reset_index(drop=True)

df_personality_pca.drop(['cluster'], axis = 1, inplace = True)

# now we have df_personality_pca (953x7) and df_movie_scaled (953x400)
# for each movie, create model, traintest split, and add r^2
r_sq = []
coeffdeter = []
rmse = [] 
model = LinearRegression()
df_new = pd.concat([df_movie_scaled, df_personality_pca], axis = 1)

for col in range(df_movie_scaled.shape[1]):
    df_newer = df_new.iloc[:,[col, 400,401,402,403,404,405,406]].dropna()
    
    x = df_newer.iloc[:,1:].values
    y = df_newer.iloc[:,0].values
    
    x = x.reshape(len(x), 7)
    y = y.reshape(len(x), 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    model = model.fit(x_train, y_train)
    
    # model eval 
    yPred = model.predict(x_test) # ypredictions
    slope = model.coef_ # slope
    intercept = model.intercept_ # intercept
    r_sqT = model.score(x, y) # r_sq
    rmse_total = mean_squared_error(y_true = y_test, y_pred= yPred, squared = False) # rmse
    
    # add to overall stats
    r_sq.append(r_sqT)
    rmse.append(rmse_total)

plt.scatter(np.arange(400), r_sq, marker = 'o', color = 'red')
plt.plot(np.arange(400), [np.mean(r_sq)]*400)
plt.title('Scatter Plot of R^2 For Movie')
plt.show()

plt.scatter(np.arange(400), rmse, marker = 'o', color = 'blue')
plt.plot(np.arange(400), [np.mean(rmse)]*400)
plt.title('Scatter Plot of RMSE For Movie')
plt.show()
print(np.mean(r_sq))
print(np.mean(rmse))

###############################################################################################
"""
Build a prediction model of your choice (regression or supervised learning) to predict movie
ratings (for all 400 movies) from gender identity, sibship status and social viewing preferences
(columns 475-477) only. Make sure to use cross-validation methods to avoid overfitting and
characterize the accuracy of your model.
"""

# have df_movie_scaled - scaled
# need df_social - no need for scaling
df_social = df.iloc[:,474:]
dfq9 = pd.concat([df_movie_scaled, df_social], axis = 1)

# now we have df_personality_pca (953x7) and df_movie_scaled (953x400)
# for each movie, create model, traintest split, and add r^2
r_sq = []
rmse = [] 
model = LinearRegression()

for col in range(df_movie_scaled.shape[1]):
    df_newer = dfq9.iloc[:,[col, 400,401,402]].dropna()
    
    x = df_newer.iloc[:,1:].values
    y = df_newer.iloc[:,0].values
    
    x = x.reshape(len(x), 3)
    y = y.reshape(len(x), 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    model = model.fit(x_train, y_train)
    
    # model eval 
    yPred = model.predict(x_test) # ypredictions
    slope = model.coef_ # slope
    intercept = model.intercept_ # intercept
    r_sqT = model.score(x, y) # r_sq
    rmse_total = mean_squared_error(y_true = y_test, y_pred= yPred, squared = False) # rmse
    
    # add to overall stats
    r_sq.append(r_sqT)
    rmse.append(rmse_total)

plt.scatter(np.arange(400), r_sq, marker = 'o', color = 'red')
plt.plot(np.arange(400), [np.mean(r_sq)]*400)
plt.title('Scatter Plot of R^2 For Each Movie')
plt.show()
plt.scatter(np.arange(400), rmse, marker = 'o', color = 'blue')
plt.plot(np.arange(400), [np.mean(rmse)]*400)
plt.title('Scatter Plot of RMSE For Each Movie')
plt.show()
print(np.mean(r_sq))
print(np.mean(rmse))

########################################################################################
# Q10
df_movie_scaled = df_movie_scaled
# need df social index to match df movie index, drop na in social
df_social = df_social.iloc[index, :].reset_index(drop=True)
df_all_feat = pd.concat([df_personality_pca, df_movie_exp_pca, df_sensation_pca, df_social], axis = 1) # 2 + 7 + 6 + 3 = 18
df_newest = pd.concat([df_movie_scaled, df_all_feat], axis = 1)

r_sq = []
rmse = [] 
model = LinearRegression() 

for movie in range(df_movie_scaled.shape[1]):
    df_newest = pd.concat([df_movie_scaled.iloc[:,movie], df_all_feat], axis = 1).dropna()
    
    x = df_newest.iloc[:,1:].values
    y = df_newest.iloc[:,0].values
    x = x.reshape(len(x), 18)
    y = y.reshape(len(y), 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    model = model.fit(x_train, y_train)
    
    # model eval
    # model eval 
    yPred = model.predict(x_test) # ypredictions
    slope = model.coef_ # slope
    intercept = model.intercept_ # intercept
    r_sqT = model.score(x, y) # r_sq
    rmse_total = mean_squared_error(y_true = y_test, y_pred= yPred, squared = False) # rmse
    
    # add to overall stats
    r_sq.append(r_sqT)
    rmse.append(rmse_total)

plt.scatter(np.arange(400), r_sq, marker = 'o', color = 'red')
plt.plot(np.arange(400), [np.mean(r_sq)]*400)
plt.title('Scatter Plot of R^2 For Each Movie')
plt.show()
plt.scatter(np.arange(400), rmse, marker = 'o', color = 'blue')
plt.plot(np.arange(400), [np.mean(rmse)]*400)
plt.title('Scatter Plot of RMSE For Each Movie')
plt.show()
print(np.mean(r_sq))
print(np.mean(rmse))










