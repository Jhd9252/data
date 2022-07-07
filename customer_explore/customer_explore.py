# -*- coding: utf-8 -*-

"""
Kaggle Dataset: https://www.kaggle.com/imakash3011/customer-personality-analysis
    Kaggle Author: https://www.kaggle.com/imakash3011
    Dataset provided by: Dr. Omar Romero-Hernandez.

Columns:
    ID
    Year_Birth
    Education
    Marital_Status
    Income
    Kidhome
    Teenhome
    Dt_Customer: date of customer enrollment in customer's household
    Recency: number of days since customers last purchase
    Complain: 1 if customer complained in last 2 years, 0 otherwise

    MntWines: amount spent on wine in last 2 years
    MntFruits: amount spent on fruits in last 2 years
    MntMeatProducts: amount spent on meat in last 2 years
    MntFishProducts: amount spent on fish in last 2 years
    MntSweetProducts: amount spent on sweets in last 2 years
    MntGoldProds: amount spent on gold in last 2 years

    NumDealsPurchases: number of purchases made with a discount
    AcceptedCmp1: 1 if accepted offer in 1st campaign, 0 otherwise
    AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
    AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
    AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
    Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

    NumWebPurchases: Number of purchases made through the company’s website
    NumCatalogPurchases: Number of purchases made using a catalogue
    NumStorePurchases: Number of purchases made directly in stores
    
    NumWebVisitsMonth: Number of visits to company’s website in the last month
    
Goals: Agglomerative Hierarchical Clustering / AGNES
    - most common type of hierarchical clustreing based on similarity. 
    - starts by treating each object as leaf
    - pairs of clusters are merged until all clusters merged into 1. 
    - result is a tree-based represnetion of objects
    - can fine tune how course or fine to cluster
    - vs Kmeans which can be used to predict from a standing model, AGNES must 
    be trained again and cannot be used to predict right with new data. 

"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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



# read in csv 'marketing_campaign.csv'
df = pd.read_csv('marketing_campaign.csv', sep = '\t')
print('Dimensions of dataset: ', df.shape) 

# looking at categories in education and marital status
print(df.Marital_Status.value_counts())
print(df.Education.value_counts())

"""
Initial Look
    there are missing values
    categorical values in marital status and education 
    Dt_Customer is not in datetime format
    There are some attrs/cols we can simplfy/combine/derive
    look for any outliers
    
Data cleaning 
    [x] Drop missing data for simple exploration
    [x] parse dtcustomer into date time format
    [x] create an age col from year of birth (easier to work with)
    [x] create col for total spent from Mnt cols 
    [x] create kids = teens + children
    [x] create total number of purchases from num cols
    [x] simplify marital status
    [x] simplify education status
    [x] remove any redundant cols
    [x] create household size
    [x] remove any outliers
    
"""

# for this exploration, we are wimply going to drop rows with missing data in any column
df = df.dropna()
print('Dimensions of dataset after removing missing values: ', df.shape)

# parsing Dt_Customer column into DateTime format YYYY:MM:DD 
df.Dt_Customer = pd.to_datetime(df.Dt_Customer)
df.drop(columns=['Dt_Customer'], inplace = True)

# create age column from year_birth col (not exact, which is fine for simple)
df['age'] = 2022 - df.Year_Birth
df.drop(columns=['Year_Birth'], inplace = True)

# create total spent column 
df['MntTotal'] = df.MntWines + df.MntFruits + df.MntMeatProducts + df.MntFishProducts + df.MntSweetProducts + df.MntGoldProds

# create new column for total kids = children + teens
# remove df.kids, df.teens
df['kids'] = df.Kidhome + df.Teenhome
df.drop(columns = ['Kidhome', 'Teenhome'], inplace = True)

# create total num of purchases from catolgue, internet, store
df['TotalPurchases'] = df.NumCatalogPurchases + df.NumStorePurchases + df.NumWebPurchases

# Simplfy martial status into
# married = married + together
# single = single  + divorced + widow + alone
# other = absurd + yolo
df.replace({'Marital_Status': {'Together':'Married',
                               'Divorced':'Single',
                               'Widow':'Single',
                               'Alone':'Single',
                               'Absurd':'Other',
                               'YOLO':'Other'}}, inplace = True)

# simplify education status
# high school = basic
# postgrad = masters, pdh
# bachelors = graduation, 2ndcycle
df.replace({'Education': {'Basic':'high school',
                          'Master':'post grad',
                          'PhD': 'post grad',
                          'Graduation':'bachelors',
                          '2n Cycle': 'bachelors'}}, inplace = True)
                    
# create household size col
# kids + marital status (married = 2, single = 1, other = 3 (keep it simple, at least 3))
df['household'] = df.kids + df.Marital_Status.replace({'Married': 2,
                                                       'Single': 1,
                                                       'Other': 3})

# removing outliers after viewing dataset
# see max age is 129
# max income is 666,666
# remove those outliers / set limits on outliers
# set max income to 600,000 -> removes 1 datapoint
# set max age to less than 100, removes 3 datapoints
df_describe = df.describe()
df = df[(df.age < 100) & (df.Income < 200000)]

# removing redundant columns we won't be concerned with (campaigns)
df.drop(columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain','Response','Z_CostContact','Z_Revenue'], inplace = True)

"""
Initial Exploration
    [] pd.describe
    [] correlations
"""
df_describe = df.describe()
df_corr = df.corr()

"""
Data processing for K-Means:
    [] change the categorical features -> label encoding: which means converting labels
    into numeric format so they are machine readable in algorithms
    [] want to scale the features using standard scalar
    [] dimensionality reduction using PCA
"""
# grab categorical variables
category = (df.dtypes == 'object') # gives true/false series of categorical types
category_cols = list(category[category].index) # create list of true values at that index = cols

# label encode the categorical columns
# now all attributes are numerical
encoder = LabelEncoder() 
encoder2 = LabelEncoder()
df[category_cols[0]] = df[[category_cols[0]]].apply(encoder.fit_transform)
df[category_cols[1]] = df[[category_cols[1]]].apply(encoder2.fit_transform)


# scale the data since we have many variables
df2 = df.copy()
scaler = StandardScaler()
scaler.fit(df2)
df3 = pd.DataFrame(scaler.transform(df2), columns = df2.columns)

# dimensionality reduction using PCA
# PCA: technique for reducing dimensionality, increasing interpretability and minimizing data loss
# reducing dimensions to 3
pca = PCA(n_components = 3)
pca.fit(df3)
pca_df = pd.DataFrame(pca.transform(df3), columns = ['col1','col2','col3'])

# plotting 3d projection in reduced dimension
x = pca_df['col1']
y = pca_df['col2']
z = pca_df['col3']
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x,y,z, c = 'maroon', marker = 'o')
ax.set_title('3D Projection in Reduced Dimension')
plt.show()

"""
Clustering 
    [] elbow method to find optimal k-val of predetermined clusters
    [] clustering using agnes
"""

# getting optimal k-val using elbow method = 4
elbow = KElbowVisualizer(KMeans(), k =10)
elbow.fit(pca_df)
elbow.show()

# fitting agnes to get final clusters
# start model
agnes = AgglomerativeClustering(n_clusters = 4)
# fit the model
model = agnes.fit_predict(pca_df)

# add cluster attribute/label to orignal dataframe
df['cluster'] = model

#Plotting the clusters
# cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
cmap = colors.ListedColormap(["#F0F8FF", "#F5F5DC", "#DEB887", "#DC143C", "#A9A9A9", "#EE82EE"])
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s = 40, c = df["cluster"], marker='o', cmap = cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()

"""
Exploring and evalutaing the model
    distribution of clusters
    check each attribute vs clusters to get idea of each cluster
"""

# check distribution of clusters
# x = cluster
# y = counts of clusters
cluster_distro = pd.DataFrame(df.cluster.value_counts().sort_index())
labels = ['Alpha', 'Bravo','Charlie','Delta']
ax.set_xticks(ticks = [0,1,2,3], labels = labels)
plt.ylabel('# Of Customers Per Cluster')
plt.title('Distribution of Clusters Alpha to Delta')
plt.bar(cluster_distro.index, cluster_distro.cluster, color = ['green','red','blue', 'yellow'],
        tick_label = labels)

# decode the education and marital status columns of each cluster dataframe
df[category_cols[0]] = encoder.inverse_transform(df[category_cols[0]])
df[category_cols[1]] = encoder2.inverse_transform(df[category_cols[1]])


# grab the numeric medians of each cluster

alpha_df = df[df['cluster'] == 0]
alpha_med = alpha_df.median(numeric_only=True)
bravo_df = df[df['cluster'] == 1]
bravo_med = bravo_df.median(numeric_only =True)
charlie_df = df[df['cluster'] == 2]
charlie_med = charlie_df.median(numeric_only =True)
delta_df = df[df['cluster'] == 3]
delta_med = charlie_df.median(numeric_only=True)

# grab the most common val of education and marital status of each cluster
alpha_edu_med = alpha_df['Education'].value_counts().idxmax()
alpha_mar_med = alpha_df['Marital_Status'].value_counts().idxmax()

bravo_edu_med = bravo_df['Education'].value_counts().idxmax()
bravo_mar_med = bravo_df['Marital_Status'].value_counts().idxmax()

charlie_edu_med = charlie_df['Education'].value_counts().idxmax()
charlie_mar_med = charlie_df['Marital_Status'].value_counts().idxmax()

delta_edu_med = delta_df['Education'].value_counts().idxmax()
delta_mar_med = delta_df['Marital_Status'].value_counts().idxmax()


# print each median customer attribute per cluster       
print('Alpha Cluster:',
      '\n\t', str('Median Age: ').ljust(30), alpha_med.loc['age'],
      '\n\t', str('Median Education: ').ljust(30), alpha_edu_med,
      '\n\t', str('Median Marital Status: ').ljust(30), alpha_mar_med,
      '\n\t', str('Median Income: ').ljust(30),  alpha_med.loc['Income'],
      '\n\t', str('Median $$ Spent: ').ljust(30),  alpha_med.loc['MntTotal'],
      '\n\t', str('Median Household Size: ').ljust(30),  alpha_med.loc['household'],
      '\n\t', str('Median Deals Purchased: ').ljust(30),  alpha_med.loc['NumDealsPurchases'],
      '\n\t', str('Median Web Purchases: ').ljust(30),  alpha_med.loc['NumWebPurchases'],
      '\n\t', str('Median Catalog Purchases: ').ljust(30),  alpha_med.loc['NumCatalogPurchases'],
      '\n\t', str('Median Store Purchases: ').ljust(30),  alpha_med.loc['NumStorePurchases'],
      '\n\t', str('Median Web Vists Per Month: ').ljust(30),   alpha_med.loc['NumWebVisitsMonth'] )

print('Bravo Cluster:',
      '\n\t', str('Median Age: ').ljust(30), bravo_med.loc['age'],
      '\n\t', str('Median Education: ').ljust(30), bravo_edu_med,
      '\n\t', str('Median Marital Status: ').ljust(30), bravo_mar_med,
      '\n\t', str('Median Income: ').ljust(30),  bravo_med.loc['Income'],
      '\n\t', str('Median $$ Spent: ').ljust(30),  bravo_med.loc['MntTotal'],
      '\n\t', str('Median Household Size: ').ljust(30),  bravo_med.loc['household'],
      '\n\t', str('Median Deals Purchased: ').ljust(30),  bravo_med.loc['NumDealsPurchases'],
      '\n\t', str('Median Web Purchases: ').ljust(30),  bravo_med.loc['NumWebPurchases'],
      '\n\t', str('Median Catalog Purchases: ').ljust(30),  bravo_med.loc['NumCatalogPurchases'],
      '\n\t', str('Median Store Purchases: ').ljust(30),  bravo_med.loc['NumStorePurchases'],
      '\n\t', str('Median Web Vists Per Month: ').ljust(30),   bravo_med.loc['NumWebVisitsMonth'] )

print('Charlie Cluster:',
      '\n\t', str('Median Age: ').ljust(30), charlie_med.loc['age'],
      '\n\t', str('Median Education: ').ljust(30), charlie_edu_med,
      '\n\t', str('Median Marital Status: ').ljust(30), charlie_mar_med,
      '\n\t', str('Median Income: ').ljust(30), charlie_med.loc['Income'],
      '\n\t', str('Median $$ Spent: ').ljust(30),  charlie_med.loc['MntTotal'],
      '\n\t', str('Median Household Size: ').ljust(30),  charlie_med.loc['household'],
      '\n\t', str('Median Deals Purchased: ').ljust(30),  charlie_med.loc['NumDealsPurchases'],
      '\n\t', str('Median Web Purchases: ').ljust(30),  charlie_med.loc['NumWebPurchases'],
      '\n\t', str('Median Catalog Purchases: ').ljust(30),  charlie_med.loc['NumCatalogPurchases'],
      '\n\t', str('Median Store Purchases: ').ljust(30),  charlie_med.loc['NumStorePurchases'],
      '\n\t', str('Median Web Vists Per Month: ').ljust(30),   charlie_med.loc['NumWebVisitsMonth'] )

print('Delta Cluster:',
      '\n\t', str('Median Age: ').ljust(30), delta_med.loc['age'],
      '\n\t', str('Median Education: ').ljust(30), delta_edu_med,
      '\n\t', str('Median Marital Status: ').ljust(30), delta_mar_med,
      '\n\t', str('Median Income: ').ljust(30),  delta_med.loc['Income'],
      '\n\t', str('Median $$ Spent: ').ljust(30),  delta_med.loc['MntTotal'],
      '\n\t', str('Median Household Size: ').ljust(30),  delta_med.loc['household'],
      '\n\t', str('Median Deals Purchased: ').ljust(30),  delta_med.loc['NumDealsPurchases'],
      '\n\t', str('Median Web Purchases: ').ljust(30),  delta_med.loc['NumWebPurchases'],
      '\n\t', str('Median Catalog Purchases: ').ljust(30),  delta_med.loc['NumCatalogPurchases'],
      '\n\t', str('Median Store Purchases: ').ljust(30),  delta_med.loc['NumStorePurchases'],
      '\n\t', str('Median Web Vists Per Month: ').ljust(30),   delta_med.loc['NumWebVisitsMonth'] )