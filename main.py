import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
# import plotly.express as px
from pandas import plotting
import missingno as ms
from sklearn.impute import SimpleImputer
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from visualizer import Visualizer
# from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

plt.style.use("ggplot") #setting the plot style
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("data/Lead_Scoring.csv")
# df.shape > give count of raw*col
# df.info() > gives column details
# print(df.isnull().sum()/df.shape[0]*100).sort_values(ascending = False))

num_col = df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].keys()
cat_col = df.dtypes[~(df.dtypes == 'int64') & ~(df.dtypes == 'float64')].keys()

num_df = df[num_col]
cat_df = df[cat_col]


# number data filtaration 
se_median = SimpleImputer(missing_values = np.nan, strategy='median')

num_df[['TotalVisits', 'Page Views Per Visit']] = se_median.fit_transform(num_df[['TotalVisits', 'Page Views Per Visit']])
num_df = num_df.drop(['Asymmetrique Activity Score', 'Asymmetrique Profile Score'], axis = 1)
# print(num_df.isnull().sum())



cat_null_cols = (cat_df.isnull().sum()/cat_df.shape[0]*100).sort_values(ascending = False)[:13].keys()
# print(cat_null_cols)


### Categorical Data Feature Engineering + (Analysis)

cat_null_cols = (cat_df.isnull().sum()/cat_df.shape[0]*100).sort_values(ascending = False)[:13].keys()
# print(cat_null_cols)

#creating a function which will iterate each value and check for the data type and returning if there as any int ot float values present 
def check_str(data, col):
    
    for i in data[col]: 
        if isinstance(i, str):
            continue
        else:
            print("Alert: Float/Int Values Found in Categorical Column..!!!")
            break
        
check_str(cat_df[cat_null_cols], 'Lead Quality')

check_str(cat_df[cat_null_cols].fillna(value = str(np.nan)), 'Lead Quality')

new_cat_df = cat_df[cat_null_cols]

# print(">>>",new_cat_df)

temp_cat_df = new_cat_df.drop(["Lead Quality", "Asymmetrique Profile Index", "Asymmetrique Activity Index"], axis = 1)

idx = cat_df.isnull().sum()[cat_df.isnull().sum() < 1].keys()

temp_cat_df = pd.concat([temp_cat_df, cat_df[idx]], axis = 1)
temp_cat_df['Lead Source'].replace(to_replace = 'google', value = 'Google', inplace = True)
temp_cat_df['Lead Source'].replace(to_replace = np.nan, value = 'Others', inplace = True)
temp_cat_df['Last Activity'].replace(to_replace = np.nan, value = 'Email Opened', inplace = True)

cat_df_copy = temp_cat_df.copy()
cat_df_copy.drop("Country", axis = 1, inplace = True)

cat_df_copy['City'].replace(to_replace = ['Select', np.nan], value = 'Unspecified', inplace = True)
cat_df_copy['Specialization'].replace(to_replace = ['Select', np.nan], value = 'Unspecified', inplace = True)
cat_df_copy['Lead Profile'].replace(to_replace = ['Select', np.nan], value = 'Unspecified', inplace = True)
cat_df_copy['How did you hear about X Education'].replace(to_replace = ['Select', np.nan], value = 'Unspecified', inplace = True)

#imputing with most frequent item in a itemset
cat_df_copy["What matters most to you in choosing a course"].replace(to_replace = np.nan, value = 'Better Career Prospects', inplace = True)
cat_df_copy["What is your current occupation"].replace(to_replace = np.nan, value = 'Unemployed', inplace = True)