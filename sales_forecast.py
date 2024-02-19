# Mona Mohammadi Feb 17th 2021 
# This code is made from Kaggle code 
# Import necessary librarires 


import numpy as np
import pandas as pd    #for reading CVS file / summurize the CVS file  
import seaborn as sns  #for scatterplot
import matplotlib.pyplot as plt 
import os 
import statmodels.formula.api as sm 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#import warnings


df = pd.read_cvs("C:\Users\Mona\Documents\Datasets\Advertising.cvs")
df.head()  
df.columns 
df.rename(columns={'Unnamed:0':'Index'}, inpalce= True)
df.info #check for null values 

#Visualization 
sns.pairplot(df, x_vars= ["TV", "Radio, "Newspaper", "Sales"], y_vars="Sales", kind= "reg") #Scatterplot to check the linearity assumption 
df.hist (bins= 24) # histograms to check the normality assumption 
sns.lmplot( x='TV', y='sales', data=df)
sns.lmplot( x= 'Radio', y='sales', data=df)
sns.lmplot(x= 'Newspaper', y='sales', data=df)
