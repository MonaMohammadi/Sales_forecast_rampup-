# Mona Mohammadi Feb 17th 2021 
# This code is made from Kaggle code 
# Import necessary librarires 


import numpy as np
import pandas as pd   #importing for reading CVS files 
import seaborn as sns 
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
