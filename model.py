import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import env
import model
import os
#-----------------------------------------------------------------------------------------------------------
#acquire 

#reading the list of prepared pickle file contained in the csv file
df = pd.read_pickle('prepared.pkl 3')

#-----------------------------------------------------------------------------------------------------------
#split data

def my_train_test_split(df, target):
    ''' 
    This function takes in a dataframe and splits data into 3 samples of train (60%), validate(20%) and test(20%).  
    '''
    
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    
    return train, validate, test

    #train, validate, test = my_train_test_split(df, 'top_25')

    #train.shape,validate.shape,test.shape


#-----------------------------------------------------------------------------------------------------------
#scale data
from sklearn.preprocessing import MinMaxScaler

#Write function to scale data for the data
def scale_data(train, validate, test, features_to_scale):
    """Scales the 3 data splits using MinMax Scaler. 
    Takes in train, validate, and test data splits as well as a list of the features to scale. 
    Returns dataframe with scaled counterparts on as columns"""
    
    
    # Make the thing to train data only
    scaler = MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Fit the thing with new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled


#-----------------------------------------------------------------------------------------------------------
#modeling
def getting_(train_scaled,validate_scaled,test_scaled):
    '''
    This function takes in train and defines x features to y target into train, validate and test
    '''
    #X will be features
    #y will be our target variable
    #these features have high correlation to top_25 videos

    scaled_features = ['age_scaled', 'num_of_tags_scaled','duration_scaled', 'num_of_tags_scaled',
        'engagement_scaled', 'sponsored_scaled', 'title_in_description', 'title_in_tags',
        'pct_tags_in_description', 'title_lengths', 'desc_lengths',
        'tags_length']
    X_train = train_scaled[scaled_features]
    y_train = train_scaled['top_25']
    X_validate = validate_scaled[scaled_features]
    y_validate = validate_scaled['top_25']
    X_test = test_scaled[scaled_features]
    y_test= test_scaled['top_25']

    return X_train, y_train, X_validate, y_validate, X_test, y_test



#-----------------------------------------------------------------------------------------------------------
#models

#Decision Tree Classifier
def run_decision_tree_models(X_train, y_train, X_validate, y_validate):
    """
    Run models with decision tree classifier with varying max_depth
    """

    #loop the model with changing max depth only
    model_scores = []

    for i in range(1,15):
        model = DecisionTreeClassifier(max_depth=i, random_state =123)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        accuracy_train = model.score(X_train,y_train)
        accuracy_validate = model.score(X_validate,y_validate)
        difference = accuracy_train-accuracy_validate
        output = {"i":i, "accuracy_train":accuracy_train,"accuracy_validate":accuracy_validate,"difference":difference}
        model_scores.append(output)
    df = pd.DataFrame(model_scores)
    
    return df
#-----------------------------------------------------------------------------------------------------------
#Random Forest

def run_random_forest_models(X_train, y_train, X_validate, y_validate):
    """
    Run models with decision tree classifier varying depth, max leaf size, criterion
    """

    model_scores = []

    for i in range(1,12):

        model = RandomForestClassifier(max_depth = i,random_state=123)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        accuracy_train = model.score(X_train,y_train)
        accuracy_validate = model.score(X_validate,y_validate)
        difference = accuracy_train-accuracy_validate
        output = {"max_depth":i, "accuracy_train":accuracy_train,"accuracy_validate":accuracy_validate,"difference":difference}
        model_scores.append(output)
    df = pd.DataFrame(model_scores)

    return df

#-----------------------------------------------------------------------------------------------------------
#KNN Classifier

def run_kneighbors_models(X_train, y_train, X_validate, y_validate):
    """
    Run models with knn classifier varying depth, max leaf size, criterion
    """

    #For loop for KNN 
    empty_model = []
    for k in range(1,10):
        model = KNeighborsClassifier(n_neighbors = k, weights = "uniform")
        model=model.fit(X_train,y_train)
        y_pred = model.predict(X_train)
        accuracy_train = model.score(X_train,y_train)
        accuracy_validate = model.score(X_validate,y_validate)
        difference = accuracy_train-accuracy_validate
        output = {"k":k, "accuracy_train":accuracy_train,"accuracy_validate":accuracy_validate,"difference":difference}
        
        
        empty_model.append(output)

    df = pd.DataFrame(empty_model)

    return df



#-----------------------------------------------------------------------------------------------------------
#Logistic Regression

def run_logistic_reg_models(X_train, y_train, X_validate, y_validate):
    """
    Run logistic models on data varying solver and C value
    """
    model = LogisticRegression(C = .1, random_state=123)
    model=model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    accuracy_train = model.score(X_train,y_train)
    accuracy_validate = model.score(X_validate,y_validate)
    difference = accuracy_train-accuracy_validate
    output = { "accuracy_train":accuracy_train,"accuracy_validate":accuracy_validate,"difference":difference}
    
    return output  


#-----------------------------------------------------------------------------------------------------------
#Test Model
def run__on_test(X_train, y_train, X_test, y_test):
    #create, fit, use, model information to model_features dfram
    model = DecisionTreeClassifier(max_depth=14, random_state=123)
    #features to be used

    scaled_features = ['age_scaled', 'num_of_tags_scaled','duration_scaled', 'num_of_tags_scaled',
           'engagement_scaled', 'sponsored_scaled', 'title_in_description', 'title_in_tags',
           'pct_tags_in_description', 'title_lengths', 'desc_lengths',
        'tags_length']
    #fit model
    model.fit(X_train, y_train)
    #score model to add to model description dataframe
    score = model.score(X_test, y_test).round(3)
    
    return score