import chartlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import warnings
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
This module provides helper methods to carry out linear regression
on flight data found on https://www.kaggle.com/usdot/flight-delays.

These methods are specific to the flight dataset and is not meant to be 
generic functions for other datasets.
"""
def select_kbest_reg(data_frame, target, k):
    """
    Selecting K-Best features regression.  Performs F-Test 
    :param data_frame: A pandas dataFrame with the training data
    :param target: target variable name in DataFrame
    :param k: desired number of features from the data
    :returns feature_scores: scores for each feature in the data as 
    pandas DataFrame
    """
    feat_selector = SelectKBest(f_regression, k=k)
    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])
    
    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns
    
    return feat_scores 

def LinearRegressionModelScore(data, featurelist, cond, target, testsize, bintime = True):
    df = EncodeDepartureTimeMonthDayOfWeek(data, cond,
              featurelist, -1, bintime)
    features = np.delete(df.columns.values, 0)
    features_train, features_test, target_train, target_test = train_test_split(df[features],
                df[target], 
                test_size = testsize)
    
    # fit a model
    lm = linear_model.LinearRegression()
    model = lm.fit(features_train, target_train)
    predictions = lm.predict(features_test)
    return model.score(features_test, target_test), len(df.index), len(features_test), predictions, target_test, lm

def PredictedVsActual(data, cond, features, target, testsize, bindata = True):
    score, dflen, testlen, predictions, target_test, lm = LinearRegressionModelScore(data, features, cond, target, testsize, bindata)
    df = pd.DataFrame()
    df['Actual Delay'] = target_test.values
    df['Predicted Delay'] = predictions 
    df['Diff In Delay'] = predictions - target_test.values 
    return df;

def PlotPredictedVsActual(df, ymax):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    ax1 = axes[0]
    ax2 = ax1.twinx()  # set up the 2nd axis

    ax1.set_ylim([0, ymax])
    ax2.set_ylim([0, ymax])

    df['Actual Delay'].hist(ax=ax1)
    df['Predicted Delay'].hist(ax=ax2, color='red', alpha = 0.5)
    sns.distplot(df['Diff In Delay'], ax = axes[1]);
    sns.boxplot(df['Diff In Delay'], ax = axes[2]);

def EncodeDepartureTimeMonthDayOfWeek(data, cond, featureList, sampleSize, binMonthAndTime=True):
    newFeatureList = list(featureList)
    df = data.loc[cond];
    #filterDF = df.groupby(groupattr, group_keys=False).apply(lambda x: x.sample(min(len(x), sampleSize)))
    if (sampleSize > 0):
        maxSize = len(df)
        if sampleSize > maxSize:
            filterDF = df
        else:
            filterDF = df.sample(n=sampleSize)
    else:
        filterDF = df
    if (binMonthAndTime == False):
        return filterDF[newFeatureList]
    one_hot = pd.get_dummies(filterDF['DEPARTURE_TIME_BIN'], prefix='DEPARTURE_TIME_HOUR')
    df = filterDF.join(one_hot)
    newFeatureList.extend(one_hot) 
    
    one_hot = pd.get_dummies(df['PRCP_BIN'], prefix='PRCP_D')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 
    
    one_hot = pd.get_dummies(df['WDF2_BIN'], prefix='WDF2_D')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 

    one_hot = pd.get_dummies(df['TAVG_BIN'], prefix='TAVG_D')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 
    
    one_hot = pd.get_dummies(df['MONTH'], prefix='MONTH')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 

    one_hot = pd.get_dummies(df['DAY_OF_WEEK'], prefix='DAY_OF_WEEK')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 

    if 'AIRLINE' in df.columns:
        one_hot = pd.get_dummies(df['AIRLINE'], prefix='AIRLINE')
        df = df.join(one_hot)
        newFeatureList.extend(one_hot) 
    
    return df[newFeatureList]
    
def GetKBestFeatureList(data, attribute, cond, featureList, sampleSize, k, filterlist = []):
    newdf = EncodeDepartureTimeMonthDayOfWeek(data, cond, featureList, sampleSize)
    if filterlist:
        newdf=newdf[filterlist];
    return select_kbest_reg(newdf, attribute, k)

def PlotFeatures(featureList):
    attribute_fscore = featureList[['Attribute','F Score']]
    df = attribute_fscore.set_index('Attribute')
    df = df.sort_values('F Score')
    ax  = df.plot.bar(figsize=(14, 6))

def GetFeatureListsDFList(data, attr, cond, sample_size, count, features, k, filterlist = []):
    first = 0
    oldsize = 0
    for x in range(0, count):
        df = GetKBestFeatureList(data, attr, cond,
                  features, sample_size, k, filterlist)
        if (first == 1):
            olddf = pd.merge(olddf, df, on='Attribute', suffixes=('_'+str(oldsize),'_'+str(oldsize+1)))
        else:
            olddf = df
        oldsize = oldsize+1
        first = 1
    return olddf

def PlotFeatureLists(data, count, figwidth= 15, figheight = 6, attribute='F Score'):
    df = data.set_index(['Attribute'])
    fig = plt.figure(figsize=(figwidth,figheight))
    ax = fig.add_subplot(111)
    p = 0
    c = ['red','green','blue','yellow','black','red','green','blue','yellow','black']
    for x in range(0, count):
        df[attribugte+'_'+ str(x+1)].plot(kind='bar', color=c[p], ax=ax, position=p, width=0.25)
        p = p+1
        
    ax.set_ylabel = ('Sample')
    plt.show()
    
def PlotFeatureList(data, figwidth= 15, figheight = 6, attribute='F Score'):
    df = data.set_index(['Attribute'])
    fig = plt.figure(figsize=(figwidth,figheight))
    df = df.sort_values(by=attribute, ascending=False)
    df[attribute].plot(kind='bar', width=0.5)
    plt.show()   

def AnalyzeSampleSize(data, attribute, attr, initcount, increment, condition, features, k, filterlist=[]):
    datasetSize = len(data.loc[condition]);
    sample_size_list=np.arange(initcount+increment,datasetSize,increment)
    a = pd.DataFrame();
    initialdf = GetFeatureListsDFList(data, attr, condition, initcount, 1, features, k, filterlist)
    a[str(initcount)] = initialdf[attribute]
    for sample_size in sample_size_list:
        b = GetFeatureListsDFList(data, attr, condition, sample_size, 1, features, k, filterlist)
        a[str(sample_size)] = b[attribute]
    
    aT= a.transpose()
    aT.columns = initialdf['Attribute'].values
    return aT

def PerformLinearRegression(data, features, conds, condslbl, target, testsize, bin_data = True):
    df = pd.DataFrame(columns=['Condition' , 'Score'])
    popDF = pd.DataFrame(columns=['# of Data' , '# of Test Data'])
    meanSquareDF = pd.DataFrame(columns=['Condition' , 'MSE'])
    for cond, lbl in zip(conds, condslbl):
        score, dflen, testlen, predictions, target_test, lm = LinearRegressionModelScore(data, features, cond, target, testsize, bin_data)
        df = df.append({'Condition': lbl, 'Score': score}, ignore_index=True)
        popDF = popDF.append({'# of Data': dflen, '# of Test Data': testlen}, ignore_index=True)
        meanSquareDF = meanSquareDF.append({'Condition': lbl, 'MSE': mean_squared_error(target_test, predictions)}, ignore_index=True)
        
    return df, popDF, meanSquareDF;

def PerformNumberOfLinearRegression(data, features, count, conds, condslbl, target, testsize, bin_data):
    for x in range(0, count):
        df, popDF, meanSquareDF = PerformLinearRegression(data, features, conds, condslbl, target, testsize, bin_data)
        if x > 0:
            olddf = pd.merge(olddf, df, on='Condition', suffixes=('_'+str(x),'_'+str(x+1)));
            oldfMSEDF = pd.merge(oldfMSEDF, meanSquareDF, on='Condition', suffixes=('_'+str(x),'_'+str(x+1)));
        else:
            olddf = df;
            oldfMSEDF = meanSquareDF;
    MSEaT = oldfMSEDF.transpose()
    MSEaT.columns = oldfMSEDF['Condition'].values
    MSEaT[1:]
    aT = olddf.transpose()
    aT.columns = olddf['Condition'].values
    aT[1:]
    overalldf = pd.DataFrame();
    overalldf['# of Data'] = popDF['# of Data']
    overalldf['# of Test Data'] = popDF['# of Test Data']
    overalldf['MSE Mean'] = MSEaT[1:].mean().values
    overalldf['MSE Std'] = MSEaT[1:].std().values
    overalldf['R-Squared Mean'] = aT[1:].mean().values
    overalldf['R-Squared Std'] = aT[1:].std().values
    overalldf['Condition'] = aT[1:].mean().index
    return overalldf;
