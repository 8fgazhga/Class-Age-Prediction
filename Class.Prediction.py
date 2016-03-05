import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('DataForFitting.csv', header=0)

# determine how to fill in missing values
from scipy.stats import mode
# Number of missing values is replaced with mean value

df.NewsArticles=df.NewsArticles.fillna(np.mean(df.NewsArticles))
df.FirstDates=df.FirstDates.fillna(np.mean(df.FirstDates))
df.Weddings=df.Weddings.fillna(np.mean(df.Weddings))
df.CookDinner=df.CookDinner.fillna(np.mean(df.CookDinner))
df.ServingsofAlcohol=df.ServingsofAlcohol.fillna(np.mean(df.ServingsofAlcohol))
df.HoursofSleep=df.HoursofSleep.fillna(np.mean(df.HoursofSleep))
df.CupsofCoffee=df.CupsofCoffee.fillna(np.mean(df.CupsofCoffee))

# make the missing shoe sizes the average of each size at that height
df.ShoeSize=df.ShoeSize.map(lambda x: np.nan if x==0 else x)
shoemeans=df.pivot_table('ShoeSize', columns='Height', aggfunc='mean')
df.ShoeSize=df[['ShoeSize', 'Height']].apply(lambda x: shoemeans[x['Height']] if pd.isnull(x['ShoeSize']) else x['ShoeSize'], axis=1)
# make the missing HoursofSleep the average number of courses
df.HoursofSleep=df.HoursofSleep.map(lambda x: np.nan if x==0 else x)
sleepmeans=df.pivot_table('HoursofSleep', columns='CourseCredits', aggfunc='mean')
df.HoursofSleep=df[['HoursofSleep', 'CourseCredits']].apply(lambda x: shoemeans[x['CourseCredits']] if pd.isnull(x['HoursofSleep']) else x['HoursofSleep'], axis=1)

# now only get the objects (strings) from dataframe
df.dtypes[df.dtypes.map(lambda x: x=='object')]

# Make categorical data into integers
df['CarTry'] = df['OwnACar'].map( {'No': 0, 'Yes': 1} ).astype(int)
df['VoteTry'] = df['Voted'].map( {'No': 0, 'Yes': 1} ).astype(int)
df['CreditTry'] = df['CreditScoreChecked'].map( {'No': 0, 'Yes': 1} ).astype(int)
df['FluTry']= df['FluVaccine'].map( {'No': 0, 'Yes': 1} ).astype(int)

df=df.join(pd.get_dummies(df['CommuteMethod'], prefix='Com'))
df=df.join(pd.get_dummies(df['PastJob'], prefix='Job'))
df=df.join(pd.get_dummies(df['Laptop'], prefix='Comp'))
df=df.join(pd.get_dummies(df['CatorDog'], prefix='Pet'))
df=df.join(pd.get_dummies(df['Field'], prefix='Dept'))

# Drop unneccessary fields
df=df.drop(['MobileOS', 'OwnACar', 'AreaCode', 'CommuteMethod', 'Voted', 'PastJob', 'CreditScoreChecked', 'FluVaccine', 'Laptop', 'MobileOS', 'CatorDog', 'Field'], axis=1)

df_array=df.values

#----NOW DATA IS ALL CLEAN----
# note: should break into train and test data before guessing NANs, but data set is small and not all categories are there in small set.

# split data into test/train data

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df_array[:,1::],df_array[:,0], test_size=0.33, random_state=42)

# Decision Tree Regression w/ AdaBoost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng=np.random.RandomState(1)

regr1=DecisionTreeRegressor(max_depth=5)
regr2=AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=500, random_state=rng)

regr1.fit(x_train, y_train)
regr2.fit(x_train, y_train)

y1=regr1.predict(x_train)
y2=regr2.predict(x_train)

from sklearn.metrics import mean_squared_error
mean_squared_error(y1, y_train)
# 0.043478260869565216
mean_squared_error(y2, y_train)
# 0.005434782608695652
# --------------------------------------------------------------------------------------------------
# Now import csv to predict
df_pred=pd.read_csv('PredictThese.csv', header=0)

df_pred.NewsArticles=df_pred.NewsArticles.fillna(np.mean(df_pred.NewsArticles))
df_pred.FirstDates=df_pred.FirstDates.fillna(np.mean(df_pred.FirstDates))
df_pred.Weddings=df_pred.Weddings.fillna(np.mean(df_pred.Weddings))
df_pred.CookDinner=df_pred.CookDinner.fillna(np.mean(df_pred.CookDinner))
df_pred.ServingsofAlcohol=df_pred.ServingsofAlcohol.fillna(np.mean(df_pred.ServingsofAlcohol))
df_pred.HoursofSleep=df_pred.HoursofSleep.fillna(np.mean(df_pred.HoursofSleep))
df_pred.CupsofCoffee=df_pred.CupsofCoffee.fillna(np.mean(df_pred.CupsofCoffee))

# make the missing shoe sizes the average of each size at that height
df_pred.ShoeSize=df_pred.ShoeSize.map(lambda x: np.nan if x==0 else x)
shoemeans=df_pred.pivot_table('ShoeSize', columns='Height', aggfunc='mean')
df.ShoeSize=df_pred[['ShoeSize', 'Height']].apply(lambda x: shoemeans[x['Height']] if pd.isnull(x['ShoeSize']) else x['ShoeSize'], axis=1)

# now only get the objects (strings) from dataframe
df_pred.dtypes[df_pred.dtypes.map(lambda x: x=='object')]

# Make categorical data into integers
df_pred['CarTry'] = df_pred['OwnACar'].map( {'No': 0, 'Yes': 1} ).astype(int)
df_pred['VoteTry'] = df_pred['Voted'].map( {'No': 0, 'Yes': 1} ).astype(int)
df_pred['CreditTry'] = df_pred['CreditScoreChecked'].map( {'No': 0, 'Yes': 1} ).astype(int)
df_pred['FluTry']= df_pred['FluVaccine'].map( {'No': 0, 'Yes': 1} ).astype(int)

df_pred=df_pred.join(pd.get_dummies(df_pred['CommuteMethod'], prefix='Com'))
df_pred=df_pred.join(pd.get_dummies(df_pred['PastJob'], prefix='Job'))
df_pred=df_pred.join(pd.get_dummies(df_pred['Laptop'], prefix='Comp'))
df_pred=df_pred.join(pd.get_dummies(df_pred['CatorDog'], prefix='Pet'))
df_pred=df_pred.join(pd.get_dummies(df_pred['Field'], prefix='Dept'))

# Drop unneccessary fields
df_pred=df_pred.drop(['MobileOS', 'OwnACar', 'AreaCode', 'CommuteMethod', 'Voted', 'PastJob', 'CreditScoreChecked', 'FluVaccine', 'Laptop', 'MobileOS', 'CatorDog', 'Field'], axis=1)

df_predarray=df_pred.values
engr=np.zeros((df_predarray.shape[0],1))
df_predarray=np.hstack((df_pred, engr))
x_pred=df_predarray[:, 1::]

# Decision Tree Regression w/ AdaBoost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

y2_pred=np.transpose(regr2.predict(x_pred))
output=np.vstack((np.transpose(df_predarray[:,0]), y2_pred))
np.savetxt('ProjOutput3.csv', output, delimiter=',')