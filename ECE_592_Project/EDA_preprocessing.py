'''
Created on 06-Dec-2022

@author: EZIGO
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import Data, data_prediction, data_forecast

''' Exploratory Data Analysis '''
Data_dir = Data
data = data_prediction
data_path = Data_dir+data
df=pd.read_csv(data_path)
print(df.head(4))
print("Empty values in dataset: "+str(df.isnull().values.any()))
description=df.describe()
print(description)
description.to_csv("Description_"+data)
if data == data_prediction:
    target_column="Bankrupt?"
elif data == data_forecast:
    target_column="class"
Not_Bankrupt=list(df[target_column]).count(0)
Bankrupt=list(df[target_column]).count(1)
plt.figure()
plt.title("Distribution of classes")
sns.barplot(x=["Bankrupt","Not Bankrupt"], y=[Bankrupt,Not_Bankrupt])
plt.savefig("Distribution of classes"+data[:data.index(".")]+".png")
# plt.figure()
# sns.pairplot(df,hue='Bankrupt')
# plt.show()
# plt.savefig("pairplot")

'''Data Preprocessing'''
# Z-score standardization
colums_to_standardise=list(description.loc[:,description.drop(labels='count', axis=0).gt(1).any()].columns)
df[colums_to_standardise] = StandardScaler().fit_transform(df[colums_to_standardise])

'''Feature Importance analysis'''
X = df.drop(target_column,axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=500,random_state=123,criterion='entropy')
rf.fit(X_train,y_train)
feat_scores = pd.Series(rf.feature_importances_,index=X.columns)
feat_scores = feat_scores.sort_values(ascending=False)
plt.figure()
plt.title("Average Entropy Importance of features"+data[:data.index(".")])
mplt.rc('ytick', labelsize=1) 
feat_scores.plot.barh()
plt.show()

'''drop features of lowest importance'''
# lowest_features = feat_scores[feat_scores<=0.001].index[:]
# print(lowest_features)
# df= df.drop(lowest_features, axis=1)

df.to_csv(Data_dir+"preprocessed_"+data,index=False)