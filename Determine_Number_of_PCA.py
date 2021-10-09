# -*- coding: utf-8 -*-
"""
Kevin
PCA Analysis #
"""

import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA
## access the dataset
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np
sonar = pd.read_csv('sonar_all_data_2.csv')

scaler = MinMaxScaler(feature_range=[0,1])
data_rescaled = scaler.fit_transform(sonar.values[:,0:59])

#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Sonar Dataset Explained Variance')
plt.show()