#######################################################################################
# Kevin Tyler
# Machine Learning - Pipe or Bomb
# Description: Data anaylysis of submarine data to determine orthogonal variable reduced data set
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm as cm
from sklearn.decomposition import PCA
import seaborn as sns
## create covariance for dataframes
def mosthighlycorrelated(mydataframe, numtoreport):  # Creates the most highly correlated dataframe
# find the correlations 
    cormatrix = mydataframe.corr() 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T # Will set diagonal correlations or lower triangle to 0
# find the top n correlations 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] # Based on data set read in will rename to easier names
    return cormatrix.head(numtoreport)
## Covariance matrix
def correl_matrix(X,cols):
    fig = plt.figure(figsize=(7,7), dpi=100) 
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)
    # creates the major ticks on correlation matrix
    major_ticks = np.arange(0,len(cols),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    # makes graph titles and makes it easier to read
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=12)
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)
## make pair plots
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')
    cols = df.columns
    sns.pairplot(df[cols],size=2.5)
    plt.show()
    
## this creates a dataframe similar to a dictionary
## a data frame can be constructed from a dictionary
sonar = pd.read_csv('sonar_all_data_2.csv')
print('first 5 observations',sonar.head(5))
cols = sonar.columns
X = sonar.iloc[:,0:59].values # Will include everything up to column 13 in X
Y = sonar.iloc[:,60].values # Will include column 13 for Y
## Identify Null values
print(' Identify Null Values ')
print( sonar.apply(lambda x: sum(x.isnull()),axis=0) ) # Finds if there are any null values in the data set
## 'setosa' 0
## 'versicolor' 1
## 'virginica' 2
##  descriptive statistics
print(' Descriptive Statistics ')
print(sonar.describe())
## most highly correlated lists
print("Most Highly Correlated")
print(mosthighlycorrelated(sonar,25)) # Printing out the top 25 correlations to observe, though there are many more
## heat plot of covariance
print(' Covariance Matrix ')
correl_matrix(sonar.iloc[:,0:14],cols[0:14]) # Makes the correlation matrix cover all 13 input variables and the heart disease output
## Pair plotting
print(' Pair plotting ')
pairplotting(sonar) # calls the pairplotting function defined above on the data set
"""
"""