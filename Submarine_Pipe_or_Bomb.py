#######################################################################################
# Kevin Tyler
# Machine Learning - Sonar Data
# Description: Applying machine learning algorithms to sonar data set to determine if a rock or a mine
#######################################################################################

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Scaling can be helpful for the machine learning technique, though not always
from sklearn.metrics import confusion_matrix # Used to see true positives, true negatives, false positives, and false negatives

#####################################
# Read File, Scaling, and PCA
#####################################

# Reading in sonar_all_data_2.csv
sonar = pd.read_csv('sonar_all_data_2.csv', header=None)
sonar.columns = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12',
                 'T13','T14','T15','T16','T17','T18','T19','T20','T21','T22',
                 'T23','T24','T25','T26','T27','T28','T29','T30','T31','T32',
                 'T33','T34','T35','T36','T37','T38','T39','T40','T41','T42',
                 'T43','T44','T45','T46','T47','T48','T49','T50','T51','T52',
                 'T53','T54','T55','T56','T57','T58','T59','T60','Classification #','Classification Mine/Rock',] # Defining columns

X = sonar.iloc[:,0:60].values # Defines X as the 60 columns of data in sonar_all_data_2.csv
y = sonar.iloc[:,60].values # Defines y as the column that indicates with 1 or a 2 whether it is a rock or a mine

# split the problem into train and test, so that the models can train on a smaller set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0) # Uses 70% of the data for the training set, 30% for the test set

# scales X and centers the data on the axis
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Sets the principal component analysis to reduce to 6 independent variables
pca = PCA(n_components=6)

# Transforms the scaled x_train_std and x_test_std data into principal components
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#####################################
# Perceptron
#####################################

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=4, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False) # Verbose turned to false to make end data easier to read
ppn.fit(X_train_pca, y_train) # Implements the perceptron ML algorithm on the training data and learns from it

print('\n')
print('Perceptron Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = ppn.predict(X_test_pca) # Uses perceptron algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% test subset
Accuracy_ppn = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = ppn.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier =ppn)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("Perceptron PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("Perceptron Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Logistic Regression
#####################################

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_pca, y_train) # Implements the logistic regression ML algorithm on the training data and learns from it


print('Logistic Regression Results')
print('Number in test ',len(y_test)) # prints number in test for user friendliness
y_pred = lr.predict(X_test_pca) # Uses linear regression algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # Gives accuracy score of the 30% subset
Accuracy_lr = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = lr.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier=lr)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("Logistic Regression PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("Logistic Regression Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Support Vector Machine (Linear)
#####################################

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_pca, y_train) # Implements the svm linear ML algorithm on the training data and learns from it

print('\n')
print('Support Vector Machine (Linear) Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = svm.predict(X_test_pca) # Uses svm linear algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% subset
Accuracy_svm_linear = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = svm.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier = svm)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("SVM Linear PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("SVM Linear Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Support Vector Machine (rbf)
#####################################

from sklearn.svm import SVC
svmrbf = SVC(kernel='rbf', random_state=0, gamma=0.2 , C=10.0) # gamma = 0.2 and C = 10 common to reduce under and over training
svmrbf.fit(X_train_pca, y_train) # Implements the svm rbf ML algorithm on the training data and learns from it

print('Support Vector Machine (rbf) Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = svmrbf.predict(X_test_pca) # Uses svm rbf algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% subset
Accuracy_svm_rbf = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = svmrbf.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier=svmrbf)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("SVM rbf PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("SVM rbf Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Decision Tree Learning
#####################################

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3 ,random_state=0)
tree.fit(X_train_pca,y_train) # Implements the decision tree learning ML algorithm on the training data and learns from it

print('Decision Tree Learning Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = tree.predict(X_test_pca) # Uses decision tree algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% subset
Accuracy_dtl = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = tree.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier=tree)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("Decision Tree Learning PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("Decision Tree Learning Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Decision Tree Learning - Random Forest
#####################################

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10 ,random_state=1, n_jobs=2)
forest.fit(X_train_pca,y_train) # Implements the random forest ML algorithm on the training data and learns from it

print('Decision Tree Learning (Random Forest) Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = forest.predict(X_test_pca) # Uses random forest algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% subset
Accuracy_random_forest = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = forest.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier=forest)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("Decision Tree Learning (Random Forest) PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("Decision Tree Learning (Random Forest) Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# K-Nearest Neighbor
#####################################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train_pca,y_train) # Implements the k-nearest neighbors ML algorithm on the training data and learns from it

print('K-Nearest Neighbor Results')
print('Number in test ',len(y_test)) # Prints number in test for user friendliness
y_pred = knn.predict(X_test_pca) # Uses k-nearest neighbor algorithm learned from training set to predict the y value in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum()) # Prints how many samples were misclassified in test set

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # Gives accuracy score of the 30% subset
Accuracy_knn = round(accuracy_score(y_test, y_pred), 2) # Documents accuracy for final accuracy summary

X_combined_pca = np.vstack((X_train_pca, X_test_pca)) # Combines train and test data to reform total data set
y_combined = np.hstack((y_train, y_test)) # Combines train and test data to reform total data set
print('Number in combined ',len(y_combined)) # Prints the amount of data in the "combined" data set (should be total amount of data)

y_combined_pred = knn.predict(X_combined_pca) # Uses algorithm leanred from training set to predict y value on the combined set
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum()) # Prints how many samples were misclassied in combined set

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred)) # Gives accuracy score for total data set
print('\n')

# This code below can be used to visualize data and ML solution when PCA n= 2
#import pml53
#plot_decision_regions(X_train_pca, y_train, classifier=knn)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#print("K-Nearest Neighbor PCA Graph: ")
#plt.show()

# Produces confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Makes a nice plot of confusion matrix that is easier to read
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i,j],
                va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
print("K-Nearest Neighbor Confusion Matrix: ")
plt.show()
print('\n')

#####################################
# Accuracy Summary Print-out
#####################################

print("Accuracy Summary:")
print("Perceptron:", Accuracy_ppn)
print("Logistic Regression:", Accuracy_lr)
print("SVM Linear:", Accuracy_svm_linear)
print("SVM rbf:", Accuracy_svm_rbf)
print("Decision Tree:", Accuracy_dtl)
print("Random Forest:", Accuracy_random_forest)
print("K-Nearest Neighbor:", Accuracy_knn)