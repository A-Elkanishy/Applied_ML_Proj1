# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:38:34 2018

@author: A.Elkanishy
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from Adaline import AdalineGD as Adaline
from AdalineSGD import AdalineSGD
from PDR import plot_decision_regions

df = pd.read_csv(sys.argv[2], header=None)
df.tail()

n_classes = int(input("What is the number of classes?  "))
c_classes = int(input("Where is class column in the dataset?  "))
N_I_C1 = int(input("Number of instance in the first class?  "))
N_I_C2 = int(input("Number of instance in the second class?  "))
n_features = int(input("What is the number of features?  "))
c_features = int(input("Where is the features columns starts in the dataset begins?  "))

y = df.iloc[0:N_I_C1+N_I_C2, c_classes].values
y[0:N_I_C1]=1
y[N_I_C1:N_I_C1+N_I_C2]=-1

X = df.iloc[0:N_I_C1+N_I_C2, [c_features, n_features]].values
plt.scatter(X[:N_I_C1, 0], X[:N_I_C1, 1], color='red', marker='o', label='Class 1')
plt.scatter(X[N_I_C1:N_I_C1+N_I_C2, 0], X[N_I_C1:N_I_C1+N_I_C2, 1], color='blue', 
            marker='x', label='Class 2')
plt.title('Events Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

n_iter = int(input("What is the number of iterations?  "))
eta = float(input("What is the training factor? (0.1 -> 1.0)  "))

def shuffle(X, y):
    r = np.random.permutation(len(y))
    return X[r], y[r]

#25% holdout
H1=round(N_I_C1*0.25)
H2=round(N_I_C2*0.25)

X[0:N_I_C1], y [0:N_I_C1]= shuffle(X[0:N_I_C1], y[0:N_I_C1])
X[N_I_C1:N_I_C1+N_I_C2], y[N_I_C1:N_I_C1+N_I_C2] = shuffle(X[N_I_C1:N_I_C1+N_I_C2], y[N_I_C1:N_I_C1+N_I_C2])

yn=np.copy(y[H1:N_I_C1])+np.copy(y[N_I_C1+H2:N_I_C1+N_I_C2])
Xn=np.copy(X[H1:N_I_C1])+np.copy(X[N_I_C1+H2:N_I_C1+N_I_C2])


X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

if sys.argv[1] == 'perceptron' or sys.argv[1]=='Perceptron':  
    classifier = Perceptron(eta, n_iter)
    classifier.fit(Xn, yn)    
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Number of misclassifications')
    plt.show()
    
if sys.argv[1] == 'adaline' or sys.argv[1]=='Adaline':
    X = X_std
    classifier = Adaline(eta, n_iter).fit(X, y)
    plt.plot(range(1, len(classifier.cost_) + 1), np.log10(classifier.cost_), marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Sum-squared-error')
    plt.show()

    
if sys.argv[1] == 'sgd'or sys.argv[1]=='SGD':
    random_state = int(input("What is the random state?  "))
    X = X_std
    classifier = AdalineSGD( eta, n_iter, random_state)
    classifier.fit(X, y)
    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Average Cost')
    plt.show()
    
plot_decision_regions(X, y, classifier)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

#25% holdout validation
IP=0
for i in range(0,H1):
    if classifier.predict(X[i])==-1 :
        IP+=1
for i in range(0,H2):
    if classifier.predict(X[N_I_C1+i])==1 :
        IP+=1

print(" Accuracy is "+ str(IP/(H1+H2)*100))

'''
from Adaline import AdalineGD

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(eta, n_itre)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()



ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
'''









         
         
         
         
         
         
         
         
         
         
         