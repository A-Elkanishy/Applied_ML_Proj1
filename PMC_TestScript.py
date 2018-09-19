# -*- coding: utf-8 -*-

import sys
import pandas as pd
from PerceptronMC import PerceptronMC as PMC


df = pd.read_csv(sys.argv[2], header=None)
df.tail()
c_classes = int(input("Where is class column in the dataset?  "))
n_classes = int(input("What is the number of classes?  "))
n_inst = [0]*n_classes
for i in range(0,n_classes):
    n_inst[i] = int(input("Number of instance in class number "+str(i+1) +" ?  "))
    
n_features = int(input("What is the number of features?  "))
c_features = int(input("Where is the features columns starts in the dataset begins?  "))
n_iter = int(input("What is the number of iterations?  "))
eta = float(input("What is the training factor? (0.1 -> 1.0)  "))


X = df.iloc[0:int(sum(n_inst)), [c_features, n_features+c_features]].values

y = df.iloc[0:int(sum(n_inst)), c_classes].values

Test= PMC(eta,n_iter,n_classes, n_inst)
Test.fit(X,y)
print (Test.predict(X[20]))
print (Test.predictNI(X[20]))
print (Test.predictC(X[20]))

print (Test.predict(X[60]))
print (Test.predictNI(X[60]))
print (Test.predictC(X[60]))

print (Test.predict(X[120]))
print (Test.predictNI(X[120]))
print (Test.predictC(X[120]))
