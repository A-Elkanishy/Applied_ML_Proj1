# -*- coding: utf-8 -*-

import numpy as np 
from Perceptron import Perceptron as P2C
import matplotlib.pyplot as plt
from PDR import plot_decision_regions as PDR

class PerceptronMC(object):
         def __init__(self, eta, n_iter, n_classes, n_inst):
             self.eta = eta
             self.n_iter = n_iter
             self.n_classes = n_classes 
             self.n_inst=n_inst
             self.sub_classifiers=[P2C() for i in range(n_classes)]             
         def fit(self, X,y):
             SI=0
             for i in range(0,self.n_classes):
                 y[0:sum(self.n_inst)]=0
                 y[SI:sum(self.n_inst[0:i+1])]=1
                 y = np.where(y == 0, -1, 1)
                 self.sub_classifiers[i] = P2C(self.eta, self.n_iter)
                 self.sub_classifiers[i].fit(X, y) 
                 SI+=self.n_inst[i]
                 plt.plot(range(1, len(self.sub_classifiers[i].errors_) + 1),self.sub_classifiers[i].errors_, marker='o')
                 plt.xlabel('Number of iterations')
                 plt.ylabel('Number of misclassifications')
                 plt.show()
                 PDR(X, y, self.sub_classifiers[i])
                 plt.xlabel('Feature 1')
                 plt.ylabel('Feature 2')
                 plt.legend(loc='upper left')
                 plt.show()
             return self
         def predict(self, X):
             r=[0]*self.n_classes
             for i in range(0,self.n_classes):
                 r[i]=int(self.sub_classifiers[i].predict(X))
             return r
         def predictNI(self, X):
             r=[0.0]*self.n_classes
             for i in range(0,self.n_classes):
                 r[i]=float(self.sub_classifiers[i].net_input(X))
             return r
         def predictC(self, X):
             r=[0.0]*self.n_classes
             for i in range(0,self.n_classes):
                 r[i]=float(self.sub_classifiers[i].net_input(X))
             return "Class " + str(r.index(max(r))+1)

 








           
            
            
            
            
