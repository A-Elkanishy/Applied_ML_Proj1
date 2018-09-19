# Applied_ML_Proj1
The project includes 6 different modules besides 3 running scripts
1-"Perceptron.py" it is the implementation of the 2 classes perceptron classifier
2-"Adaline.py" it is the implementation of the 2 classes Adaline classifier
3-"AdalineSGD.py" it is the implementation of the 2 classes stochastic gradient descent classifier
4-"PerceptronMC.py" is the implementation of multiple classes perceptron classifier 
5-"SGDMC.py" is the implementation of multiple classes perceptron classifier 
6-"PDR.py" is used to plot the decision regions 

There are 3 running scripts:
1- "main.py" is used to test the binary classes classifiers modules such as "Perceptron.py", "Adaline.py", and "AdalineSGD.py"
  Here are the commands to run the code:
      1. python main.py perceptron iris.data
      2. python main.py adaline iris.data
      3. python main.py sgd iris.data
      4. python main.py perceptron wine.data
      5. python main.py adaline wine.data
      6. python main.py sgd <dataset2>
2- "PMC_TestScript.py" is used to test muiltiple classes perceptron classifier in "PerceptronMC.py" file.
  Here are the commands to run the code
      1. python PMC_TestScript.py PMC iris.data
      2. python PMC_TestScript.py PMC wine.data
3- "SGDMC_Test.py" is used to test muiltiple classes perceptron classifier in "SGDMC.py" file.
  Here are the commands to run the code
      1. python SGDMC_Test.py SGDMC iris.data
      2. python SGDMC_Test.py SGDMC wine.data
All the running scripts, at the beginning of the execution, the script askes for dataset parameters such as, number of features, 
number of instance in each class, and the position of the classes labels.
