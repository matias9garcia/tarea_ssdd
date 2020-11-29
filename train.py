import pandas as pd
import numpy as np

inputTrain = pd.read_csv("train.txt", header=None)
etiquetaTrain = inputTrain[41]
inputTrain = inputTrain.drop([41],axis=1)
numpyInputTrain = inputTrain.to_numpy() ##
N = np.size(numpyInputTrain,axis=0)
print(etiquetaTrain)
print(numpyInputTrain)
print(N)
