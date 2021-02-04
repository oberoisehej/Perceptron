##############
# Name: Sehej Oberoi
# email: oberois@purdue.edu
# Date: 11/12/2020

import numpy as np
import sys
import os
import math
import random


def getWeights(data, labels, maxIterations, validationData, validationLabels):
  w = [0]*len(data.columns)
  b = 0
  sinceUpdate = 0
  bestCorrect = -1
  bestWeights = None
  sumCorrect = 0
  for i in range(0, maxIterations):
    allCorrect = True
    hingeLoss = 0
    correct = 0
    sumW = [0]*len(data.columns)

    for j in range(0, len(data)):
      v = val(w, b, data.iloc[j].tolist())
      s = sign(v)
      actual = int(labels.iloc[j]) * 2 - 1    # this will map 0s to -1 and 1s to 1

      if s != actual:
        w = updateW(w, actual, data.iloc[j].tolist())
        b += actual
        allCorrect = False
      sumW = addW(sumW, w)

    w = avgW(sumW, len(data))

    for j in range(0, len(validationSet)):
      v = val(w, b, validationData.iloc[j].tolist())
      s = sign(v)
      actual = int(validationLabels.iloc[j]) * 2 - 1

      if s != actual:
        hingeLoss += math.sqrt((v - actual) ** 2)
      else:
        correct += 1

    if correct > bestCorrect:
      bestCorrect = correct
      bestWeights = (w, b)
      sinceUpdate = 0

    if allCorrect:
      break

  return bestWeights

def addW(sumW, w):
  for i in range(0, len(w)):
    sumW[i] += w[i]
  return sumW

def avgW(sumW, n):
  for i in range(0, len(sumW)):
    sumW[i] /= n
  return sumW

def val(w, b, row):
  total = b
  for i in range(0, len(row)):
    total += w[i] * row[i]

  return total

def sign(val):
  if val >= 0:
    return 1
  return -1

def updateW(w, actual, row):
  for i in range(0, len(w)):
    w[i] += (actual * row[i])

  return w

if __name__ == "__main__":
    # parse arguments
    import argparse
    import pandas as pd

    testDataFile = ''
    testLabelFile = ''
    trainDataFile = ''
    trainLabelFile = ''

    for arg in sys.argv[1:]:
      if 'test' in arg:
        if '.label' in arg:
          testLabelFile = arg
        elif '.data' in arg:
          testDataFile = arg
      if 'train' in arg:
        if '.label' in arg:
          trainLabelFile = arg
        elif '.data' in arg:
          trainDataFile = arg

    if testDataFile == '' or testLabelFile == '' or trainDataFile == '' or trainLabelFile == '':
      print("Not all arguments provided")
      exit

    trainData = pd.read_csv(trainDataFile, delimiter=',', index_col=None, engine='python')
    trainLabel = pd.read_csv(trainLabelFile, delimiter=',', index_col=None, engine='python')

    trainData = trainData.join(trainLabel)
    trainData[trainData.columns] = trainData[trainData.columns].apply(pd.to_numeric, errors='coerce')
    trainData = trainData.fillna(trainData.median())

    weightsList = list()
    k = 5
    for i in range(0, k):
      validationSet = trainData.sample(frac = 1/(1.0 * k))
      validationSet.reset_index(inplace=True, drop=True)
      trainSet = pd.concat([trainData, validationSet]).drop_duplicates(keep=False)
      trainSet.reset_index(inplace=True, drop=True)

      weights = getWeights(trainSet.iloc[:,:-1], trainSet.iloc[:,-1:], 200, validationSet.iloc[:,:-1], validationSet.iloc[:,-1:])
      weightsList.append(weights)

    testData = pd.read_csv(testDataFile, delimiter=',', index_col=None, engine='python')
    testLabel = pd.read_csv(testLabelFile, delimiter=',', index_col=None, engine='python')

    hingeLoss = 0
    correct = 0
    for i in range(0, len(testData)):
      predict = 0
      for j in range(0, len(weightsList)):
        w = weightsList[j][0]
        b = weightsList[j][1]

        v = val(w, b, testData.iloc[i].tolist())
        predict += sign(v)


      if predict >= 0:
        s = -1
      else:
        s = 1
      actual = int(testLabel.iloc[i]) * 2 - 1

      if s != actual:
        hingeLoss += math.sqrt((v - actual) ** 2)
      else:
        correct += 1


    print("Hinge LOSS=%.2f" % (hingeLoss/len(testData)))
    print("Test Accuracy=%.4f" %(correct/len(testData)))
