#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def load_data():
    with np.load("notMNIST.npz") as data:
        data, targets = data["images"], data["labels"]
        
        np.random.seed(521)
        rand_idx = np.arange(len(data))
        np.random.shuffle(rand_idx)
        
        data = data[rand_idx] / 255.0
        targets = targets[rand_idx].astype(int)
        
        train_data, train_target = data[:10000], targets[:10000]
        valid_data, valid_target = data[10000:16000], targets[10000:16000]
        test_data, test_target = data[16000:], targets[16000:]
    return train_data, valid_data, test_data, train_target, valid_target, test_target

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x,0)

def softmax(x):
    x = x - np.amax(x, axis=1, keepdims=True)
    return np.exp(x)/(np.sum(np.exp(x), axis=1, keepdims=True))


def computeLayer(X, W, b):
    return np.matmul(X,W)+b


def CE(target, prediction):
    return (-1/target.shape[0])*np.sum(target*np.log(prediction))


def gradCE(target, prediction):
    return (softmax(prediction) - target)/target.shape[0]

def backprop(xi, xh, w, target, prediction):
  gradce = gradCE(target, prediction)
  dwo = np.matmul(np.transpose(xh),gradce) #10000,1000  100000,10
  dbo = np.transpose(sum(gradce)).reshape(1, 10)
  dwh = np.matmul(np.transpose(xi),np.where(xh > 0, 1, 0)*np.matmul(gradce,np.transpose(w)))
  dbh = sum(np.where(xh > 0, 1, 0) * np.dot(gradce, np.transpose(w))).reshape(1, 1000)
  return dwo,dbo,dwh,dbh

def learning():
  trainData, validData, testData, trainTarget, validTarget, testTarget = load_data()
  trainData = trainData.reshape((trainData.shape[0], -1))
  validData = validData.reshape((validData.shape[0], -1))
  testData = testData.reshape((testData.shape[0], -1))
  newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

  epoch=200
  H=1000
  F=trainData.shape[1]
  gamma=0.9
  alpha=0.1
  xi = trainData
  wo = np.random.normal(0, np.sqrt(2/(H+10)), (H,10))
  wh = np.random.normal(0, np.sqrt(2/(F+H)), (F,H))
  bo = np.zeros((1,10))
  bh = np.zeros((1,H))
  train_loss = []
  valid_loss = []
  train_acc = []
  valid_acc = []
  test_acc = []

  dwh = np.zeros((F,H))
  dwo = np.zeros((H,10))
  dbh = np.zeros((1,H))
  dbo = np.zeros((1,10))

  vwh = np.full((F,H),1e-5)
  vwo = np.full((H,10),1e-5)
  vbh = np.full((1,H),1e-5)
  vbo = np.full((1,10),1e-5)

  sh = np.zeros((10000,1000))
  so = np.zeros((10000,10))
  sh_ = np.zeros((6000,1000))
  so_ = np.zeros((6000,10))
  for i in range(epoch):
    
    sh = computeLayer(xi, wh, bh)
    xh = relu(sh)
    so = computeLayer(xh, wo, bo)
    yo = softmax(so)
    train_loss.append(CE(newtrain,yo))
    compare = np.equal(np.argmax(yo,axis=1),np.argmax(newtrain,axis=1))
    train_accuracy = np.sum((compare==True))/(trainData.shape[0])
    train_acc.append(train_accuracy)
    print("epoch", i, ": accuracy = ",train_accuracy)

    sh_ = computeLayer(validData, wh, bh)
    xh_ = relu(sh_)
    so_ = computeLayer(xh_, wo, bo)
    valid_pre = softmax(so_)
    valid_loss.append(CE(newvalid,valid_pre))
    compare_valid = np.equal(np.argmax(valid_pre,axis=1),np.argmax(newvalid,axis=1))
    valid_accuracy = np.sum((compare_valid==True))/(validData.shape[0])
    valid_acc.append(valid_accuracy)
    dwo,dbo,dwh,dbh = backprop(xi, xh, wo, newtrain, so)

    if (i==epoch-1):
      print("train_acc is",train_accuracy)
      print("train_loss is",train_loss[-1])
      print("valid_acc is",valid_accuracy)
      print("valid_loss is",valid_loss[-1])

    vwh = gamma*vwh + alpha*dwh
    vwo = gamma*vwo + alpha*dwo
    vbh = gamma*vbh + alpha*dbh
    vbo = gamma*vbo + alpha*dbo
    wo = wo - vwo
    wh = wh - vwh
    bo = bo - vbo
    bh = bh - vbh
    
      
  plt.plot(range(epoch),train_loss,label = "training")
  plt.plot(range(epoch),valid_loss,label = "validation")
  plt.legend()
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.title('Train and Validation Loss', fontsize=16)
  plt.show()

  print(train_acc)
  plt.plot(range(epoch),train_acc,label = "training")
  plt.plot(range(epoch),valid_acc,label = "validation")
  plt.legend()
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.title('Train and Validation Accuracy', fontsize=16)
  plt.show()

learning()
