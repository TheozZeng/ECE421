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


def convert_onehot(train_target, valid_target, test_target):
    new_train = np.zeros((train_target.shape[0], 10))
    new_valid = np.zeros((valid_target.shape[0], 10))
    new_test = np.zeros((test_target.shape[0], 10))

    for item in range(0, train_target.shape[0]):
        new_train[item][train_target[item]] = 1
    for item in range(0, valid_target.shape[0]):
        new_valid[item][valid_target[item]] = 1
    for item in range(0, test_target.shape[0]):
        new_test[item][test_target[item]] = 1

##    print("trainTarget:",new_train.shape)
##    print("validTarget:",new_valid.shape)
##    print("testTarget:",new_test.shape)
    
    return new_train, new_valid, new_test





def shuffle(data, target):
    np.random.seed(421)
    rand_idx = np.random.permutation(len(data))
    return data[rand_idx], target[rand_idx]


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
# s(l) - sigma -> x(l)
def relu(x):
    sigma = np.maximum(x,0) #Element-wise maximum of array elements.
    return sigma

def softmax(x):
    x = x - np.amax(x, axis=1, keepdims=True)
    numerator = np.exp(x)
    demoninator = (np.sum(np.exp(x), axis=1, keepdims=True))
    res = numerator/demoninator
    return res


def compute(x, w, b):
    # This function will accept 3 arguments: a weight matrix, an input vector, and a
    # bias vector and return the product between the weights and input, plus the biases
##    print("x",x.shape)
##    print("w",w.shape)
##    print("b",b.shape)
    res = np.matmul(x,w)+b
    return res

def average_ce(target, prediction):
    #target yk(n)
    #prediction pk(n)
##    matrix = target*np.log(prediction) #point_wise multiply y*log(p)
##    res = (-1/target.shape[0])*np.sum(matrix)
    
    return (-1/target.shape[0])*np.sum(target*np.log(prediction))


def grad_ce(target, logits):
    # gradient of the cross entropy loss withrespect to the inputs to the softmax function
    p = softmax(logits)
    res = (p - target)/target.shape[0]
    return res

#================================== back prop =================================#
def backprop(xi, xh, w, target, prediction):
    dl_dwo = DL_Dwo(xh, target, prediction)
    dl_dbo = DL_Dbo(target, prediction)
    dl_dwh = DL_Dwh(xi, xh, w, target, prediction)
    dl_dbh = DL_Dbh(xi, xh, w, target, prediction)
    return dl_dwo,dl_dbo,dl_dwh,dl_dbh

def DL_Dwo(xh, target, prediction):
    softmax_ce = grad_ce(target, prediction)
    xh_transpose = np.transpose(xh)
    Dwo = np.matmul(xh_transpose,softmax_ce)
    return Dwo

def DL_Dbo(target, prediction):
    softmax_ce = grad_ce(target, prediction)
    Dbo = np.transpose(sum(softmax_ce)).reshape(1, 10) #K=10
    return Dbo

def DL_Dwh(xi, xh, wo, target, prediction):
    DL_Dso = grad_ce(target, prediction)
    Dso_Dxh = np.transpose(wo)
    Dxh_Dsh = np.where(xh > 0, 1, 0)
    Dsh_Dwh = np.transpose(xi)
    dwh = np.matmul(Dsh_Dwh,Dxh_Dsh*np.matmul(DL_Dso,Dso_Dxh))
    return dwh

def DL_Dbh(xi, xh, wo, target, prediction):
    DL_Dso = grad_ce(target, prediction)
    Dso_Dxh = np.transpose(wo)
    Dxh_Dsh = np.where(xh > 0, 1, 0)
    Dsh_Dwh = 1
    dbh = sum(Dxh_Dsh * np.dot(DL_Dso, Dso_Dxh)).reshape(1, 1000)#H = 1000
    return dbh


    
    
    
#================================= Data Processing ===========================#

def Process_Data():
    # 10000*28*28 6000*28*28  2720*28*28
    #[trainData,  validData,  testData, trainTarget, validTarget, testTarget]
    dataList = load_data()
    dataList = list(dataList)

    # 10000*784   6000*784   2720*784
    # [trainData, validData, testData]
    for i, data in enumerate(dataList[:3]):
        dataList[i] = data.reshape(len(data), -1)

    trainData  = dataList[0] #10000*784
    validData  = dataList[1] #6000*784
    testData   = dataList[2] #2720*784

    trainTarget= dataList[3]
    validTarget= dataList[4]
    testTarget = dataList[5]

    #one hot encoding of Target 
    TargetList = convert_onehot(trainTarget, validTarget, testTarget)
    TargetList = list(TargetList)

    trainTarget= TargetList[0]
    validTarget= TargetList[1]
    testTarget = TargetList[2]

    

    print("trainData:",trainData.shape)
    print("validData:",validData.shape)
    print("testData:",testData.shape)

    print("trainTarget:",trainTarget.shape)
    print("validTarget:",validTarget.shape)
    print("testTarget:",testTarget.shape)


    return trainData,validData,testData,trainTarget,validTarget,testTarget

def initialize_weight(F,H,K):
    wo = np.random.normal(0, np.sqrt(2/(H+K)), (H,K))
    bo = np.zeros((1,K))
    wh = np.random.normal(0, np.sqrt(2/(F+H)), (F,H))
    bh = np.zeros((1,H))
    return wo,bo,wh,bh

def initialize_V_matrix(F,H,K):
    #initialize them to the same size as the hidden and output layer weight matrix sizes
    #with a very small value (e.g. 1e-5).
    V_wo = np.full((H,K),1e-5)
    V_bo = np.full((1,K),1e-5)
    V_wh = np.full((F,H),1e-5)
    V_bh = np.full((1,H),1e-5)
    return V_wo,V_bo,V_wh,V_bh

def initialize_sh_so(N,F,H,K):
    sh = np.zeros((N,H))
    so = np.zeros((N,K))
    return sh, so

def foward_prop(xi,wh,bh,wo,bo):
    sh = compute(xi, wh, bh)
    xh = relu(sh)
    so = compute(xh, wo, bo)
    yo = softmax(so)    
    return yo,xh,so

def compute_loss_and_accuracy(Target,predicton,Data):
    loss  = average_ce(Target,predicton)
    #check if the prediction is same as target
    compare = np.equal(np.argmax(predicton,axis=1),np.argmax(Target,axis=1))
    accuracy = np.sum((compare==True))/(Data.shape[0])
    return loss, accuracy

def plot_loss():
    iteration = range(epoch)
    plt.plot(iteration,train_loss,label = "training")
    plt.plot(iteration,valid_loss,label = "validation")
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Train and Validation Loss', fontsize=16)
    plt.show()
    return

def plot_accuracy():
  iteration = range(epoch)
  plt.plot(iteration,train_acc,label = "training")
  plt.plot(iteration,valid_acc,label = "validation")
  plt.legend()
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.title('Train and Validation Accuracy', fontsize=16)
  plt.show()
  return

    


#================================= global Variable  ===================================#
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

epoch=200

# define Dimension 
F= 784
H= 1000
K = 10

# define learning rate and momentum 
momentum =0.9
learn_rate=0.1

#================================= Learning  ===================================#
def Train_Network():
  trainData,validData,testData,trainTarget,validTarget,testTarget = Process_Data()

  wo,bo,wh,bh = initialize_weight(F,H,K)
  V_wo,V_bo,V_wh,V_bh = initialize_V_matrix(F,H,K)
  Train_sh,Train_so = initialize_sh_so(10000,F,H,K)
  Valid_sh,Valid_so = initialize_sh_so(6000,F,H,K)

  
  for i in range(epoch):
    #training set foward
    yo,xh,so = foward_prop(trainData,wh,bh,wo,bo)
    loss, accuracy = compute_loss_and_accuracy(trainTarget,yo,trainData)
    train_loss.append(loss)
    train_acc.append(accuracy)
   
    #validation set forward
    yo_,xh_,so_ = foward_prop(validData,wh,bh,wo,bo)
    loss, accuracy = compute_loss_and_accuracy(validTarget,yo_,validData)
    valid_loss.append(loss)
    valid_acc.append(accuracy)

    #print("epoch =",i, "train loss=",train_loss[-1],"train acc=",train_acc[-1])

    #back propagation
    dwo,dbo,dwh,dbh = backprop(trainData, xh, wo, trainTarget, so)
    
    V_wo = momentum *V_wo + learn_rate*wo
    wo = wo - V_wo
    
    V_bo = momentum *V_bo + learn_rate*dbo
    bo = bo - V_bo
    
    V_wh = momentum *V_wh + learn_rate*wh
    wh = wh - V_wh
    
    V_bh = momentum *V_bh + learn_rate*dbh
    bh = bh - V_bh  
  return



Train_Network()
plot_loss()
plot_accuracy()
#======================== Test =========================#

