#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# sigma(w*x+b)
def sigma(w, b, x):
    z = np.matmul(x, w) + b
    #print("z",z)
    sig = 1/(1 + np.exp(-z))
    return sig

# L = Lce + Lw
def loss(w, b, x, y, reg):
    y_hat = sigma(w, b, x) #3500*1
    Lce = np.sum(-y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)) / len(y)
    #print("Lce",Lce)
    Lw = (np.linalg.norm(w)**2)*reg/2
    #print("Lw",Lw)
    return Lce + Lw



def grad_loss(w, b, x, y, reg):
    y_hat = sigma(w, b, x)
    diff_w = np.matmul(np.transpose(x), (y_hat - y))/(np.shape(y)[0]) + reg*w
    diff_b = np.sum(y_hat - y, keepdims=True) / (np.shape(y)[0])
  
    return diff_w, diff_b


# Global variable to store the changes of loss
loss_array = []
accu_array = []
def grad_descent(w, b, x, y, alpha, epochs, reg, error_tol = 1e-7):
    global loss_array
    global accu_array
    loss_array = []
    loss_array.append(loss(w, b, x, y, reg))
    accu_array = []
    out_train = np.matmul(x,w)+b
    accur = [np.sum((out_train>=0.5)==y)/(x.shape[0])]
    accu_array.append(accur)
    
    #Stop when total number of epochs reached
    for i in range(epochs):
        grad_w, grad_b = grad_loss(w, b, x, y, reg)
        new_w = w - alpha * grad_w
        new_b = b - alpha * grad_b
        # calculate the new loss value
        loss_array.append(loss(new_w, new_b, x, y, reg))
        # calculate the new accuracy value
        out_train = np.matmul(x,new_w)+new_b
        accur = [np.sum((out_train>=0.5)==y)/(x.shape[0])]
        accu_array.append(accur)        
        # check norm(new_w - w)< error_tol
        norm_err = np.linalg.norm(new_w - w)
        if norm_err < error_tol:
            return new_w,new_b
        else:
            w = new_w
            b = new_b
    return w,b

def Plot_Q3():
    # 3500*28*28 100*28*28  145*28*28
    #[trainData, validData, testData, trainTarget, validTarget, testTarget]
    dataList = loadData()
    dataList = list(dataList)

    # 3500*784    100*784    145*784
    # [trainData, validData, testData]
    for i, data in enumerate(dataList[:3]):
        dataList[i] = data.reshape(len(data), -1)

    trainData  = dataList[0] #3500*784
    validData  = dataList[1] #100*784
    testData   = dataList[2] #145*784

    trainTarget= dataList[3].astype(int) #3500*1
    validTarget= dataList[4].astype(int) #100*1
    testTarget = dataList[5].astype(int) #145*1

    # Initialize weights
    w = np.zeros((dataList[0].shape[1], 1)) #weight 784*1
    b = np.zeros((1, 1))                    #bias   1*1
    r = 0                                   #regularization parameter
    alpha = 0.005
    epochs = 5000

    # Training loss
    alpha = 0.005
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss1 = loss_array.copy()
    accu1 = accu_array.copy()
    iterations1 = range(len(loss1))

    alpha = 0.001
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss2 = loss_array.copy()
    accu2 = accu_array.copy()
    iterations2 = range(len(loss2))

    alpha = 0.0001
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss3 = loss_array.copy()
    accu3 = accu_array.copy()
    iterations3 = range(len(loss3))

    # Validation Loss
    alpha = 0.005
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss1 = loss_array.copy()
    V_accu1 = accu_array.copy()
    V_iterations1 = range(len(V_loss1))

    alpha = 0.001
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss2 = loss_array.copy()
    V_accu2 = accu_array.copy()
    V_iterations2 = range(len(V_loss2))

    alpha = 0.0001
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss3 = loss_array.copy()
    V_accu3 = accu_array.copy()
    V_iterations3 = range(len(V_loss3))

    # Figure generation
    # Loss
    fig1, F1 = plt.subplots(1, 3, constrained_layout=True, sharey=True)
    F1[0].plot(iterations1,loss1,label = "Training")
    F1[0].plot(V_iterations1,V_loss1,label = "Validation")
    F1[0].legend()
    F1[0].set_title('alpha = 0.005')
    F1[1].plot(iterations2,loss2,label = "Training")
    F1[1].plot(V_iterations2,V_loss2,label = "Validation")
    F1[1].legend()
    F1[1].set_title('alpha = 0.001')
    F1[2].plot(iterations3,loss3,label = "Training")
    F1[2].plot(V_iterations3,V_loss3,label = "Validation")
    F1[2].legend()
    F1[2].set_title('alpha = 0.0001')
    
    fig1.suptitle('Loss', fontsize=16)

    # Accuracy
    fig2, F2 = plt.subplots(1, 3, constrained_layout=True, sharey=True)
    F2[0].plot(iterations1,accu1,label = "Training")
    F2[0].plot(V_iterations1,V_accu1,label = "Validation")
    F2[0].legend()
    F2[0].set_title('alpha = 0.005')
    F2[1].plot(iterations2,accu2,label = "Training")
    F2[1].plot(V_iterations2,V_accu2,label = "Validation")
    F2[1].legend()
    F2[1].set_title('alpha = 0.001')
    F2[2].plot(iterations3,accu3,label = "Training")
    F2[2].plot(V_iterations3,V_accu3,label = "Validation")
    F2[2].legend()
    F2[2].set_title('alpha = 0.0001')
    fig2.suptitle('Accuracy', fontsize=16)

    plt.show()

    return


def Plot_Q4():
    # 3500*28*28 100*28*28  145*28*28
    #[trainData, validData, testData, trainTarget, validTarget, testTarget]
    dataList = loadData()
    dataList = list(dataList)

    # 3500*784    100*784    145*784
    # [trainData, validData, testData]
    for i, data in enumerate(dataList[:3]):
        dataList[i] = data.reshape(len(data), -1)

    trainData  = dataList[0] #3500*784
    validData  = dataList[1] #100*784
    testData   = dataList[2] #145*784

    trainTarget= dataList[3].astype(int) #3500*1
    validTarget= dataList[4].astype(int) #100*1
    testTarget = dataList[5].astype(int) #145*1

    # Initialize weights
    w = np.zeros((dataList[0].shape[1], 1)) #weight 784*1
    b = np.zeros((1, 1))                    #bias   1*1
    r = 0                                   #regularization parameter
    alpha = 0.005
    epochs = 5000

    # Training loss
    r = 0.001
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss1 = loss_array.copy()
    accu1 = accu_array.copy()
    iterations1 = range(len(loss1))

    r = 0.1
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss2 = loss_array.copy()
    accu2 = accu_array.copy()
    iterations2 = range(len(loss2))

    r = 0.5
    grad_descent(w, b,trainData, trainTarget,alpha,epochs, r)
    loss3 = loss_array.copy()
    accu3 = accu_array.copy()
    iterations3 = range(len(loss3))

    # Validation Loss
    r = 0.001
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss1 = loss_array.copy()
    V_accu1 = accu_array.copy()
    V_iterations1 = range(len(V_loss1))

    r = 0.1
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss2 = loss_array.copy()
    V_accu2 = accu_array.copy()
    V_iterations2 = range(len(V_loss2))

    r = 0.5
    grad_descent(w, b,validData, validTarget,alpha,epochs, r)
    V_loss3 = loss_array.copy()
    V_accu3 = accu_array.copy()
    V_iterations3 = range(len(V_loss3))

    # Figure generation
    # Loss
    fig1, F1 = plt.subplots(1, 3, constrained_layout=True, sharey=True)
    F1[0].plot(iterations1,loss1,label = "Training")
    F1[0].plot(V_iterations1,V_loss1,label = "Validation")
    F1[0].legend()
    F1[0].set_title('r = 0.001')
    F1[1].plot(iterations2,loss2,label = "Training")
    F1[1].plot(V_iterations2,V_loss2,label = "Validation")
    F1[1].legend()
    F1[1].set_title('r = 0.1')
    F1[2].plot(iterations3,loss3,label = "Training")
    F1[2].plot(V_iterations3,V_loss3,label = "Validation")
    F1[2].legend()
    F1[2].set_title('r = 0.5')
    
    fig1.suptitle('Loss', fontsize=16)

    # Accuracy
    fig2, F2 = plt.subplots(1, 3, constrained_layout=True, sharey=True)
    F2[0].plot(iterations1,accu1,label = "Training")
    F2[0].plot(V_iterations1,V_accu1,label = "Validation")
    F2[0].legend()
    F2[0].set_title('r = 0.001')
    F2[1].plot(iterations2,accu2,label = "Training")
    F2[1].plot(V_iterations2,V_accu2,label = "Validation")
    F2[1].legend()
    F2[1].set_title('r = 0.1')
    F2[2].plot(iterations3,accu3,label = "Training")
    F2[2].plot(V_iterations3,V_accu3,label = "Validation")
    F2[2].legend()
    F2[2].set_title('r = 0.5')
    fig2.suptitle('Accuracy', fontsize=16)

    plt.show()

    return

#============================== Main part ==============================#
Plot_Q3()


#============================== Test part ==============================#
### 3500*28*28 100*28*28  145*28*28
###[trainData, validData, testData, trainTarget, validTarget, testTarget]
##dataList = loadData()
##dataList = list(dataList)
##
### 3500*784    100*784    145*784
### [trainData, validData, testData]
##for i, data in enumerate(dataList[:3]):
##    dataList[i] = data.reshape(len(data), -1)
##
##trainData  = dataList[0] #3500*784
##validData  = dataList[1] #100*784
##testData   = dataList[2] #145*784
##
##trainTarget= dataList[3] #3500*1
##validTarget= dataList[4] #100*1
##testTarget = dataList[5] #145*1
##
### Initialize weights
##w = np.zeros((dataList[0].shape[1], 1)) #weight 784*1
##b = np.zeros((1, 1))                    #bias   1*1
##r = 0                                   #regularization parameter
##
##cross_entropy_loss = loss(w, b, trainData, trainTarget, r)
##print(cross_entropy_loss)
####
####loss = grad_loss(w, b, trainData, trainTarget, r)
####print(loss[0].shape)
####print(loss[1])
##

##[W_best,b_best] = grad_descent(w, b,trainData, trainTarget,0.005,1000, r)
##print(loss_array)
##print(accu_array)


##
##iterations = range(len(loss_array))
##plt.plot(iterations,loss_array)
##plt.show()

