import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch

prefix = "/Users/yxqu/Desktop/Research/Koopman/data"
#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4, train_size=100000, test_size=10000):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise, theta)    
    # if name == 'inverted_pendulum':
    #     return inverted_pendulum(train_size, test_size)
    # if name == 'inverted_double_pendulum':
    #     return inverted_double_pendulum(train_size, test_size)
    if name == "Ant-v3":
        return mujoco("Ant-v3", train_size, test_size)
    if name == "inverted_pendulum":
        return mujoco("InvertedPendulum-v2", train_size, test_size)
    if name == "inverted_double_pendulum":
        return mujoco("InvertedDoublePendulum-v2", train_size, test_size)
    else:
        raise ValueError('dataset {} not recognized'.format(name))

def inverted_pendulum():
    path = "{}/{}/trained.npz".format(prefix, "InvertedPendulum-v2")
    raw_data = np.load(path)
    states = raw_data["states"]
    done = raw_data["dones"]
    print(done)
    exit()

    # data = np.load('../StateGen/data/InvertedPendulum-v2/states.npy')
    # X_train = data[0:50000]   
    # X_test = data[50000:100001]
    # X_train_clean = X_train.copy()
    # X_test_clean = X_test.copy()
    return X_train, X_test, X_train_clean, X_test_clean, 4, 1

def inverted_double_pendulum(X_train_size, X_test_size):
    path = "{}/{}/trained.npz".format(prefix, "InvertedDoublePendulum-v2")
    raw_data = np.load(path)
    states = raw_data["states"]
    dones = raw_data["dones"]
    for i in range(X_train_size - 1, len(states)):
        if dones[i]:
            X_train_size = i + 1
            break
    X_train = states[0:X_train_size]
    indices_train = [0]
    for i in range(X_train_size-1):
        if dones[i]:
            indices_train.append(i + 1)
    
    for i in range(X_train_size + X_test_size - 1, len(states)):
        if dones[i]:
            X_test_size = i - X_train_size + 1
            break
    X_test = states[X_train_size:X_train_size + X_test_size]

    indices_test = [0]
    for i in range(X_test_size-1):
        if dones[i + X_train_size]:
            indices_test.append(i+1)
    
    X_train, X_test = rescale(X_train[:,:7], X_test[:,:7])
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    return X_train, X_test, X_train_clean, X_test_clean, indices_train, indices_test, 7, 1

def mujoco(env_id, X_train_size, X_test_size):
    path = "{}/{}/trained.npz".format(prefix, env_id)
    raw_data = np.load(path)
    states = raw_data["states"]
    actions = raw_data["actions"]
    dones = raw_data["dones"]
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    if env_id == "Ant-v3":
        state_dim = 26
    elif env_id == "InvertedPendulum-v2":
        state_dim = 4
    elif env_id == "InvertedDoublePendulum-v2":
        state_dim = 7
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)

    # print("X_train_size: ", X_train_size)
    # print("X_test_size: ", X_test_size)
    
    states = np.concatenate((states[:,:state_dim], actions), axis=1)

    for i in range(X_train_size - 1, len(states)):
        if dones[i]:
            X_train_size = i + 1
            break
    X_train = states[0:X_train_size]
    indices_train = [0]
    for i in range(X_train_size-1):
        if dones[i]:
            indices_train.append(i + 1)
    # print("indices_train: ", indices_train)
    for i in range(X_train_size + X_test_size - 1, len(states)):
        if dones[i]:
            X_test_size = i - X_train_size + 1
            break
    X_test = states[X_train_size:X_train_size + X_test_size]

    indices_test = [0]
    for i in range(X_test_size-1):
        if dones[i + X_train_size]:
            indices_test.append(i+1)
    
    # print("indices_test: ", indices_test)
    X_train, X_test = rescale(X_train[:, :], X_test[:,:])
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    return X_train, X_test, X_train_clean, X_test_clean, indices_train, indices_test, state_dim, action_dim


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min(axis=0)
    Xmax = Xsmall.max(axis=0)
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test


def pendulum_lin(noise):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    X = sol(anal_ts, 0.8)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate   
    Xclean = Xclean.T.dot(Q.T)     
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, [0], [0], 64, 1


def pendulum(noise, theta=2.4):
    
    np.random.seed(1)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_train_clean.shape)
    # print(X_test_clean.shape)
    return X_train, X_test, X_train_clean, X_test_clean, [0], [0], 64, 1
