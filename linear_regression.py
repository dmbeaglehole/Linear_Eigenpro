import torch
import torch.nn as nn
from torchmetrics.functional import accuracy, mean_squared_error as mse
import torch.cuda.comm
from functorch import vmap #, jvp, jvp

import functools
from tqdm import tqdm
from math import sqrt
import numpy as np

import time
from copy import deepcopy

from features import *

def get_devices_list():
    DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
    if torch.cuda.is_available():
        devices = [f'cuda:{i}' for i in DEVICES.split(',')]
    else:
        devices = ['cpu']
    return devices

def train(Phi, train_X, train_y, test_X, test_y, s=5120, q=8, epochs=5, reg=5e-5, jacobian_reg=None, mb_reiterates=1):

    devices = get_devices_list()
    num_devices = len(devices)
    print(f'Total devices: {num_devices}')
    num_classes = train_y.shape[1]

    s = min(len(train_X),s)
    q = min(len(train_X)-1,q)
    m = int(128*num_devices)
    
    s_idx = np.random.choice(len(train_X), size=s, replace=False)

    print("Computing K_s")
    start = time.time()
    Ks = batch_multiply(train_X[s_idx], train_X[s_idx], m//num_devices).to(devices[0]).double()
    end = time.time()
    print("Ks comp. time:",end-start)

    print("Running LOBPCG")
    L, U = torch.lobpcg(Ks, q+1) # top q+1 eigenvalues/vectors of Ks
    beta = Ks.max()
    del Ks
    
    print("transforming eigenvectors")
    U = U.to(train_X.dtype)
    L = L.to(train_X.dtype)
    U = U[:,:q]
    
    new_U = [0. for _ in range(num_devices)]
    Xs_mbs = torch.cuda.comm.scatter(train_X[s_idx], devices)
    Us = torch.cuda.comm.scatter(U, devices)
    Ls = scatter_copies(L, devices)

    print("scattered copies")

    # Not parallel because Phi_X_s is on CPU
    for i, device in enumerate(devices):
        ub_size = m//num_devices
        Xs_ubs = torch.split(Xs_mbs[i], ub_size)
        U_batches = torch.split(Us[i], ub_size)
        for b in range(len(U_batches)):
            U_ub = U_batches[b]
            Phi_ub = Phi(Xs_ubs[b])
            new_U[i] += Phi_ub.T@U_ub / (Ls[i][:q]**0.5) # L:(q,) , Phi_X_s.T@U : (p,q) -> U: (p,q)

    U = torch.cuda.comm.reduce_add(new_U)

    L *= len(train_X)/s
    lam_qp1 = L[q].cpu()+0
    L = L[:q]

    print("eigenvalues:",L)
    print("Pre-conditioner svd done")
    U = U.to(train_X.dtype)
    L = L.to(train_X.dtype)

    print("U",U.shape)
    print("L",L.shape)
    
    del Xs_mbs, new_U, 

    Us = scatter_copies(U, devices)
    Ls = scatter_copies(L, devices)

    def preconditioned(gradient, device_id, tau=1.):
        U = Us[device_id]
        L = Ls[device_id]
        
        t1 = U.T @ gradient
        t2 = (1-tau*lam_qp1/L).view(-1,1)*t1
        t3 = gradient - U @ t2 # (n_params, 1)
        return t3

    def train_step(w, X_mb, y_mb, eta, devices):
        bs = len(X_mb)

       
        # get Phi(X) for each microbatch
        X_ubs = torch.cuda.comm.scatter(X_mb, devices)
        Phi_ubs = [None]*num_devices
        for i, device in enumerate(devices):
            Phi_ubs[i] = Phi(X_ubs[i])

        # gradient update
        def microbatch_grad(X_ub, Phi_ub, y_ub, w, device_id):

            minibatch_gradient = Phi_ub.T @ (Phi_ub @ w - y_ub) # train_y[b]) 

            jac_grad = 0.
            if jacobian_reg is not None:
                start = time.time()
                def single_JtJw(w_):
                    return my_JtJw(Phi, X_ub, w_)
                jac_grad = jacobian_reg * vmap(single_JtJw)(w.T).T
            return preconditioned(minibatch_gradient + reg*w + jac_grad, device_id)
        
        for _ in range(mb_reiterates):
            ws = scatter_copies(w, devices)
            y_ubs = torch.cuda.comm.scatter(y_mb, devices)
            grads = [None]*num_devices
            for i, device in enumerate(devices):
                grads[i] = microbatch_grad(X_ubs[i], Phi_ubs[i], y_ubs[i], ws[i], i)/len(devices)

            grad = torch.cuda.comm.reduce_add(grads)
            w = w - eta/bs*grad
        
        Xws = [None]*num_devices
        for i in range(num_devices):
            Xws[i] = Phi_ubs[i]@ws[i]
        Xw = torch.cuda.comm.gather(Xws, destination="cpu")

        train_error_mb = mse(Xw, y_mb)
        train_acc_mb = accuracy(Xw, torch.argmax(y_mb,dim=1), 
                                    task='multiclass', num_classes=num_classes) 

        return w, train_error_mb, train_acc_mb

    def test_step(w, X_mb, y_mb, devices):
        X_ubs = torch.cuda.comm.scatter(X_mb, devices)
        Phi_ubs = [None]*num_devices
        for i, device in enumerate(devices):
            Phi_ubs[i] = Phi(X_ubs[i])

        Xws = [None]*num_devices
        for i in range(num_devices):
            Xws[i] = Phi_ubs[i]@ws[i]
        Xw = torch.cuda.comm.gather(Xws, destination="cpu")
        loss = mse(Xw,y_mb)
        acc = accuracy(Xw, torch.argmax(y_mb,dim=1), 
                        task='multiclass', num_classes=num_classes)
        return loss, acc
        
    
    def tune_eta(train_X, train_y, beta, m):
        if m < beta / lam_qp1 + 1:
            eta = m / beta
        else:
            eta = 0.99 * 2 * m / (beta + (m - 1) * lam_qp1)
        
        factor = 1.5
        num_classes = train_y.shape[1]
        
        while True:

            print(f'Current eta: {eta}')
            w = torch.zeros((num_ntk_params, num_classes)).to(train_X.dtype).to(devices[0])
            batch_ids = torch.split(torch.randperm(len(train_X)), m)
                
            comp_error = None
            thresh = 1.01
            test_steps = 95
            test_epochs = 3
            
            t = 0
            for _ in range(test_epochs): # epoch
                t = t + 1
                for s, b in enumerate(batch_ids): # batch
                    t = t + s
                    if t>=test_steps:
                        break
                    
                    X_mb = train_X[b]
                    y_mb = train_y[b]
                    w, train_error_mb, train_acc_mb = train_step(w, X_mb, y_mb, eta, devices)

                    if t==1:
                        comp_error = train_error_mb
                        print("Comp error",comp_error)

                    if train_error_mb > thresh*comp_error:
                        return eta/(factor**3)

            print("Train error at last batch",train_error_mb.item())
            print("Doubling eta")
            print()
            eta = eta*factor

    print("Tuning LR")
    start = time.time()
    eta = tune_eta(train_X, train_y, beta, m)
    end = time.time()
    print("Tuning time:",end-start)

    w = torch.zeros((num_ntk_params,num_classes)).to(train_X.dtype).to(devices[0])
    
    best_w = None
    best_loss = None
    best_acc = -1*float("inf")

    for t in range(epochs): # epoch
        print(f'Epoch {t+1} out of {epochs}')
        start = time.time()
        batch_ids = torch.split(torch.randperm(len(train_X)), m)
            
        # update w
        tot_loss = 0
        tot_acc = 0
        for s, b in enumerate(batch_ids): # batch
            X_mb = train_X[b]
            y_mb = train_y[b]
            bs = len(X_mb)
             
            w, train_error_mb, train_acc_mb = train_step(w, X_mb, y_mb, eta, devices)

            tot_loss += train_error_mb*bs
            tot_acc += train_acc_mb*bs

        end = time.time()
        print(f'epoch time: {end-start}')
        print(f'train acc: {100*tot_acc/len(train_X)}')
        print(f'train error: {tot_loss/len(train_X)}')
        print()
         
        if (t==epochs-1) or (t%1==0):
            # get test loss
            n_test = 0
            batch_ids = torch.split(torch.randperm(len(test_X)), m)
            ws = scatter_copies(w, devices)

            print("Testing..")
            tot = 0
            loss = 0
            for b in batch_ids: # batch
                X_mb = test_X[b]
                y_mb = test_y[b]
                bs = len(X_mb)
               
                loss_mb, acc_mb = test_step(w, X_mb, y_mb, devices)
                loss += loss_mb*bs
                tot += acc_mb*bs
                n_test += len(b)

            test_acc = 100*tot/n_test
            test_loss = loss/n_test

            if test_acc > best_acc:
                best_w = w.clone()
                best_acc = test_acc
                best_loss = test_loss

            print(f'test acc: {test_acc}')
            print(f'test loss: {test_loss}')
            print()
    
    print("best acc",best_acc.item())
    print("best loss",best_loss.item())
    return best_w

