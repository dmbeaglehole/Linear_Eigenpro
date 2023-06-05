import os
import sys
import argparse
import pickle
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics.functional import accuracy, mean_squared_error as mse

import torch.multiprocessing as mp

import utils
from datasets import *
import linear_regression
import models


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n_train',type=int, default=1000)
    parser.add_argument('-n_test',type=int, default=1000)
    parser.add_argument('-width',type=int, default=128)
    parser.add_argument('-epochs',type=int, default=11)
    parser.add_argument('-patch_size',type=int, default=3)
    parser.add_argument('-s',type=int, default=1024)
    parser.add_argument('-q',type=int, default=64)
    parser.add_argument('-ntk_layer',type=int, default=0)
    parser.add_argument('-mb_reiterates',type=int, default=1)
    parser.add_argument('-start_step',type=int, default=0)
    parser.add_argument('-end_step',type=int, default=1)
    parser.add_argument('-lr',type=float, default=1e-1)
    parser.add_argument('-reg',type=float, default=1e-5)
    parser.add_argument('-jacobian_reg',type=float, default=1e-4)
    parser.add_argument('-dataset',default="cifar10")
    parser.add_argument('-arch',default="myrtle10")
    parser.add_argument('-whiten_image', action='store_true')
    parser.add_argument('-whiten_patches', action='store_true')
    parser.add_argument('-full_train',action='store_true')
    parser.add_argument('-full_test',action='store_true')
    args = parser.parse_args()

    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    if args.dataset == "cifar10":
        trainset, testset = get_cifar_data(args.n_train, args.n_test, args.full_train,\
                                            args.full_test, args.whiten_patches)
        num_classes = 10
        dataset_getter = get_cifar_data

    if args.dataset == "svhn":
        trainset, testset = get_svhn_data(args.n_train, args.n_test)
        num_classes = 10
        dataset_getter = get_cifar_data

    train_X,train_y=process_for_kernels(trainset)
    test_X,test_y=process_for_kernels(testset)

    train_X = train_X.float()

    test_X = test_X.to(train_X.dtype)
    train_y = train_y.to(train_X.dtype)
    test_y = test_y.to(train_X.dtype)

    train_y -= 0.1
    test_y -= 0.1
    
    if args.arch == "myrtle10":
        model = models.Myrtle10(args.width)
    elif args.arch == "myrtle5":
        model = models.Myrtle5(args.width)
    elif args.arch == "myrtleMax":
        model = models.MyrtleMax(args.width)
    elif args.arch == "vgg":
        model = models.VGG(args.width)
    elif args.arch == "simplenet":
        model = models.SimpleNet(args.width)
    elif args.arch == "vanilla":
        model = models.Vanilla(args.width)
    elif args.arch == "mlp":
        model = models.MLP(args.width)

    model.to(train_X.dtype).cuda()
    model.eval()

    print(f'X train shape {train_X.shape}')
    print(f'X test shape {test_X.shape}')
    
    n = len(train_X)

    w = linear_regression.train(model, train_X, train_y, test_X, test_y, jacobian_reg=args.jacobian_reg,
                                s=args.s, q=args.q, ntk_layer=args.ntk_layer, epochs=args.epochs, 
                                mb_reiterates=args.mb_reiterates)
    



