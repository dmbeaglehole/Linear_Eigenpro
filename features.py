import torch
import torch.nn as nn
from functorch import make_functional, vmap, jacrev, vjp, jvp
#from torch.autograd.functional import vjp, jvp

from copy import deepcopy
from tqdm import tqdm
import numpy as np

def scatter_copies(A, devices):
    return [deepcopy(A).to(device) for device in devices]

def get_feature_map(model, X, ntk_layer=0):
    fnet, params = make_functional(model)
    
    all_params = list(params)
    ntk_params = tuple(all_params[ntk_layer:])
    feature_params = all_params[:ntk_layer] 

    def fnet_single(ntk_params, x):
        new_params = tuple(feature_params + list(ntk_params))
        return fnet(new_params, x.unsqueeze(0)).squeeze(0)

    jac = vmap(jacrev(fnet_single), (None, 0))(ntk_params, X)
    jac = [j.flatten(2) for j in jac]

    return torch.squeeze(torch.concat(jac,dim=2).detach())

def batch_feature_map(model, device, X, ntk_layer=0, mb_size=32):
    n_batches = len(X) + 1
    batches = []
    for i in range(n_batches):
        s = i*mb_size
        e = (i+1)*mb_size if i<n_batches-1 else len(X)
        X_mb = X[s:e].to(device)
        if len(X_mb)==0:
            continue

        batches.append(get_feature_map(model, X_mb, ntk_layer).cpu())

        del X_mb
    return torch.concat(batches, dim=0).to(X.dtype)


def get_param_count(model, ntk_layer=0):
    tot = 0
    for i, p in enumerate(model.parameters()):
        if i >= ntk_layer:
            tot += p.numel() 
    return tot

def batch_norm(x1, device):
    mb_size = 128
    n1_batches = len(x1)//mb_size  + 1
    n2_batches = len(x2)//mb_size  + 1

    prod = []
    for i in range(n1_batches):
        s1 = i*mb_size
        e1 = (i+1)*mb_size if i<n1_batches-1 else len(x1)
        x1_mb = x1[s1:e1].to(device)
        if len(x1_mb)==0:
            continue

        prod.append((x1_mb.norm(dim=-1)**2).cpu())
        del x1_mb

    return torch.concat(prod,dim=0)

def batch_multiply(x1, x2, device):
    mb_size = 64
    n1_batches = len(x1)//mb_size  + 1
    n2_batches = len(x2)//mb_size  + 1

    prod = []
    for i in range(n1_batches):
        s1 = i*mb_size
        e1 = (i+1)*mb_size if i<n1_batches-1 else len(x1)
        x1_mb = x1[s1:e1].to(device).double()
        if len(x1_mb)==0:
            continue

        prod_1 = []
        for j in range(n2_batches):
            s2 = j*mb_size
            e2 = (j+1)*mb_size if i<n2_batches-1 else len(x2)
            x2_mb = x2[s2:e2].to(device).double()
            if len(x1_mb)==0:
                continue

            prod_1.append((x1_mb@x2_mb.T).cpu())

            del x2_mb
        del x1_mb

        prod.append(torch.concat(prod_1,dim=1))

    return torch.concat(prod,dim=0)

def batch_feature_norm(model, device, x1):
    mb_size = 256
    n1_batches = len(x1)//mb_size  + 1

    prod = []
    for i in range(n1_batches):
        s1 = i*mb_size
        e1 = (i+1)*mb_size if i<n1_batches-1 else len(x1)
        x1_mb = x1[s1:e1].to(device)
        if len(x1_mb)==0:
            continue
        
        Phi_mb = get_feature_map(model, x1_mb)
        prod.append((Phi_mb.norm(dim=-1)**2).cpu())
        del Phi_mb, x1_mb

        torch.cuda.empty_cache()

    return torch.concat(prod,dim=0)

def batch_feature_multiply(model, device, x1, x2, mb_size=32):
    n1_batches = len(x1)//mb_size + 2
    n2_batches = len(x2)//mb_size + 2

    prod = []
    for i in range(n1_batches):
        s1 = i*mb_size
        e1 = (i+1)*mb_size if i<n1_batches-1 else len(x1)
        x1_mb = x1[s1:e1]
        if len(x1_mb)==0:
            continue

        Phi_x1_mb = get_feature_map(model, x1_mb)

        prod_1 = []
        for j in range(n2_batches):
            s2 = j*mb_size
            e2 = (j+1)*mb_size if i<n2_batches-1 else len(x2)
            x2_mb = x2[s2:e2]
            if len(x2_mb)==0:
                continue

            Phi_x2_mb = get_feature_map(model, x2_mb)

            prod_1.append(Phi_x1_mb@Phi_x2_mb.T)


        prod.append(torch.concat(prod_1,dim=1))

    return torch.concat(prod,dim=0)

def parallel_feature_multiply(models, devices, x1, x2, mb_size=32):
    
    K_rows = [None]*len(devices)
    x1_rows = torch.cuda.comm.scatter(x1, devices)
    x2_copies = scatter_copies(x2, devices)
    for i, device in enumerate(devices):
        K_rows[i] = batch_feature_multiply(models[i], devices[i], x1_rows[i], x2_copies[i], mb_size=mb_size)
    
    return torch.cuda.comm.gather(K_rows, destination="cpu")

def my_JtJw(model, X_ub, w):
    n, d = X_ub.shape 
    def Phi(X):
        """
        X : (n*d, )
        out : (p, )
        """
        return get_feature_map(model, X.reshape(n, d)).sum(dim=0)
    
    Jw_f = vjp(Phi, X_ub.reshape(-1))[1]
    Jw = Jw_f(w)[0]
    JtJw = jvp(Phi, (X_ub.reshape(-1),), (Jw,))[1]
    return JtJw

