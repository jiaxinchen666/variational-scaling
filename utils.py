import os
import shutil
import time
import pprint
import math
import numpy as np
import numpy
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def dot_metric(a, b):
    return torch.mm(a, b.t())

def dot_metric_normalize(a,b,B):
    logits=torch.mm(torch.div(a,torch.norm(a,dim=1).unsqueeze(1)), torch.div(b,torch.norm(b,dim=1).unsqueeze(1)).t())
    logits_scaled=logits*B
    return logits,logits_scaled

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def euclidean_metric_normalize(a, b, B):
    a=torch.div(a, torch.norm(a, dim=1).unsqueeze(1))
    b=torch.div(b, torch.norm(b, dim=1).unsqueeze(1))
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits=-((a - b) ** 2).sum(dim=2)
    logits_scaled = logits* (B/2)
    #logits = -((a - b)**2).sum(dim=2).sqrt()*(B)
    return logits,logits_scaled

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2


class scaling(nn.Module):
    def __init__(self,margin):
        super(scaling,self).__init__()
        self.margin=margin
        return
    def forward(self,logits,label):
        mask=numpy.ones(logits.shape)
        for i in range(logits.shape[0]):
            mask[i][label[i]]=self.margin
        mask=torch.from_numpy(mask).float().cuda()
        return logits*mask

class sample(nn.Module):
    def __init__(self,mean,std):
        super(sample,self).__init__()
        self.mean=mean
        self.std=std
        return

    def forward(self,logits):
        B = torch.normal(mean=torch.tensor(self.mean), std=torch.tensor(self.std)).cuda()
        return logits*torch.abs(B)

class Bufferswitch(nn.Linear):
    def __init__(self,multiscale=True):
        super().__init__(1, 1)
        if multiscale:
            self.register_buffer('mask',torch.ones(1600))
        else:
            self.register_buffer('mask',torch.ones(1)) #torch.ones(1600) for multiscale

    def set(self, mask):
        self.mask.data.copy_(mask)

    def get(self):
        return self.mask.data


def euclidean_multiscale(a, b, B):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits_vector=-((a - b) ** 2)
    logits_scaled = ((B.squeeze().unsqueeze(0).unsqueeze(1) / 2)*logits_vector).sum(dim=2)
    return logits_vector,logits_scaled

def euclidean_normalize_multiscale(a, b, B):
    a=torch.div(a, torch.norm(a, dim=1).unsqueeze(1))
    b=torch.div(b, torch.norm(b, dim=1).unsqueeze(1))
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)#75 1600 1600
    b = b.unsqueeze(0).expand(n, m, -1)#75 1600 5
    logits_vector=-((a - b) ** 2)
    #print(B.shape,logits_vector.shape) [1600],[75,5,1600]
    logits_scaled = ((B.squeeze().unsqueeze(0).unsqueeze(1) / 2)*logits_vector).sum(dim=2)
    return logits_vector,logits_scaled

def dot_normalize_multiscale(a, b, B):
    a=torch.div(a, torch.norm(a, dim=1).unsqueeze(1)) #75,1600 nm
    b=torch.div(b, torch.norm(b, dim=1).unsqueeze(1)) #5,1600 t() 1600,way mp query*way way 1600
    l = [a[i] * b for i in range(a.shape[0])]
    logits_vector = torch.stack(l)
    logits_scaled = ((B.squeeze().unsqueeze(0).unsqueeze(1)) * logits_vector).sum(dim=2)
    return logits_vector, logits_scaled

#torch.set_printoptions(edgeitems=1600)




