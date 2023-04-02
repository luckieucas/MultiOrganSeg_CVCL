'''
Descripttion: kl divergence loss for consistency
version: 
Author: Luckie
Date: 2021-12-01 22:01:49
LastEditors: Luckie
LastEditTime: 2021-12-02 00:02:37
'''
import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
import numpy as np
import pickle


class KL_DIV(nn.Module):
    def __init__(self):
        super(KL_DIV, self).__init__()
    
    def forward(self, output1=None, output2=None, ul1=None, br1=None, ul2=None, br2=None):
        logits1 = F.softmax(output1,1)
        logits2 = F.softmax(output2,1)
        for idx in range(logits1.size(0)): # iterate use batch size
            logits1_idx = logits1[idx]
            logits2_idx = logits2[idx]
            logits1_idx_overlap = logits1_idx[:, ul1[idx][0]:br1[idx][0], ul1[idx][1]:br1[idx][1], ul1[idx][2]:br1[idx][2]] #.view(-1, output_ul1.size(1))
            logits2_idx_overlap = logits2_idx[:, ul2[idx][0]:br2[idx][0], ul2[idx][1]:br2[idx][1], ul2[idx][2]:br2[idx][2]] #.view(-1, output_ul1.size(1))
