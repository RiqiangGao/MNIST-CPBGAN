from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import logging
from pathlib import Path
from datetime import datetime
import pprint
import argparse
from masked_mnist import IndepMaskedMNIST, BlockMaskedMNIST, FashionMNIST, MNIST, MultiMNIST
from mnist_decoder import ConvDecoder
from mnist_encoder import ConvEncoder
from utils import mkdir
from visualize import Visualizer
from torchvision import datasets, transforms
import pdb
import os
import shutil
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

from pretrain_onemiss_mcls import * 

pretrain_path = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/fashion_128dim/models/model_epoch_0134.pth'

# pretrain_pbigan = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/20210128_withobsv_trclsori/model/0399.pth'

pretrain_pbigan = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/20210216_trvaltt/model/0929.pth'

mnist_path = '/nfs/masi/gaor2/data/mnist/MNIST'
fashion_path = '/nfs/masi/gaor2/data/mnist/fashion'
block_len = 10
batch_size = 128
latent = 128
flow = 2
aeloss = 'bce'
keepobsv = True


trans_mnist = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])


test_data = MultiMNIST(m1root = mnist_path, m2root = fashion_path, block_len = block_len, train=False,transform2=trans_mnist)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

decoder = ConvDecoder(latent)
encoder = ConvEncoder(latent, flow, logprob=False)
pbigan = PBiGAN(encoder, decoder, aeloss).to(device)

e_net = Net().cuda()
e_net.load_state_dict(torch.load(pretrain_path))
for p in e_net.parameters():
    p.requires_grad_(False)
e_net.eval()

cls_net = Net().cuda()

cls_multinet = MMNet(in_channel = 1).to(device)

if pretrain_pbigan is not None:
    model_dict = torch.load(pretrain_pbigan)
    pbigan.load_state_dict(model_dict['pbigan'])
    cls_net.load_state_dict(model_dict['clsnet'])
    cls_multinet.load_state_dict(model_dict['cls_multinet'])
tar_list, bothpred_list, m_pred_list = [], [], []
with torch.no_grad():
    for batch_idx, (xlist, mask, _, label) in enumerate(test_loader):
        x, x2 = xlist
        x = x.to(device)
        mask = mask.to(device).float()

        cls_multinet.eval()
        pbigan.eval()

            #print ('feat is not none')
        x2 = x2.to(device)
        feat, out = e_net(x2)
        
        # ----- This is the four different settings ------------# 

#        z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, None, ae=False)  # (1) pbigan
#        z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, feat, ae=False) # (2) c-pbigan
        
#         x_rec = x * mask # (3) raw
        x_rec = x  # (4) complete

#         if keepobsv:
#             x_rec = x * mask + (1 - mask) * x_rec
        


        _, m_pred = cls_net(x_rec)

        m_pred_cls = m_pred.data.max(1)[1]
        m_pred_list += m_pred_cls.data.cpu().numpy().tolist()

        bothpred = cls_multinet(x_rec, x2)
        cls_multinet.train()
        pbigan.train()

        tar_list += label.data.cpu().numpy().tolist()
        pred_cls = bothpred.data.max(1)[1]
        bothpred_list += pred_cls.data.cpu().numpy().tolist()

accuracy=accuracy_score(tar_list,bothpred_list)
print ('testacc', accuracy)

accuracy=accuracy_score(tar_list,m_pred_list)
print ('testmacc', accuracy)


    