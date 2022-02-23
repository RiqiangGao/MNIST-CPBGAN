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
from inception import InceptionV3
from compute_fid import *
from pretrain_onemiss_mcls import * 
from scipy import linalg
pretrain_path = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/fashion_128dim/models/model_epoch_0134.pth'

# pretrain_pbigan = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/20210128_withobsv_trclsori/model/0399.pth'

pretrain_pbigan = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/20210216_trvaltt/model/0929.pth'

mnist_path = '/nfs/masi/gaor2/data/mnist/MNIST'
fashion_path = '/nfs/masi/gaor2/data/mnist/fashion'
block_len = 10
batch_size = 32
latent = 128
flow = 2
aeloss = 'bce'
keepobsv = True
FEATURE_DIM = 2048

trans_mnist = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])


'''
test_data = MultiMNIST(m1root = mnist_path, m2root = fashion_path, block_len = block_len, train=False,transform2=trans_mnist)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

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

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
model = InceptionV3([block_idx], RESIZE).to(device)

ori_feat = torch.zeros((len(test_data), FEATURE_DIM))
gen_feat = torch.zeros((len(test_data), FEATURE_DIM))
start = 0
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

#        z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, None, ae=False)  # pbigan
        z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, feat, ae=False) # c-pbigan
        
#         x_rec = x * mask # raw
#         x_rec = x  # complete
        

        x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        rec224 = F.interpolate(x_rec, size=(224, 224), mode='bilinear', align_corners=False)
        x224 = x224.repeat(1,3,1,1)
        rec224 = rec224.repeat(1, 3, 1, 1)
        
        pred = model(x224)[0]
        pred_gen = model(rec224)[0]
        
        end = start + pred.shape[0]
        
        ori_feat[start:end] = pred.squeeze(2).squeeze(2)
        gen_feat[start:end] = pred_gen.squeeze(2).squeeze(2)
        
        start += pred.shape[0]
pdb.set_trace()        
print (start, end, ori_feat.shape)
'''

ori_feat = 0.4 * torch.randn((1000, 2048)).float()

gen_feat = 0.5 * torch.randn((1000, 2048)).float()

ori_mu = np.mean(ori_feat.numpy(), axis=0)
ori_sigma = np.cov(ori_feat.numpy(), rowvar=False)

gen_mu = np.mean(gen_feat.numpy(), axis=0)
gen_sigma = np.cov(gen_feat.numpy(), rowvar=False)

# print (calculate_frechet_distance(ori_mu, ori_sigma, gen_mu, gen_sigma, eps=1e-6))
print ('-----')
pdb.set_trace()
covmean, _ = linalg.sqrtm(ori_sigma.dot(gen_sigma), disp=False)

covmean, _ = linalg.sqrtm(np.random.normal(0,1, (2048, 2048)), disp=False)