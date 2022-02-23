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
from masked_mnist import MultiMaskMNIST,MultiMaskMNISTv2
from mnist_decoder import ConvDecoder
from mnist_encoder import ConvEncoder
from utils import mkdir
from visualize import Visualizer
from torchvision import datasets, transforms
import pdb
import os
import shutil
from sklearn.metrics import accuracy_score
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class MMNet(nn.Module):              
    def __init__(self, in_channel):          
        super(MMNet, self).__init__()
        self.fc1_m1 = nn.Linear(320, 128)  
        self.fc1_m2 = nn.Linear(320, 128)  
        
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channel, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.model2 = nn.Sequential(
            nn.Conv2d(in_channel, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Linear(128, 10) 
        self.fc2_m1 = nn.Linear(128, 10) 
        self.fc2_m2 = nn.Linear(128, 10) 

    def forward(self, x1, x2):   # define the network using the module in __init__
        x1 = self.model1(x1)
        x1 = x1.view(-1, 320)
        x1 = nn.ReLU()(self.fc1_m1(x1))
        pred1 = self.fc2_m1(x1)
        
        x2 = self.model2(x2)
        x2 = x2.view(-1, 320)
        x2 = nn.ReLU()(self.fc1_m2(x2))
        pred2 = self.fc2_m2(x2)
        
        x = torch.mean(torch.stack([x1, x2]), 0)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x1, F.log_softmax(pred1, dim=1), x2, F.log_softmax(pred2, dim=1), F.log_softmax(x, dim=1)


class PBiGAN(nn.Module):
    def __init__(self, encoder, decoder, ae_loss='bce'):  
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ae_loss = ae_loss

    def forward(self, x, mask, feat, ae=True):
        z_T = self.encoder(x * mask, None)  # incomplete data encoder
        #if len(z_gen) == 0:
        z_gen = torch.empty_like(z_T).normal_()  # priori distribution encoder, here maybe can transfer from another domain
        
        if feat is not None:
            z_T = F.normalize(z_T) + F.normalize(feat)
            z_gen = F.normalize(z_gen) + F.normalize(feat)
        else:
            z_T = 2 * F.normalize(z_T)
            z_gen = 2 * F.normalize(z_gen)
        
        x_gen_logit, x_gen = self.decoder(z_gen)  # got x for z

        x_logit, x_recon = self.decoder(z_T)  # got x for z_T

        recon_loss = None
        if ae:
            if self.ae_loss == 'mse':
                recon_loss = F.mse_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'l1':
                recon_loss = F.l1_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'smooth_l1':
                recon_loss = F.smooth_l1_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'bce':
                # Bernoulli noise
                # recon_loss: -log p(x|z)
                #pdb.set_trace()
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_logit * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()

        return z_T, z_gen, x_recon, x_gen, recon_loss

class ConvCritic(nn.Module):  # for what? Discriminator
    def __init__(self, latent_size):
        super().__init__()

        self.DIM = 64
        self.main = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, 2 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )

        embed_size = 64

        self.z_fc = nn.Sequential(
            nn.Linear(latent_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, embed_size),
        )

        self.x_fc = nn.Linear(4 * 4 * 4 * self.DIM, embed_size)

        self.xz_fc = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, 1),
        )

    def forward(self, input):
        x, z = input
        x = x.view(-1, 1, 28, 28)
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.x_fc(x)
        z = self.z_fc(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_fc(xz)
        return xz.view(-1)
    

class GradientPenalty:
    def __init__(self, critic, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)

    def interpolate(self, real, fake):
        eps = self.eps.view([-1] + [1] * (len(real.shape) - 1))
        return (eps * real + (1 - eps) * fake).requires_grad_()

    def __call__(self, real, fake):
        real = [x.detach() for x in real]
        fake = [x.detach() for x in fake]
        self.eps.uniform_(0, 1)
        interp = [self.interpolate(a, b) for a, b in zip(real, fake)]
        grad_d = grad(self.critic(interp),
                      interp,
                      grad_outputs=self.ones,
                      create_graph=True)
        batch_size = real[0].shape[0]
        grad_d = torch.cat([g.view(batch_size, -1) for g in grad_d], 1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        return grad_penalty


class Trainer(object):
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        trans_mnist = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))  
        ])
        self.data = MultiMaskMNISTv2(m1root = args.mnist_path, m2root = args.fashion_path, m1_block_len=args.m_block_len, m2_block_len = args.f_block_len, m1_len_max = args.m_len_max, m2_len_max = args.f_len_max, fix_seed = args.fix_seed, train='train')
        
        self.val_data = MultiMaskMNISTv2(m1root = args.mnist_path, m2root = args.fashion_path, m1_block_len=args.m_block_len, m2_block_len = args.f_block_len,m1_len_max = args.m_len_max, m2_len_max = args.f_len_max, fix_seed = args.fix_seed, train='val')
        
        
        self.data_loader = DataLoader(self.data, batch_size=args.batch_size,shuffle = True, drop_last=True)
        
        self.mask_loader = DataLoader(self.data, batch_size=args.batch_size, shuffle= True, drop_last=True)
        
        self.val_loader = DataLoader(self.val_data, batch_size=args.batch_size, shuffle = False,  drop_last=False)
        
        self.test_data = MultiMaskMNISTv2(m1root = args.mnist_path, m2root = args.fashion_path, m1_block_len=args.m_block_len, m2_block_len = args.f_block_len, m1_len_max = args.m_len_max, m2_len_max = args.f_len_max, fix_seed = args.fix_seed, train='test')
        
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, drop_last=False)
        
        m_decoder = ConvDecoder(args.latent)
        m_encoder = ConvEncoder(args.latent, args.flow, logprob=False)
        self.m_pbigan = PBiGAN(m_encoder, m_decoder, args.aeloss).to(self.device)

        f_decoder = ConvDecoder(args.latent)
        f_encoder = ConvEncoder(args.latent, args.flow, logprob=False)
        self.f_pbigan = PBiGAN(f_encoder, f_decoder, args.aeloss).to(self.device)

#         self.f_net = Net().to(self.device)
#         self.m_net = Net().to(self.device)

        self.m_critic = ConvCritic(args.latent).to(self.device) 
        
        self.f_critic = ConvCritic(args.latent).to(self.device)
        
        self.multi_cls = MMNet(in_channel = 1).to(self.device)
        self.multi_cls.load_state_dict(torch.load(args.pretrain_path))
        self.multi_cls.eval()
        lrate = 1e-4
        self.optim_m_pbigan = optim.Adam(self.m_pbigan.parameters(), lr=lrate, betas=(.5, .9))
        self.optim_f_pbigan = optim.Adam(self.f_pbigan.parameters(), lr=lrate, betas=(.5, .9))

#         self.optim_m_net = optim.Adam(self.m_net.parameters(), lr=lrate)
#         self.optim_f_net = optim.Adam(self.f_net.parameters(), lr= lrate)

        self.optim_m_critic = optim.Adam(self.m_critic.parameters(), lr=lrate, betas=(.5, .9))
        self.optim_f_critic = optim.Adam(self.f_critic.parameters(), lr=lrate, betas=(.5, .9))

        self.m_grad_penalty = GradientPenalty(self.m_critic, args.batch_size)
        self.f_grad_penalty = GradientPenalty(self.f_critic, args.batch_size)
        
        #self.optim_multi = optim.Adam(self.multi_cls.parameters(), lr = lrate)

        self.scheduler = None
        if args.min_lr is not None:
            lr_steps = 10
            step_size = args.epoch // lr_steps
            gamma = (args.min_lr / args.lr)**(1 / lr_steps)
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
            
        self.ae_weight = 0
        self.n_critic = 5
        self.critic_updates = 0
        self.ae_flat = args.ae_flat
        self.usefeat = args.usefeat
        #mask_str = f'{args.mask}_{args.m_block_len}_{args.f_block_len}_{args.block_len_max}'
        path = args.save_foldname
        self.output_dir = Path(args.save_root) / path
        mkdir(self.output_dir)
        print(self.output_dir)
        #torch.manual_seed(args.seed)
        self.loss_breakdown = defaultdict(float)
    
    def train_gaonet(self):
        
        if self.args.pretrain_pbigan is not None:
            model_dict = torch.load(self.args.pretrain_pbigan)
            self.m_critic.load_state_dict(model_dict['m_critic'])
            self.f_pbigan.load_state_dict(model_dict['f_pbigan'])
            #self.multi_cls.load_state_dict(model_dict['multi_cls'])
            #self.f_net.load_state_dict(model_dict['f_net'])
            self.f_critic.load_state_dict(model_dict['f_critic'])
            self.m_pbigan.load_state_dict(model_dict['m_pbigan'])
            #self.m_net.load_state_dict(model_dict['m_net'])
            

        mkdir(self.output_dir /  'code')
        codefile = os.listdir('./')
        for f in codefile:
            try:
                shutil.copy(f,os.path.join(str(self.output_dir) + '/code' ,f))
            except:
                print (f, ' copy failed')

        if self.args.save_interval > 0:
            model_dir = mkdir(self.output_dir / 'model')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.output_dir / 'log.txt'),
                logging.StreamHandler(sys.stdout),
            ],
        )

        with (self.output_dir / 'args.txt').open('w') as f:
            print(pprint.pformat(vars(self.args)), file=f)

        self.vis = Visualizer(self.output_dir)

        test_xlist, test_mask, index, tt_label = iter(self.test_loader).next()
        test_x1, test_x2 = test_xlist[0].to(device), test_xlist[1].to(device)
        test_mask1, test_mask2 = test_mask[0].to(device).float(), test_mask[1].to(device).float()

        bbox1, bbox2 = None, None
        if self.data.m1_mask_loc is not None:
            bbox1 = [self.data.m1_mask_loc[idx] for idx in index] 
        if self.data.m2_mask_loc is not None:
            bbox2 = [self.data.m2_mask_loc[idx] for idx in index] 

        ## -----------------    Train   ----------------------  ##    

        for epoch in range(self.args.start_epoch, self.args.epoch):
            
            self.loss_breakdown = defaultdict(float)

            if epoch > self.ae_flat:
                self.ae_weight = self.args.ae * (epoch - self.ae_flat) / (self.args.epoch - self.ae_flat)
            
            self.train_epoch(epoch) 
            self.val_epoch(epoch, 'val')
            self.val_epoch(epoch, 'tt')
            self.vis.plot_loss(epoch, self.loss_breakdown)
            
            if epoch % self.args.plot_interval == 0:
                with torch.no_grad():
                    self.m_pbigan.eval()
                    self.f_pbigan.eval()
                    #self.m_net.eval()
                    #self.f_net.eval()

                    if self.ae_weight > 0:
                        self.f_pbigan.eval()
                        self.m_pbigan.eval()
                        _, _, f_x_rec, _, _ = self.f_pbigan(test_x2 * test_mask2, test_mask2, None, ae=False)
                        _, _, m_x_rec, _, _ = self.m_pbigan(test_x1 * test_mask1, test_mask1, None, ae=False)
                        if self.args.keepobsv:
                            m_x_rec = m_x_rec * (1 - test_mask1) + test_x1 * test_mask1
                            f_x_rec = f_x_rec * (1 - test_mask2) + test_x2 * test_mask2
                        
                        feat1, pred1, feat2, pred2, out = self.multi_cls(m_x_rec.detach(), f_x_rec.detach())
            
                    else:
                        feat1 = None
                        feat2 = None
                        #test_feat = None

                    z, z_gen, x_rec, x_gen, ae_loss = self.m_pbigan(test_x1 * test_mask1, test_mask1, feat2)
                    fz, fz_gen, fx_rec, fx_gen, fae_loss = self.f_pbigan(test_x2 * test_mask2, test_mask2, feat1)
                    if self.args.keepobsv:
                        x_rec = x_rec * (1 - test_mask1) + test_x1 * test_mask1
                        fx_rec = fx_rec * (1 - test_mask2) + test_x2 * test_mask2
                        
                    feat1, pred1_gen, feat2, pred2_gen, test_out = self.multi_cls(x_rec.detach(), fx_rec.detach())
                    
                    pred_cls = test_out.data.max(1)[1]
                mkdir(self.output_dir / 'txt')
                f = open(str(self.output_dir) + '/txt/' + str(epoch)+ ".txt", 'w')
                for i in range(tt_label.shape[0]):
                    f.write( str(i) + ' ' + str(tt_label[i].numpy()) + ' pred: ' + str(pred_cls[i].data.cpu().numpy()) +'\n')
                f.close()
                self.vis.plot(epoch, test_x1, test_mask1, bbox1, x_rec, x_gen)
                self.vis.plot(epoch + 10000, test_x2, test_mask2, bbox2, fx_rec, fx_gen)
                
            model_dict = {
            'multi_cls': self.multi_cls.state_dict(),
            'f_pbigan': self.f_pbigan.state_dict(),
            #'f_net': self.f_net.state_dict(),
            'f_critic': self.f_critic.state_dict(),
            'm_pbigan': self.m_pbigan.state_dict(),
            #'m_net': self.m_net.state_dict(),
            'm_critic': self.m_critic.state_dict(),
            'history': self.vis.history,
            'epoch': epoch,
            'args': self.args,
            }
            torch.save(model_dict, str(self.output_dir / 'model.pth'))
            
            if self.args.save_interval > 0 and (epoch + 1) % self.args.save_interval == 0:
                torch.save(model_dict, str(model_dir / f'{epoch:04d}.pth'))
            
    
    def train_epoch(self, epoch):
        pred_list, target_list, loss_list = [], [], []
        m_pred_list, f_pred_list = [], []
        for (xlist, masklist, _, label), (_, mask_genlist, _, _) in zip(self.data_loader, self.mask_loader):
            # x1 and x2 are from the same class, x1 is missing larger, x2 is missing less
            x1, x2 = xlist
            mask1, mask2 = masklist  
            mask_gen1, mask_gen2 = mask_genlist
            x1, x2 = x1.to(device), x2.to(device)
            mask1, mask2 = mask1.to(device).float(), mask2.to(device).float()  
            label = label.to(device)
            mask_gen1, mask_gen2 = mask_gen1.to(device).float(), mask_gen2.to(device).float()
            m_rec = self.train_m_pbigan(x1, mask1, mask_gen1, label, x2, mask2)
            f_rec = self.train_f_pbigan(x2, mask2, mask_gen2, label, x1, mask1) 
            
#             self.cycle_m_pbigan(x2, mask2, label, x1, mask1)   # update the generator of m_pbigan 
#             self.cycle_f_pbigan(x1, mask1, label, x2, mask2)   # update the generator of f_pbigan
#             if epoch > self.ae_flat:
#                 self.multi_cls.train()
            
#                 self.f_pbigan.eval()
#                 #self.f_net.eval()
#                 self.f_critic.eval()
#                 self.m_pbigan.eval()
#                 #self.m_net.eval()
#                 self.m_critic.eval()
            if self.args.trclsori:
                feat1, pred1, feat2, pred2, pred = self.multi_cls(x1, x2) 
            else:
                feat1, pred1, feat2, pred2, pred = self.multi_cls(m_rec, f_rec)   

#             loss = min(20 * self.ae_weight, 1) * nn.CrossEntropyLoss()(pred, label) 
#             loss1 = min(20 * self.ae_weight, 1) * nn.CrossEntropyLoss()(pred1, label) 
#             loss2 = min(20 * self.ae_weight, 1) * nn.CrossEntropyLoss()(pred2, label) 
#             self.optim_multi.zero_grad()
#             (loss + loss1 + loss2).backward()
#             torch.nn.utils.clip_grad_norm_(self.multi_cls.parameters(), 5)
#             self.optim_multi.step()
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += label.data.cpu().numpy().tolist()
                
            m_pred_cls = pred1.data.max(1)[1]
            f_pred_cls = pred2.data.max(1)[1]
            m_pred_list += m_pred_cls.data.cpu().numpy().tolist()
            f_pred_list += f_pred_cls.data.cpu().numpy().tolist()

#            loss_list.append(loss.data.cpu().numpy().tolist())
                
        
        accuracy = accuracy_score(target_list, pred_list)
        self.loss_breakdown['acc'] = accuracy
        
        accuracy = accuracy_score(target_list, m_pred_list)
        self.loss_breakdown['macc'] = accuracy
        
        accuracy = accuracy_score(target_list, f_pred_list)
        self.loss_breakdown['facc'] = accuracy
        

        if self.scheduler:
            self.scheduler.step()

    def val_epoch(self, epoch, phase):
        self.multi_cls.eval() 
        self.f_pbigan.eval() 
        #self.f_net.eval() 
        self.f_critic.eval() 
        self.m_pbigan.eval() 
        #self.m_net.eval() 
        self.m_critic.eval() 
        pred_list, target_list, loss_list = [], [], []
        f_pred_list, m_pred_list = [], [] 
        if phase == 'tt':
            loader = self.test_loader
        if phase == 'val':
            loader = self.val_loader
        for batch_idx, (xlist, masklist, _, label) in enumerate(loader):
            x1, x2 = xlist 
            mask1, mask2 = masklist  
            #mask_gen1, mask_gen2 = mask_genlist
            x1, x2 = x1.to(device), x2.to(device)
            mask1, mask2 = mask1.to(device).float(), mask2.to(device).float() 
            
            if epoch > self.ae_flat:
                _, _, m_rec, _, _ = self.m_pbigan(x1 * mask1, mask1, None, ae=False)
                _, _, f_rec, _, _ = self.f_pbigan(x2 * mask2, mask2, None, ae=False)
                if self.args.keepobsv:       
                    m_rec = m_rec * (1 - mask1) + x1 * mask1
                    f_rec = f_rec * (1 - mask2) + x2 * mask2
                    
                feat1, pred1_gen, feat2, pred2_gen, pred_gen = self.multi_cls(m_rec, f_rec)             # here should be careful
                #loss = nn.CrossEntropyLoss()(pred, label) 
                _, _, m_rec, _, _ = self.m_pbigan(x1 * mask1, mask1, feat2, ae=False)
                _, _, f_rec, _, _ = self.f_pbigan(x2 * mask2, mask2, feat1, ae=False)
                
                if self.args.keepobsv: 
                    m_rec = m_rec * (1 - mask1) + x1 * mask1
                    f_rec = f_rec * (1 - mask2) + x2 * mask2
                    
                feat1, pred1_gen, feat2, pred2_gen, pred_gen = self.multi_cls(m_rec, f_rec)
                
                pred_cls = pred_gen.data.max(1)[1]
                m_pred_cls = pred1_gen.data.max(1)[1]
                f_pred_cls = pred2_gen.data.max(1)[1]
                
                pred_list += pred_cls.data.cpu().numpy().tolist()
                m_pred_list += m_pred_cls.data.cpu().numpy().tolist()
                f_pred_list += f_pred_cls.data.cpu().numpy().tolist()
                target_list += label.data.cpu().numpy().tolist()
                
        
        accuracy = accuracy_score(target_list, pred_list)
        self.loss_breakdown[phase + '_acc'] = accuracy
        accuracy = accuracy_score(target_list, m_pred_list)
        self.loss_breakdown[phase + '_macc'] = accuracy
        accuracy = accuracy_score(target_list, f_pred_list)
        self.loss_breakdown[phase + '_facc'] = accuracy
        

    
    def train_m_pbigan(self, x, mask, mask_gen, label, another_x, another_mask):
        
        self.m_pbigan.train()
        #self.m_net.train()
        self.m_critic.train()
        
        self.f_pbigan.eval()
        #self.f_net.eval()
        self.f_critic.eval()
        self.multi_cls.eval()
        
        if self.ae_weight > 0 and self.usefeat:
            self.m_pbigan.eval()
            #self.f_pbigan.eval()
            _, _, m_x_rec, _, _ = self.m_pbigan(x * mask, mask, None, ae=False)
            _, _, f_x_rec, _, _ = self.f_pbigan(another_x * another_mask, another_mask, None, ae=False)
            if self.args.keepobsv: 
                m_x_rec = m_x_rec * (1 - mask) + x * mask
                f_x_rec = f_x_rec * (1 - another_mask) + another_x * another_mask
            feat1, pred1_gen, feat2, pred2_gen, pred_gen = self.multi_cls(m_x_rec.detach(), f_x_rec.detach())
            self.m_pbigan.train()
        else:
            feat2 = None
            
        z_enc, z_gen, x_rec, x_gen, _ = self.m_pbigan(x * mask, mask, feat2, ae=False)
        if self.args.keepobsv: 
            x_rec = x_rec * (1 - mask) + x * mask
        real_score = self.m_critic((x * mask, z_enc)).mean()
        fake_score = self.m_critic((x_gen * mask_gen, z_gen)).mean()
        
        w_dist = real_score - fake_score  # why here does not need a abs()
        D_loss = -w_dist + self.m_grad_penalty((x * mask, z_enc),
                                            (x_gen * mask_gen, z_gen))
        self.optim_m_critic.zero_grad()
        D_loss.backward(retain_graph=True)
        self.optim_m_critic.step()
        
#         if self.ae_weight > 0:
            
#             if self.args.trclsori:
#                 gen_feat, gen_out = self.m_net(x)
#             else:
#                 gen_feat, gen_out = self.m_net(x_rec)
#             cls_loss  = F.nll_loss(F.log_softmax(gen_out, 1), label.cuda())
#             cls_loss = min(20 * self.ae_weight, 1) * cls_loss
#             self.optim_m_net.zero_grad() 
#             cls_loss.backward(retain_graph=True) 
#             self.optim_m_net.step()
# #             self.loss_breakdown['cls_D'] += cls_loss.item()
# #         else:
# #             self.loss_breakdown['cls_D'] += 0

#         #self.loss_breakdown['D'] += D_loss.item() 
        
        self.critic_updates += 1
        
#        gen_feat, gen_out = self.m_net(x_rec)
        
        if self.critic_updates == self.n_critic:
            self.critic_updates = 0
        
            for p in self.m_critic.parameters(): 
                p.requires_grad_(False) 

#             for p in self.m_net.parameters(): 
#                 p.requires_grad_(False) 

            z_enc, z_gen, x_rec, x_gen, ae_loss = self.m_pbigan(x * mask, mask, feat2)
            if self.args.keepobsv: 
                x_rec = x_rec * (1 - mask) + x * mask
            # GAN loss, G
            real_score = self.m_critic((x * mask, z_enc)).mean()
            fake_score = self.m_critic((x_gen * mask_gen, z_gen)).mean()
            G_loss = real_score - fake_score

            # ae loss, when epoch < 100, ae_loss = 0. 
            ae_loss = ae_loss * self.ae_weight
            loss = G_loss + ae_loss
            
            if self.ae_weight > 0 and self.usefeat:
               # print ('feat is not noen')
                cls_loss  = F.nll_loss(F.log_softmax(pred_gen, 1), label.cuda()) + F.nll_loss(F.log_softmax(pred1_gen, 1), label.cuda())
                cls_loss = min(10 * self.ae_weight, 1) * cls_loss
                loss = loss + cls_loss
#                 self.loss_breakdown['cls_G'] += cls_loss.item()
#             else:
#                 self.loss_breakdown['cls_G'] += 0

            self.optim_m_pbigan.zero_grad()
            loss.backward(retain_graph=True) 
            self.optim_m_pbigan.step()

            for p in self.m_critic.parameters():
                p.requires_grad_(True)
#             for p in self.m_net.parameters():
#                 p.requires_grad_(True)

        return x_rec
        
    def train_f_pbigan(self, x, mask, mask_gen, label, another_x, another_mask):
        
        self.f_pbigan.train()
        #self.f_net.train()
        self.f_critic.train()
        
        self.m_pbigan.eval()
        #self.m_net.eval()
        self.m_critic.eval()
        self.multi_cls.eval()
        
        if self.ae_weight > 0 and self.usefeat:
            self.f_pbigan.eval()
            _, _, f_x_rec, _, _ = self.f_pbigan(x * mask, mask, None, ae=False)
            _, _, m_x_rec, _, _ = self.m_pbigan(another_x * another_mask, another_mask, None, ae=False)
            if self.args.keepobsv: 
                f_x_rec = f_x_rec * (1 - mask) + x * mask
                m_x_rec = m_x_rec * (1 - another_mask) + another_x * another_mask
            feat1, pred1_gen, feat2, pred2_gen, pred_gen = self.multi_cls(m_x_rec.detach(), f_x_rec.detach())
            self.f_pbigan.train()
        else:
            feat1 = None
        
        z_enc, z_gen, x_rec, x_gen, _ = self.f_pbigan(x * mask, mask, feat1, ae=False)
        if self.args.keepobsv: 
            x_rec = x_rec * (1 - mask) + x * mask
        real_score = self.f_critic((x * mask, z_enc)).mean()
        fake_score = self.f_critic((x_gen * mask_gen, z_gen)).mean()
        
        w_dist = real_score - fake_score  # why here does not need a abs()
        D_loss = -w_dist + self.f_grad_penalty((x * mask, z_enc), (x_gen * mask_gen, z_gen))
        
        self.optim_f_critic.zero_grad()
        D_loss.backward(retain_graph=True)
        self.optim_f_critic.step()
        
        
        for p in self.f_critic.parameters():
            p.requires_grad_(False)

#         for p in self.f_net.parameters():
#             p.requires_grad_(False)

        z_enc, z_gen, x_rec, x_gen, ae_loss = self.f_pbigan(x * mask, mask, feat1)
        if self.args.keepobsv: 
            x_rec = x_rec * (1 - mask) + x * mask
        # GAN loss, G
        real_score = self.f_critic((x * mask, z_enc)).mean()
        fake_score = self.f_critic((x_gen * mask_gen, z_gen)).mean()
        G_loss = real_score - fake_score

        # ae loss, when epoch < 100, ae_loss = 0. 
        ae_loss = ae_loss * self.ae_weight
        loss = G_loss + ae_loss

        if self.ae_weight > 0 and self.usefeat:
            
            cls_loss  = F.nll_loss(F.log_softmax(pred_gen, 1), label.cuda()) + F.nll_loss(F.log_softmax(pred2_gen, 1), label.cuda())
            cls_loss = min(10 * self.ae_weight, 1) * cls_loss
            loss = loss + cls_loss


        self.optim_f_pbigan.zero_grad()
        loss.backward(retain_graph=True)
        self.optim_f_pbigan.step()



        for p in self.f_critic.parameters():
            p.requires_grad_(True)
#         for p in self.f_net.parameters():
#             p.requires_grad_(True)
        return x_rec
    
    def generate_imgs(self, epoch, save_path):
        print ('---- ---      --- have not keepobsv ----------')
        self.multi_cls.eval()
        self.f_pbigan.eval()
        self.f_net.eval()
        self.f_critic.eval()
        self.m_pbigan.eval()
        self.m_net.eval()
        self.m_critic.eval()
        models = torch.load(str(self.output_dir) + '/model/' +  f'{epoch:04d}.pth')
        self.multi_cls.load_state_dict(models['multi_cls'])
        self.f_pbigan.load_state_dict(models['f_pbigan'])
        self.m_pbigan.load_state_dict(models['m_pbigan'])
        self.f_net.load_state_dict(models['f_net'])
        self.f_critic.load_state_dict(models['f_critic'])
        self.m_net.load_state_dict(models['m_net'])
        self.m_critic.load_state_dict(models['m_critic'])
        pred_list, target_list, loss_list = [], [], []
        
        len_dataset = 60000
        
        alldata = torch.zeros((len_dataset, 28, 28))
        alllabel = torch.zeros(len_dataset)
        
        for batch_idx, (xlist, masklist, _, label) in enumerate(self.data_loader):
            x1, x2 = xlist 
            mask1, mask2 = masklist  
            #mask_gen1, mask_gen2 = mask_genlist
            x1, x2 = x1.to(device), x2.to(device)
            mask1, mask2 = mask1.to(device).float(), mask2.to(device).float() 
            #m_rec = self.train_m_pbigan(x1, mask1, mask_gen1, label, x2, mask2) # response to x1
            #f_rec = self.train_f_pbigan(x2, mask2, mask_gen2, label, x1, mask1) # response to x2
            
            
            _, _, m_rec, _, _ = self.m_pbigan(x1 * mask1, mask1, None, ae=False)
            _, _, f_rec, _, _ = self.f_pbigan(x2 * mask2, mask2, None, ae=False)
            feat, out = self.multi_cls(m_rec, f_rec)             # here should be careful
            #loss = nn.CrossEntropyLoss()(pred, label) 
            _, _, m_rec, _, _ = self.m_pbigan(x1 * mask1, mask1, feat, ae=False)
            _, _, f_rec, _, _ = self.f_pbigan(x2 * mask2, mask2, feat, ae=False)
            _, pred = self.multi_cls(m_rec, f_rec) 
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += label.data.cpu().numpy().tolist()
            try:
                if (batch_idx + 1)* self.args.batch_size >= len_dataset:
                    alldata[batch_idx * self.args.batch_size: len_dataset] = mask2.squeeze(1).data.cpu()
                    alllabel[batch_idx * self.args.batch_size: len_dataset] = label
                else:
                    alldata[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] = mask2.squeeze(1).data.cpu()
                    alllabel[batch_idx * self.args.batch_size: (batch_idx + 1)* self.args.batch_size] = label
            except:
                pdb.set_trace()
        if save_path is not None:     
            torch.save([alldata, alllabel], save_path)
        
        accuracy = accuracy_score(target_list, pred_list)
            #print ('acc: ', accuracy)
        
        print (accuracy)
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3)
    # training options
    parser.add_argument('--mnist_path', type=str, default='/nfs/masi/gaor2/data/mnist/MNIST_trvaltt')
    parser.add_argument('--fashion_path', type=str, default='/nfs/masi/gaor2/data/mnist/fashion_trvaltt')
    parser.add_argument('--pretrain_path', type=str, default='/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/20210115_mmnet/models/model_epoch_0092.pth') # 49 for mnist, 134 for fashion
    parser.add_argument('--pretrain_pbigan', type=str, default=None)
    parser.add_argument('--plot-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--save_root', type=str, default='/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/crossM_cls')
    # mask options (data): block|indep 
    parser.add_argument('--mask', default='block')
    # option for block: set to 0 for variable size
    parser.add_argument('--m-block-len', type=int, default=10)
    parser.add_argument('--f-block-len', type=int, default=20)
    parser.add_argument('--m-len-max', type=int, default=14)
    parser.add_argument('--f-len-max', type=int, default=28)
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2)
    parser.add_argument('--obs-prob-max', type=float, default=None)
    parser.add_argument('--ae_flat', type=int, default=100)
    parser.add_argument('--flow', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=None)

    parser.add_argument('--arch', default='conv')   # fc | conv
    parser.add_argument('--start_epoch', type=int, default= 0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--ae', type=float, default=.1)
    parser.add_argument('--trclsori', type=bool, default=True)
    parser.add_argument('--save_foldname', default='20210216_v2_bothvarlen_pretrain')
    parser.add_argument('--keepobsv', type=bool, default=True)
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--usefeat', type=bool, default=True)
    parser.add_argument('--aeloss', default='mse')   # mse|bce|smooth_l1|l1
    parser.add_argument('--fix_seed', default=False) 
    
    args = parser.parse_args()
    
    trainer = Trainer(args)

    trainer.train_gaonet()
    
#    trainer.generate_imgs(349, save_path = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/crossM_cls/20210120_base/gen_fashion/processed/mask_train.pt')


if __name__ == '__main__':
    main()