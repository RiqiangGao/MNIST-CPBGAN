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
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        
        x = F.relu(x)
        feat = F.dropout(x, training=self.training)
        out = self.fc2(feat)
        return feat, out
    
class MMNet(nn.Module):              
    def __init__(self, in_channel):          
        super(MMNet, self).__init__()
        self.fc1_m1 = nn.Linear(320, 50)  
        self.fc1_m2 = nn.Linear(320, 50)  
        
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
        
        self.fc2 = nn.Linear(50, 10)  
        
    def forward(self, x1, x2):      # define the network using the module in __init__
        x1 = self.model1(x1)
        x1 = x1.view(-1, 320)
        x1 = self.fc1_m1(x1)
        
        x2 = self.model2(x2)
        x2 = x2.view(-1, 320)
        x2 = self.fc1_m2(x2)
        
        x = torch.mean(torch.stack([x1, x2]), 0)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

def train_gaonet(args):
    torch.manual_seed(args.seed)
    
    ## -----------------      DATA LOADER ----------------------  ##

    trans_mnist = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    
    data = MultiMNIST(m1root = args.mnist_path, m2root = args.fashion_path, block_len = args.block_len, train=True, transform2=trans_mnist, fix_seed = args.fix_seed)
    val_data = MultiMNIST(m1root = args.mnist_path, m2root = args.fashion_path, block_len = args.block_len, train=True, transform2=trans_mnist, fix_seed = args.fix_seed)
    
    indices = list(range(60000))
    #if shuffle == True:
    np.random.seed(0)
    np.random.shuffle(indices)

    #train_idx, valid_idx = indices[10000:], indices[:10000]
    train_idx, valid_idx = indices[:50000], indices[50000:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    
    mask_str = f'{args.mask}_{args.block_len}_{args.block_len_max}'
    data_loader = DataLoader(data, batch_size=args.batch_size,sampler=train_sampler, drop_last=True)
    
    mask_loader = DataLoader(data, batch_size=args.batch_size,sampler=train_sampler, drop_last=True)
    
    val_loader = DataLoader(data, batch_size=args.batch_size,sampler=valid_sampler, drop_last=True)

    # Evaluate the training progress using 2000 examples from the training data
    test_data = MultiMNIST(m1root = args.mnist_path, m2root = args.fashion_path, block_len = args.block_len, train=False,transform2=trans_mnist, fix_seed = args.fix_seed)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=False)
    
    ## -----------------   Network ----------------------  ##

    decoder = ConvDecoder(args.latent)
    encoder = ConvEncoder(args.latent, args.flow, logprob=False)
    pbigan = PBiGAN(encoder, decoder, args.aeloss).to(device)
    
    e_net = Net().cuda()
    e_net.load_state_dict(torch.load(args.pretrain_path))
    for p in e_net.parameters():
        p.requires_grad_(False)
    e_net.eval()
    
    cls_net = Net().cuda()
    
    cls_multinet = MMNet(in_channel = 1).to(device)

    critic = ConvCritic(args.latent).to(device)  # think more if need two versions
    
    ## -----------------     optimizer   ----------------------  ##

    lrate = 1e-4
    optimizer = optim.Adam(pbigan.parameters(), lr=lrate) # , betas=(.5, .9)
    optimizer_cls = optim.Adam(cls_net.parameters(), lr=lrate) # , betas=(.5, .9)
    optim_mcls = optim.Adam(cls_multinet.parameters(), lr=lrate) # , betas=(.5, .9)

    critic_optimizer = optim.Adam(critic.parameters(), lr=lrate) # , betas=(.5, .9)

    grad_penalty = GradientPenalty(critic, args.batch_size)

    scheduler = None
    if args.min_lr is not None:
        lr_steps = 10
        step_size = max(2000, args.epoch) // lr_steps
        gamma = (args.min_lr / args.lr)**(1 / lr_steps)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    path = args.save_foldname
    output_dir = Path(args.save_root) / path
    mkdir(output_dir)
    print(output_dir)
    
    mkdir(output_dir / 'code')
    codefile = os.listdir('./')
    for f in codefile:
        try:
            shutil.copy(f,os.path.join(str(output_dir) + '/code' ,f))
        except:
            print (f, ' copy failed')

    if args.save_interval > 0:
        model_dir = mkdir(output_dir / 'model')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(output_dir / 'log.txt'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    with (output_dir / 'args.txt').open('w') as f:
        print(pprint.pformat(vars(args)), file=f)

    vis = Visualizer(output_dir)

    test_xlist, test_mask, tt_index, tt_label = iter(test_loader).next()
    test_x, test_x2 = test_xlist[0].to(device), test_xlist[1].to(device)
    test_mask = test_mask.to(device).float()
    
    train_xlist, train_mask, index, tr_label = iter(data_loader).next()
    train_x, train_x2 = train_xlist[0].to(device), train_xlist[1].to(device)
    train_mask = train_mask.to(device).float()
    

    #test_feat = args.featwei * test_feat + (1 - args.featwei) * torch.empty_like(test_feat).normal_()
    bbox = None
    if data.mask_loc is not None:
        bbox = [data.mask_loc[idx] for idx in index]
    test_bbox = None
    if test_data.mask_loc is not None:
        test_bbox = [test_data.mask_loc[idx] for idx in tt_index]
        
    if args.pretrain_pbigan is not None:
        model_dict = torch.load(args.pretrain_pbigan)
        pbigan.load_state_dict(model_dict['pbigan'])
        critic.load_state_dict(model_dict['critic'])
        cls_net.load_state_dict(model_dict['clsnet'])
        cls_multinet.load_state_dict(model_dict['cls_multinet'])
        
    ## -----------------    Train   ----------------------  ##    

    n_critic = 5
    critic_updates = 0
    ae_weight = 0
    ae_flat = 100

    for epoch in range(args.start_epoch, args.epoch):
        loss_breakdown = defaultdict(float)

        if epoch > ae_flat:
            ae_weight = args.ae * (epoch - ae_flat) / (args.epoch - ae_flat)
        tar_list, bothpred_list, m_pred_list = [], [], []
        for (xlist, mask, _, label), (_, mask_gen, _, _) in zip(data_loader, mask_loader):
            # data loader and mask_loader are from the same class
            x, x2 = xlist
            x = x.to(device)
            mask = mask.to(device).float()   # the mask and mask_gen are from the same distribution, but with different value. 
            mask_gen = mask_gen.to(device).float()
            label = label.to(device)
            x2 = x2.to(device)
            if ae_weight > 0 and args.usefeat:
                #print ('feat is not none')
                feat, out = e_net(x2)
            else:
                feat = None
            


            z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask,feat, ae=False)
            if args.keepobsv:
                x_rec = x * mask + (1 - mask) * x_rec
            
            real_score = critic((x * mask, z_enc)).mean()
            fake_score = critic((x_gen * mask_gen, z_gen)).mean()
            
            # GAN loss, D
            w_dist = real_score - fake_score  # why here does not need a abs()
            D_loss = -w_dist + grad_penalty((x * mask, z_enc),
                                            (x_gen * mask_gen, z_gen))
            
            critic_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            critic_optimizer.step()
            
            if ae_weight > 0:
                #pdb.set_trace()

                ori_feat, ori_out = cls_net(x)

                gen_feat, gen_out = cls_net(x_rec)
                if args.trclsori:
                    cls_loss  = F.nll_loss(F.log_softmax(ori_out, 1), label)
                else:
                    cls_loss  = F.nll_loss(F.log_softmax(gen_out, 1), label)
                    
                cls_loss = min(40 * ae_weight, 1) * cls_loss
                optimizer_cls.zero_grad()
                cls_loss.backward()
                optimizer_cls.step()
                
                if args.trclsori:
                    bothpred = cls_multinet(x, x2)
                else:
                    bothpred = cls_multinet(x_rec.detach(), x2)
                
                mcls_loss = min(40 * ae_weight, 1) * nn.CrossEntropyLoss()(bothpred, label) 
                optim_mcls.zero_grad()
                mcls_loss.backward()
                optim_mcls.step()
                
                m_pred_cls = gen_out.data.max(1)[1]
                m_pred_list += m_pred_cls.data.cpu().numpy().tolist()
                tar_list += label.data.cpu().numpy().tolist()
                pred_cls = bothpred.data.max(1)[1]
                bothpred_list += pred_cls.data.cpu().numpy().tolist()
                
                
                #loss_breakdown['cls_D'] += cls_loss.item()
#             else:
#                 loss_breakdown['cls_D'] += 0

            loss_breakdown['D'] += D_loss.item()
            

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in critic.parameters():
                    p.requires_grad_(False)
                    
                for p in cls_net.parameters():
                    p.requires_grad_(False)

                z_enc, z_gen, x_rec, x_gen, ae_loss = pbigan(x * mask, mask, feat)
                if args.keepobsv:
                    x_rec = x * mask + (1 - mask) * x_rec
                # GAN loss, G
                real_score = critic((x * mask, z_enc)).mean()
                fake_score = critic((x_gen * mask_gen, z_gen)).mean()
                G_loss = real_score - fake_score
                
                # ae loss, when epoch < 100, ae_loss = 0. 
                ae_loss = ae_loss * ae_weight
                loss = G_loss + ae_loss

                if ae_weight > 0 and args.usefeat:
                   # print ('feat is not noen')
                    
                    gen_feat, gen_out = cls_net(x_rec)
                    cls_loss  = F.nll_loss(F.log_softmax(gen_out, 1), label)
                    cls_loss = min(40 * ae_weight, 1) * cls_loss
                    loss = loss + cls_loss
#                     loss_breakdown['cls_G'] += cls_loss.item()
#                 else:
#                     loss_breakdown['cls_G'] += 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_breakdown['G'] += G_loss.item()
                loss_breakdown['AE'] += ae_loss.item()
                loss_breakdown['total'] += loss.item()

                for p in critic.parameters():
                    p.requires_grad_(True)
                    
                for p in cls_net.parameters():
                    p.requires_grad_(True)
              
        if ae_weight <= 0:
            loss_breakdown['acc'] = 0
            loss_breakdown['m_acc'] = 0
        else:
            accuracy=accuracy_score(tar_list,bothpred_list)
            loss_breakdown['acc'] = accuracy
            accuracy=accuracy_score(tar_list,m_pred_list)
            loss_breakdown['m_acc'] = accuracy
        
        tar_list, bothpred_list, m_pred_list = [], [], []
        if ae_weight <= 0:
            loss_breakdown['testacc'] = 0
            loss_breakdown['testmacc'] = 0
        else:
            for batch_idx, (xlist, mask, _, label) in enumerate(test_loader):
                x, x2 = xlist
                x = x.to(device)
                x2 = x2.to(device)
                mask = mask.to(device).float()
                
                cls_multinet.eval()
                pbigan.eval()

                    #print ('feat is not none')
                if args.usefeat:
                    
                    feat, out = e_net(x2)
                else:
                    feat = None

                z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, feat, ae=False)
                
                if args.keepobsv:
                    x_rec = x * mask + (1 - mask) * x_rec
                
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
            loss_breakdown['testacc'] = accuracy
            accuracy=accuracy_score(tar_list,m_pred_list)
            loss_breakdown['testmacc'] = accuracy
            
        tar_list, bothpred_list, m_pred_list = [], [], []
        if ae_weight <= 0:
            loss_breakdown['valacc'] = 0
            loss_breakdown['valmacc'] = 0
        else:
            for batch_idx, (xlist, mask, _, label) in enumerate(val_loader):
                x, x2 = xlist
                x = x.to(device)
                mask = mask.to(device).float()
                
                cls_multinet.eval()
                pbigan.eval()

                    #print ('feat is not none')
                x2 = x2.to(device)
                feat, out = e_net(x2)

                z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask, feat, ae=False)
                
                if args.keepobsv:
                    x_rec = x * mask + (1 - mask) * x_rec
                
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
            loss_breakdown['valacc'] = accuracy
            accuracy=accuracy_score(tar_list,m_pred_list)
            loss_breakdown['valmacc'] = accuracy

        if scheduler:
            scheduler.step()

        vis.plot_loss(epoch, loss_breakdown)

        if epoch % args.plot_interval == 0:
            with torch.no_grad():
                pbigan.eval()
                cls_net.eval()
                if ae_weight > 0:
                    train_feat, _ =  e_net(train_x2)
                    test_feat, _ = e_net(test_x2)
                else:
                    train_feat = None
                    test_feat = None
                _, _, tr_x_rec, tr_x_gen, _ = pbigan(train_x * train_mask, train_mask, train_feat)
                z, z_gen, x_rec, x_gen, ae_loss = pbigan(test_x * test_mask, test_mask, test_feat)
                if args.keepobsv:
                    tr_x_rec = train_x * train_mask + (1 - train_mask) * tr_x_rec
                    x_rec = test_x * test_mask + (1 - test_mask) * x_rec
                
                if ae_weight > 0:
                  #  print ('feat is not none')
                    _, test_out = cls_net(x_rec)
                else:
                    _, test_out = cls_net(x_rec)
                    test_feat = None
                
                cls_net.train()
                pbigan.train()
                pred_cls = test_out.data.max(1)[1]
                
            mkdir(output_dir / 'txt')
            f = open(str(output_dir) + '/txt/' + str(epoch)+ ".txt", 'w')
            for i in range(tt_label.shape[0]):

                f.write( str(i) + ' ' + str(tt_label[i].numpy()) + ' pred: ' + str(pred_cls[i].data.cpu().numpy()) +'\n')
            f.close()
            
            vis.plot(1000 + epoch, train_x, train_mask, bbox, tr_x_rec, tr_x_gen)
            vis.plot(epoch, test_x, test_mask, test_bbox, x_rec, x_gen)

        model_dict = {
            'pbigan': pbigan.state_dict(),
            'critic': critic.state_dict(),
            'clsnet': cls_net.state_dict(),
            'cls_multinet': cls_multinet.state_dict(),
            'history': vis.history,
            'epoch': epoch,
            'args': args,
        }
        torch.save(model_dict, str(output_dir / 'model.pth'))
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            torch.save(model_dict, str(model_dir / f'{epoch:04d}.pth'))

    print(output_dir)
    
def generate_imgs(args, model_path, save_path):
    trans_mnist = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    data = MultiMNIST(m1root = args.mnist_path, m2root = args.fashion_path, block_len = args.block_len, train=True, transform2 = trans_mnist)
    
    mask_str = f'{args.mask}_{args.block_len}_{args.block_len_max}'
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    mask_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                             drop_last=False)

    # Evaluate the training progress using 2000 examples from the training data
#     test_data = MultiMNIST(m1root = args.mnist_path, m2root = args.fashion_path, block_len = args.block_len, train=False)
#     test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=False)
    
    ## -----------------   Network ----------------------  ##
    
    decoder = ConvDecoder(args.latent)
    encoder = ConvEncoder(args.latent, args.flow, logprob=False)
    pbigan = PBiGAN(encoder, decoder, args.aeloss).to(device)
    
    pbigan.eval()
    
    trained_model = torch.load(model_path)
    pbigan.load_state_dict(trained_model['pbigan'])
    e_net = Net().cuda()
    e_net.load_state_dict(torch.load(args.pretrain_path))
    for p in e_net.parameters():
        p.requires_grad_(False)
    e_net.eval()
    
    
    cls_net = Net().cuda()
    
    print (len(data), 'len of data')
    alldata = torch.zeros((len(data), 28, 28))
    alllabel = torch.zeros(len(data))
    
    for batch_idx, (xlist, mask, _, label) in enumerate(data_loader):
        x, x2 = xlist
        x = x.to(device)
        mask = mask.to(device).float()

#         cls_multinet.eval()
#         pbigan.eval()

            #print ('feat is not none')
        x2 = x2.to(device)
        feat, out = e_net(x2)

        z_enc, z_gen, x_rec, x_gen, _ = pbigan(x * mask, mask,feat, ae=False)  
        #print (x_rec.shape)
        if (batch_idx + 1)* args.batch_size >= len(data):
            alldata[batch_idx * args.batch_size: len(data)] = x_rec.squeeze(1).data.cpu()
            alllabel[batch_idx * args.batch_size: len(data)] = label
        else:
            alldata[batch_idx * args.batch_size: (batch_idx + 1)* args.batch_size] = x_rec.squeeze(1).data.cpu()
            alllabel[batch_idx * args.batch_size: (batch_idx + 1)* args.batch_size] = label
            
    torch.save([alldata, alllabel], save_path)
#         bothpred = cls_multinet(x_rec, x2)
#         cls_multinet.train()
#         pbigan.train()

#         tar_list += label.data.cpu().numpy().tolist()
#         pred_cls = bothpred.data.max(1)[1]
#         bothpred_list += pred_cls.data.cpu().numpy().tolist()

#     accuracy=accuracy_score(tar_list,bothpred_list)
#     loss_breakdown['testacc'] = accuracy
    
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3)
    # training options
    parser.add_argument('--mnist_path', type=str, default='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/mnist/MNIST')
    parser.add_argument('--fashion_path', type=str, default='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/mnist/fashion')
    parser.add_argument('--pretrain_path', type=str, default='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/MissClinic/cls/fashion_128dim/models/model_epoch_0134.pth') # 49 for mnist, 134 for fashion
    parser.add_argument('--pretrain_pbigan', type=str, default='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/MissClinic/MissGAN/pretrain_onemiss_mcls/20210214_withobsv_trclsori_nofeat/model_backup/0509.pth')
    parser.add_argument('--plot-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--save_root', type=str, default='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/MissClinic/MissGAN/pretrain_onemiss_mcls')
    # mask options (data): block|indep
    parser.add_argument('--mask', default='block')
    # option for block: set to 0 for variable size
    parser.add_argument('--block-len', type=int, default=10)
    parser.add_argument('--block-len-max', type=int, default=None)
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2)
    parser.add_argument('--obs-prob-max', type=float, default=None)

    parser.add_argument('--flow', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=None)

    parser.add_argument('--arch', default='conv')   # fc | conv
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--ae', type=float, default=.1)
    parser.add_argument('--keepobsv', type=bool, default=True)
    parser.add_argument('--trclsori', type=bool, default=True)
    parser.add_argument('--usefeat', type=bool, default=False)
    parser.add_argument('--save_foldname', default='20210214_withobsv_trclsori_nofeat')
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--aeloss', default='bce')   # mse|bce|smooth_l1|l1
    parser.add_argument('--fix_seed', default=False) 

    args = parser.parse_args()
    
#     generate_imgs(args, model_path = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/fashion-pbigan/allfeatv2_1231.111205_block_10_None/model/0639.pth', save_path = '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/fashion-pbigan/allfeatv2_1231.111205_block_10_None/gen_imgs/train_0639.pt')

    train_gaonet(args)


if __name__ == '__main__':
    main()
