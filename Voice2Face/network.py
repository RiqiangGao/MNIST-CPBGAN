import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PBiGAN(nn.Module):
    def __init__(self, encoder, decoder, ae_loss='bce'):  
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ae_loss = ae_loss

    def forward(self, x, mask, feat, ae=True):
        z_T = self.encoder(x * mask, feat)  # incomplete data encoder
        #if len(z_gen) == 0:
        z_gen = torch.empty_like(z_T).normal_()  # priori distribution encoder, here maybe can transfer from another domain
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
    

class VoiceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x

class Generator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class FaceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[4], output_channel, 4, 1, 0, bias=True),
        )
 
    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x

def get_network(net_type, params, train=True):
    net_params = params[net_type]
    net = net_params['network'](net_params['input_channel'],
                                net_params['channels'],
                                net_params['output_channel'])

    if params['GPU']:
        net.cuda()

    if train:
        net.train()
        optimizer = optim.Adam(net.parameters(),
                               lr=params['lr'],
                               betas=(params['beta1'], params['beta2']))
    else:
        net.eval()
        net.load_state_dict(torch.load(net_params['model_path']))
        optimizer = None
    return net, optimizer
