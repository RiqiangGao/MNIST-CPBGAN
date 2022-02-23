import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from data import MNIST, FashionMNIST, IndepMaskedMNIST, BlockMaskedMNIST
from network import *
import time
import pdb
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import yaml

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

f = open('config.yaml', 'r').read()
cfig = yaml.load(f)

print('Initializing networks...')
e_net = Net().cuda()
g_net = Generator().cuda()
f_net1 = Embed().cuda()
f_net2 = Embed().cuda()
d_net = Classifier(output_channel = 1).cuda()
c_net = Classifier(output_channel = 10).cuda()

if cfig['pretrain']:
    e_net.load_state_dict(torch.load('pretrain_model/mnist.pth'))
else:
    e_optimizer = optim.Adam(e_net.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))

g_optimizer = optim.Adam(g_net.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))
f_optimizer1 = optim.Adam(f_net1.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))
f_optimizer2 = optim.Adam(f_net2.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))
d_optimizer = optim.Adam(d_net.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))
c_optimizer = optim.Adam(c_net.parameters(),lr=cfig['lr'],betas=(cfig['beta1'], cfig['beta2']))

if cfig['mask_loader']:
    dataset = BlockMaskedMNIST(data_dir = cfig['mnist_path'], train=True, 
                     transform=transforms.Compose(
            [transforms.Resize(cfig['img_size']), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    MNISTloader = torch.utils.data.DataLoader(dataset, batch_size=cfig['batch_size'],
    shuffle=True,)

else:
    MNISTloader = torch.utils.data.DataLoader(
        MNIST(
            cfig['mnist_path'],
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(cfig['img_size']), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=cfig['batch_size'],
        shuffle=True,
    )


Fashloader = torch.utils.data.DataLoader(
    FashionMNIST(
        cfig['fashion_path'],
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(cfig['img_size']), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=cfig['batch_size'],
    shuffle=True,
)

MNIST_iterator = iter(cycle(MNISTloader))
Fash_iterator = iter(cycle(Fashloader))

real_label = torch.full((cfig['batch_size'], 1), 1)
fake_label = torch.full((cfig['batch_size'], 1), 0)


for iter in range(5000):
    if iter % 10 == 0:
        print ('Iter %d'%(iter))
    
    voice, voice_label = next(MNIST_iterator)
    face, face_label = next(Fash_iterator)
    
    #noise = 0.05*torch.randn(cfig['batch_size'], 64, 1, 1)
    noise = 0.05*torch.randn(cfig['batch_size'], 50)

    # use GPU or not
    if cfig['GPU']: 
        voice, voice_label = voice.cuda(), voice_label.cuda()
        face, face_label = face.cuda(), face_label.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()

    #data_time.update(time.time() - start_time)

    # get embeddings and generated faces
    
    if not cfig['pretrain']:
        e_optimizer.zero_grad()
        embeddings, out = e_net(voice)
        E_loss = F.nll_loss(F.log_softmax(out, 1), voice_label)
        E_loss.backward(retain_graph=True)
        e_optimizer.step()
    else:
        embeddings, out = e_net(voice)
    
    embeddings = F.normalize(embeddings)
    # introduce some permutations
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    fake = g_net(embeddings)
    
#     z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (face.shape[0], 128))))
#     fake = g_net(z)
    
    if iter % 200 == 0:
        f = open(cfig['save_root'] + '/%d.txt'% iter, 'w')
        for i in range(25):
            f.write(str(voice_label[i].data.cpu().numpy()) + '\n')
        f.close()
        save_image(fake.data[:25], cfig['save_root'] + "/%d.png" % iter, nrow=5, normalize=True)

    f_optimizer1.zero_grad() 
    f_optimizer2.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    real_score_out = d_net(f_net1(face))
    fake_score_out = d_net(f_net1(fake.detach()))
    real_label_out = c_net(f_net2(face))

    D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label)
    D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label)
    C_real_loss = F.nll_loss(F.log_softmax(real_label_out, 1), face_label)
#     D_real.update(D_real_loss.item())
#     D_fake.update(D_fake_loss.item())
#     C_real.update(C_real_loss.item())
    (D_real_loss + D_fake_loss + C_real_loss).backward()
#    (D_real_loss + D_fake_loss).backward()
    f_optimizer1.step()
    f_optimizer2.step()
    d_optimizer.step()
    c_optimizer.step()
    
    # Generator
    g_optimizer.zero_grad()
    fake_score_out = d_net(f_net1(fake))
    fake_label_out = c_net(f_net2(fake))
    GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label)
    GC_fake_loss = F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
    (GD_fake_loss + GC_fake_loss).backward()
#    GD_fake_loss.backward()
#     GD_fake.update(GD_fake_loss.item())
#     GC_fake.update(GC_fake_loss.item())
    g_optimizer.step()

#    batch_time.update(time.time() - start_time)
