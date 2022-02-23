import os
import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import *
from utils import Meter, cycle,cycleface, save_model
from en_de_coder import *
import shutil

os.makedirs(DATASET_PARAMETERS['save_path'] +  '/code', exist_ok = True)
codefile = os.listdir('./')
for f in codefile:
    try:
        shutil.copy(f,os.path.join(DATASET_PARAMETERS['save_path']+ '/code' ,f))
    except:
        print (f, ' copy failed')

# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num

print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list, 16)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycleface(face_loader))
face_gen_iterator = iter(cycleface(face_loader))

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)
#g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)

decoder = ConvDecoder(NETWORKS_PARAMETERS['p']['latent'])
encoder = ConvEncoder(NETWORKS_PARAMETERS['p']['latent'], NETWORKS_PARAMETERS['p']['flow'], logprob=False)
pbigan = PBiGAN(encoder, decoder).cuda()
p_optimizer = optim.Adam(pbigan.parameters(), lr=0.0002, betas=(.5, .9))

critic = ConvCritic(NETWORKS_PARAMETERS['p']['latent']).cuda()
critic_optimizer = optim.Adam(critic.parameters(), lr=0.0002, betas=(.5, .9))
grad_penalty = GradientPenalty(critic, DATASET_PARAMETERS['batch_size'])

f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)
c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)

# label for real/fake faces
real_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 1)
fake_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 0)

# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_loss_list = Meter('D_loss', 'avg', ':3.2f')
G_loss_list = Meter('G_loss', 'avg', ':3.2f')
C_real_list = Meter('C_real', 'avg', ':3.2f')
ae_loss_list = Meter('ae_loss', 'avg', ':3.2f')
GC_fake_list = Meter('G_C_fake', 'avg', ':3.2f')

print('Training models...')
ae_weight = 0
for it in range(50000):
    # data
    start_time = time.time()
    if it >= 1000:
        ae_weight = NETWORKS_PARAMETERS['p']['aeloss']
    voice, voice_label = next(voice_iterator)
    face, face_label, mask = next(face_iterator)
    face_gen, face_label_gen, mask_gen = next(face_iterator)
    assert not (mask_gen == mask).all()
    noise = 0.05*torch.randn(DATASET_PARAMETERS['batch_size'], 64, 1, 1)

    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']: 
        voice, voice_label = voice.cuda(), voice_label.cuda()
        face, face_label, mask = face.cuda(), face_label.cuda(), mask.cuda()
        face_gen, face_label_gen, mask_gen = face_gen.cuda(), face_label_gen.cuda(), mask_gen.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()
    data_time.update(time.time() - start_time)

    # get embeddings and generated faces
    embeddings = e_net(voice).squeeze()
    embeddings = 10 * F.normalize(embeddings)
    embeddings = torch.cat((embeddings, embeddings), 1)
    # introduce some permutations
#     embeddings = embeddings + noise
#     embeddings = 10 * F.normalize(embeddings)
    
    
    z_enc, z_gen, face_rec, face_gen, _ = pbigan(face, mask, embeddings, ae=False)
    real_score = critic((face * mask, z_enc)).mean()
    fake_score = critic((face_gen * mask_gen, z_gen)).mean()
    w_dist = real_score - fake_score
    D_loss = -w_dist + grad_penalty((face * mask, z_enc),
                                                (face_gen * mask_gen, z_gen))
    
    critic_optimizer.zero_grad()
    D_loss.backward(retain_graph=True)
    D_loss_list.update(D_loss.item())
    critic_optimizer.step()
    
    #fake = g_net(embeddings)
    
    #fake = face * mask + fake * (1 - mask)
    # Discriminator
    f_optimizer.zero_grad()
    #d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    #real_score_out = d_net(f_net(face))
    #fake_score_out = d_net(f_net(fake.detach()))
    real_label_out = c_net(f_net(face_rec))
    #D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label)
    #D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label)
    C_real_loss = 500 * ae_weight * F.nll_loss(F.log_softmax(real_label_out, 1), face_label)
    
    #D_fake.update(D_fake_loss.item())
    C_real_list.update(C_real_loss.item())
    (C_real_loss).backward(retain_graph = True)
    f_optimizer.step()
    #d_optimizer.step()
    c_optimizer.step()

    # Generator
    p_optimizer.zero_grad()
    z_enc, z_gen, face_rec, face_gen, ae_loss = pbigan(face, mask, embeddings, ae= True)
    real_score = critic((face * mask, z_enc)).mean()
    fake_score = critic((face_gen * mask_gen, z_gen)).mean()
    G_loss = real_score - fake_score
    
    ae_loss = ae_loss * ae_weight
    loss = G_loss + ae_loss
    fake_label_out = c_net(f_net(face_rec))
    GC_fake_loss = 500 * ae_weight * F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
    G_loss_list.update(G_loss.item())
    ae_loss_list.update(ae_loss.item())
    GC_fake_list.update(GC_fake_loss.item())
    (loss + GC_fake_loss).backward()
    p_optimizer.step()
    
    
#     g_optimizer.zero_grad()
#     fake_score_out = d_net(f_net(fake))
#     fake_label_out = c_net(f_net(fake))
#     GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label)
#     GC_fake_loss = F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
#     (GD_fake_loss + GC_fake_loss).backward()
#     GD_fake.update(GD_fake_loss.item())
#     GC_fake.update(GC_fake_loss.item())
#     g_optimizer.step()

    batch_time.update(time.time() - start_time)

    # print status
    if it % 200 == 0:
        print(iteration, data_time, batch_time, 
              D_loss_list, G_loss_list, C_real_list, ae_loss_list, GC_fake_list)
        data_time.reset()
        batch_time.reset()
        D_loss_list.reset()
        G_loss_list.reset()
        C_real_list.reset()
        ae_loss_list.reset()
        GC_fake_list.reset()

        # snapshot
        if it % 4000 == 0:
            os.makedirs(NETWORKS_PARAMETERS['p']['model_path'] + '/' + str(it))
            save_model(pbigan, NETWORKS_PARAMETERS['p']['model_path'] + '/' + str(it) + '/pbigan.pth')
            save_model(f_net, NETWORKS_PARAMETERS['f']['model_path']+ '/' + str(it) + '/face_embedding.pth')
        #save_model(d_net, NETWORKS_PARAMETERS['d']['model_path'])
            save_model(c_net, NETWORKS_PARAMETERS['c']['model_path'] + '/' + str(it) + '/classifier.pth')
            save_model(critic, NETWORKS_PARAMETERS['critic']['model_path']+ '/' + str(it) + '/critic.pth')
    iteration.update(1)

