

#from func.data.data_generator import Dataset
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
#from func.models.Net_3D_conv1 import net_conv1
import pdb
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score


from dataset import MNIST, FashionMNIST
from mnist_loader import get_train_valid_loader, get_test_loader
import numpy as np


class Net(nn.Module):              # Define a class called Net, which makes it easy to call later.
    def __init__(self, in_channel, feat_dim = 50):           #Initialize the value of the instance. These values are generally used by other methods.
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 10, kernel_size=5)   # 2D convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(320, feat_dim)             # fully connected layer
        self.fc2 = nn.Linear(feat_dim, 10)              # fully connected layer

    def forward(self, x):                    # define the network using the module in __init__
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        
        if self.cfig['model_name'] == 'bl':
            self.model = Net(in_channel = 1, feat_dim = cfig['feat_dim']).to(self.device)
            print ('the bl model')
            
        self.train_loader, self.val_loader = get_train_valid_loader(self.cfig['data_root'],
                       self.cfig['batch_size'],
                       model_name = self.cfig['model_name'],                                                                   
                       random_seed = 1234,
                       augment=False,
                       valid_size=0.01,
                       shuffle=True,
                       show_sample=False,
                       num_workers=4,
                       pin_memory=True)
        self.test_loader = get_test_loader(self.cfig['data_root'], 
                self.cfig['batch_size'],
                model_name = self.cfig['model_name'],                                                     
                shuffle=False,
                num_workers=4,
                pin_memory=True)
        print ('len test_loader: ', len(self.test_loader) )
        print ('len train_loader: ', len(self.train_loader) )
        print ('len val_loader: ', len(self.val_loader) )
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999))
        
        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            model_root = osp.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                if self.device == 'cuda': #there is a GPU device
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            else:
                self.train_epoch(epoch)
                if self.cfig['savemodel']:
                    torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                self.eval_epoch(epoch)
                self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ('Epoch: ', epoch)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device).long()
            self.optim.zero_grad()
            if batch_idx == 0: print (data.shape)
            data = data.type(torch.cuda.FloatTensor)
            
            pred = self.model(data)             # here should be careful
            if batch_idx == 0:
                print ('data.shape',data.shape)
                print ('pred.shape', pred.shape)
                print('Epoch: ', epoch)
 
            loss = F.nll_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d, loss=%.4f\n' % (
            epoch, batch_idx, len(self.train_loader), loss.data[0])
            
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            

        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print('------------------', tmp_epoch)
        print ('train accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)

        data['epoch'], data['loss'], data['accuracy'] = tmp_epoch, tmp_loss, tmp_acc
        data.to_csv(train_csv)
        print ('train acc: ', accuracy)
        
        
    def eval_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            
            data, target = data.to(self.device), target.to(self.device).long()
            self.optim.zero_grad()
            #print ('=================',data.shape)
            data = data.type(torch.cuda.FloatTensor)
            
            pred = self.model(data)             # here should be careful
            #loss = self.criterion(pred, target)
            
            loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1]  # not test yet
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print ('------------------', tmp_epoch)
        print ('val accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        
        data['epoch'], data['loss'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_acc
        print ('max val accuracy at: ', max(tmp_acc), tmp_acc.index(max(tmp_acc)))
        data.to_csv(eval_csv)
        
        
            
    def test_epoch(self, epoch):
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ()
        for batch_idx, (data, target) in enumerate(self.test_loader):    # test_loader
            
            data, target = data.to(self.device), target.to(self.device).long()
            self.optim.zero_grad()
            #print ('=================',data.shape)
            data = data.type(torch.cuda.FloatTensor)
            
            pred = self.model(data)             # here should be careful
            #loss = self.criterion(pred, target)
            
            loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1] 
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())

        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        
        print ('test accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        
        data['epoch'], data['loss'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_acc
        
        data.to_csv(eval_csv)
        
    def test(self, model_path):
        model_dict = torch.load(model_path)
        self.model.eval()
        self.model.load_state_dict(model_dict)
        
        pred_list, target_list, loss_list = [],[],[]
        print ()
        for batch_idx, (data, target) in enumerate(self.test_loader):    # test_loader
            
            data, target = data.to(self.device), target.to(self.device).long()
            self.optim.zero_grad()
            #print ('=================',data.shape)
            data = data.type(torch.cuda.FloatTensor)
            
            pred = self.model(data)             # here should be careful
            #loss = self.criterion(pred, target)
            
            loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1] 
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())

        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
    
        print ('test accuracy: ', accuracy)
        
        
import yaml
import shutil        
if __name__ == '__main__':
    cfig = {
    'data_root': '/nfs/masi/gaor2/data/mnist/fashion_trvaltt', #"/nfs/masi/gaor2/data/mnist/MNIST", 
    'save_path': '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/justtest',
    'max_epoch': 150, 
    'start_epoch': 0, 
    'n_classes': 10, 
    'learning_rate': 0.001,
    'batch_size': 256,
    'feat_dim': 128, 
    'use_exist_model': False, 
    'save_tensorlog': True,
    'iseval': True,
    'savemodel': True,
    'model_name': 'bl',

    }
    
    trainer = Trainer(cfig)
   # trainer.test('/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/mnist_128dim/models/model_epoch_0034.pth')
    
#     if not os.path.exists(cfig['save_path'] + '/code'):
#         os.mkdir(cfig['save_path'] + '/code')
    
#     codefile = os.listdir('./')
#     for f in codefile:
#         try:
#             shutil.copy(f,os.path.join(cfig['save_path'] + '/code' ,f))
#         except:
#             print (f, ' copy failed')
    
    
    trainer.train()
