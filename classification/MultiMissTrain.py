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

from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

from dataset import MNIST, MultiMNIST, MultiMaskMNIST
from mnist_loader import get_train_valid_loader, get_test_loader
import numpy as np
import pdb


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
    

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        
        if self.cfig['model_name'] == 'bl':
            self.model = Net(in_channel = 1).to(self.device)
            print ('the bl model')
        if self.cfig['model_name'] in ['MMNet', 'MultiMisNet']:
            self.model = MMNet(in_channel = 1).to(self.device)
            print (self.cfig['model_name'] + ' model')
            
        self.train_loader, self.val_loader = get_train_valid_loader(self.cfig['data_root'],
                       self.cfig['batch_size'],
                       model_name = self.cfig['model_name'],                                                m_len = self.cfig['m_len'],
                       f_len = self.cfig['f_len'],                                 
                       random_seed = 1234,
                       augment=False,
                       valid_size=0.01,
                       shuffle=True,
                       show_sample=False,
                       num_workers=4,
                       pin_memory=True)
        self.test_loader = get_test_loader(self.cfig['data_root'], 
                self.cfig['batch_size'],
                model_name = self.cfig['model_name'],                                                 m_len = self.cfig['m_len'],
                f_len = self.cfig['f_len'],          
                shuffle=False,
                num_workers=4,
                pin_memory=True)
        print ('len test_loader: ', len(self.test_loader) )
        print ('len train_loader: ', len(self.train_loader) )
        print ('len val_loader: ', len(self.val_loader) )
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.5, 0.9))
        
        
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
#             if self.cfig['use_rnn']:
#                 sequence_length, input_size = 28, 28
#                 data = data.reshape(-1, sequence_length, input_size)
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            if batch_idx == 0: print (data[0].shape, data[1].shape)
            data[0] = data[0].type(torch.cuda.FloatTensor)
            data[1] = data[1].type(torch.cuda.FloatTensor)

            pred = self.model(data1, data2)             # here should be careful
            

            loss = nn.CrossEntropyLoss()(pred, target) 
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
            
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            #print ('=================',data.shape)
            if batch_idx == 0: print (data[0].shape, data[1].shape)
            
            
            pred = self.model(data1, data2)             # here should be careful
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
            
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            
            
            pred = self.model(data1, data2)             # here should be careful
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
        
    def test(self):
        model_root = osp.join(self.save_path, 'models')
        model_list = os.listdir(model_root)
        Acc, F1, Recl, Prcn = [], [], [], []
        for epoch in range(len(model_list)):
            print ('epoch: ', epoch)
            model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
            accuracy, f1, recall, precision = self.test_epoch(model_pth)
            print (accuracy, f1, recall, precision)
            Acc.append(accuracy)
            F1.append(f1)
            Recl.append(recall)
            Prcn.append(precision)
        data = pd.DataFrame()
        data['accuracy'] = Acc
        data['f1'] = F1
        data['recall'] = Recl
        data['precision'] = Prcn
        print ('Acc: ', Acc)
        print ('f1:', F1)
        print ('Recl', Recl)
        print ('Prcn', Prcn)
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        test_csv = os.path.join(self.csv_path, 'test.csv')
        data.to_csv(test_csv)
              
        
import yaml
import shutil        
if __name__ == '__main__':
    
    cfig = {
    'data_root': ["/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/MissGAN/pretrain_onemiss_mcls/fashion-pbigan/allfeatv2_1231.111205_block_10_None/gen_imgs", "/nfs/masi/gaor2/data/mnist/fashion"], # '../data/nlst_withlabel.csv',
    'save_path': '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/20210121_gen_obvs',
    'max_epoch': 250, 
    'start_epoch': 0, 
    'n_classes': 10, 
    'learning_rate': 0.0001,
    'batch_size': 64,
    'feat_dim': 50, 
    'use_exist_model': True, 
    'save_tensorlog': True,
    'iseval': True,
    'savemodel': False,
    'model_name': 'MultiMisNet',
    'm_len': 28,
    'f_len': 28, 
    }
    
    trainer = Trainer(cfig)
    trainer.train()
