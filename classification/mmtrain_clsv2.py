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

from dataset import MNIST, MultiMNIST, MultiMaskMNIST, MultiMNISTv2
from mnist_loader import get_train_valid_loader, get_test_loader
import numpy as np


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
        return F.log_softmax(pred1, dim=1), F.log_softmax(pred2, dim=1), F.log_softmax(x, dim=1)
    

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')

        self.model = MMNet(in_channel = 1).to(self.device)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
       #     normalize
        ])
        if self.cfig['augment']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            #    normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
           #     normalize
            ])    
            
        trainset = MultiMNISTv2(m1root=self.cfig['data_root'][0], m2root=self.cfig['data_root'][1], train='train', transform=train_transform)
        valset = MultiMNISTv2(m1root=self.cfig['data_root'][0], m2root=self.cfig['data_root'][1], train='val', transform=test_transform)
        testset = MultiMNISTv2(m1root=self.cfig['data_root'][0], m2root=self.cfig['data_root'][1], train='test', transform=test_transform)
        
        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=self.cfig['batch_size'], 
                                              shuffle=True, 
                                              num_workers=4,
                                              pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(valset, 
                                              batch_size=self.cfig['batch_size'], 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                              batch_size=self.cfig['batch_size'], 
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
            

            self.train_epoch(epoch)
                
            torch.save(self.model.state_dict(), model_pth)

            self.eval_epoch(epoch)
            self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ('Epoch: ', epoch)
        pred1_list, pred2_list = [], []
        for batch_idx, (data, target) in enumerate(self.train_loader):
#             if self.cfig['use_rnn']:
#                 sequence_length, input_size = 28, 28
#                 data = data.reshape(-1, sequence_length, input_size)
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            if batch_idx == 0: print (data[0].shape, data[1].shape)
            data[0] = data[0].type(torch.cuda.FloatTensor)
            data[1] = data[1].type(torch.cuda.FloatTensor)
            
            pred1, pred2, pred = self.model(data1, data2)             # here should be careful
            

            loss = nn.CrossEntropyLoss()(pred, target) 
            loss1 = nn.CrossEntropyLoss()(pred1, target) 
            loss2 = nn.CrossEntropyLoss()(pred2, target)
            (loss + loss1 + loss2).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()
            
            
            pred1_cls = pred1.data.max(1)[1]  # not test yet
            pred1_list += pred1_cls.data.cpu().numpy().tolist()
            
            pred2_cls = pred2.data.max(1)[1]  # not test yet
            pred2_list += pred2_cls.data.cpu().numpy().tolist()
            
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            
        accuracy1=accuracy_score(target_list,pred1_list)
        accuracy2=accuracy_score(target_list,pred2_list)
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

        
        
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)

        data['epoch'], data['accuracy'] = tmp_epoch, tmp_acc
        data.to_csv(train_csv)
        print ('train acc: ', accuracy, accuracy1, accuracy2)
        
            
        
    def eval_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, target_list, loss_list = [],[],[]
        pred1_list, pred2_list = [], []
        for batch_idx, (data, target) in enumerate(self.val_loader):
            
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            #print ('=================',data.shape)
            if batch_idx == 0: print (data[0].shape, data[1].shape)
            
            
            pred1, pred2, pred = self.model(data1, data2)             # here should be careful
            #loss = self.criterion(pred, target)
            
            #loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1]  # not test yet
            pred_list += pred_cls.data.cpu().numpy().tolist()
            
            pred1_cls = pred1.data.max(1)[1]  # not test yet
            pred1_list += pred1_cls.data.cpu().numpy().tolist()
            
            pred2_cls = pred2.data.max(1)[1]  # not test yet
            pred2_list += pred2_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            
            
        accuracy=accuracy_score(target_list,pred_list)
        accuracy1=accuracy_score(target_list,pred1_list)
        accuracy2=accuracy_score(target_list,pred2_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'accuracy', 'accuracy1', 'accuracy2']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print ('------------------', tmp_epoch)
        print ('val accuracy: ', accuracy, accuracy1, accuracy2)
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_acc1 = df['accuracy1'].tolist()
        tmp_acc1.append(accuracy1)
        tmp_acc2 = df['accuracy2'].tolist()
        tmp_acc2.append(accuracy2)
        
        
        data['epoch'], data['accuracy'],data['accuracy1'],data['accuracy2'] =tmp_epoch, tmp_acc, tmp_acc1, tmp_acc2 
        #print ('max val accuracy at: ', max(tmp_acc), tmp_acc.index(max(tmp_acc)))
        data.to_csv(eval_csv)
        
        
            
    def test_epoch(self, epoch):
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list = [],[],[]
        pred1_list, pred2_list = [], []
        for batch_idx, (data, target) in enumerate(self.test_loader):    
            
            data1, data2, target = data[0].to(self.device), data[1].to(self.device), target.to(self.device)
            self.optim.zero_grad()
            
            
            pred1, pred2, pred = self.model(data1, data2)             # here should be careful
            #loss = self.criterion(pred, target)
            
            #loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1]  # not test yet
            pred_list += pred_cls.data.cpu().numpy().tolist()
            
            pred1_cls = pred1.data.max(1)[1]  # not test yet
            pred1_list += pred1_cls.data.cpu().numpy().tolist()
            
            pred2_cls = pred2.data.max(1)[1]  # not test yet
            pred2_list += pred2_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            
            
        accuracy=accuracy_score(target_list,pred_list)
        accuracy1=accuracy_score(target_list,pred1_list)
        accuracy2=accuracy_score(target_list,pred2_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'accuracy', 'accuracy1', 'accuracy2']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print ('------------------', tmp_epoch)
        print ('test accuracy: ', accuracy, accuracy1, accuracy2)
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_acc1 = df['accuracy1'].tolist()
        tmp_acc1.append(accuracy1)
        tmp_acc2 = df['accuracy2'].tolist()
        tmp_acc2.append(accuracy2)
        
        
        data['epoch'], data['accuracy'],data['accuracy1'],data['accuracy2'] =tmp_epoch, tmp_acc, tmp_acc1, tmp_acc2 
        #print ('max val accuracy at: ', max(tmp_acc), tmp_acc.index(max(tmp_acc)))
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
    'data_root': ["/nfs/masi/gaor2/data/mnist/MNIST_trvaltt", "/nfs/masi/gaor2/data/mnist/fashion_trvaltt"],
    'save_path': '/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/20210115_mmnet', 
    'max_epoch': 100,
    'start_epoch': 0, 
    'n_classes': 10, 
    'learning_rate': 0.001,
    'batch_size': 128, 
    'feat_dim': 50, 
    'augment': True, 
    'iseval': True, 
    
    }
    if not os.path.exists(cfig['save_path'] + '/code'):
        os.mkdir(cfig['save_path'] + '/code')
    
    codefile = os.listdir('./')
    for f in codefile:
        try:
            shutil.copy(f,os.path.join(cfig['save_path'] + '/code' ,f))
        except:
            print (f, ' copy failed')
    
    trainer = Trainer(cfig)
    trainer.train()
