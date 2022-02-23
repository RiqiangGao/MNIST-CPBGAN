import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from masked_mnist import MultiMNIST
import argparse 

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
parser.add_argument('--mnist_path', type=str, default='/nfs/masi/gaor2/data/mnist/MNIST')
parser.add_argument('--fashion_path', type=str, default='/nfs/masi/gaor2/data/mnist/fashion')
parser.add_argument('--pretrain_path', type=str, default='/nfs/masi/gaor2/saved_file/MissClinic/MNIST2Fashion/cls/mnist_128dim/models/model_epoch_0049.pth')
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()


trans_mnist = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
data = MultiMNIST(m1root = args.fashion_path, m2root = args.mnist_path, train=False, transform2=trans_mnist)
data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)

e_net = Net().cuda()
e_net.load_state_dict(torch.load(args.pretrain_path))
e_net.eval()



out_target = []
out_output = []

for batch_idx, (xlist, mask, _, targets) in enumerate(data_loader):
    x2 = xlist[1]
    x2, targets = x2.cuda(), targets.cuda()
    outputs, out = e_net(x2)
    output_np = outputs.data.cpu().numpy()
    target_np = targets.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:,np.newaxis])

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)
np.save('./mnist_test.npy', output_array, allow_pickle=False)
np.save('./mnist_test_label.npy', target_array, allow_pickle=False)

#feature = np.load('./label_smooth1.npy').astype(np.float64)
#target = np.load('./label_smooth_target1.npy')

print('Pred shape :',output_array.shape)
print('Target shape :',target_array.shape)

tsne = TSNE(n_components=2, init='pca', random_state=0)
output_array = tsne.fit_transform(output_array)
plt.rcParams['figure.figsize'] = 10,10
plt.scatter(output_array[:, 0], output_array[:, 1], c= target_array[:,0])
#plt.title(title)
title = 'tSNE'
plt.savefig('./'+title+'.png', bbox_inches='tight')
