import torch
import torchvision
import os
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from PIL import Image
import numpy as np
import pdb

def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    
def data_view(data):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    print('Labels: ', labels)
    print('Batch shape: ', images.size())
    show_batch(images)

    
class MultiMaskMNIST(Dataset): # adapted from MNIST2Fashion/classification/dataset.py
    def __init__(self, m1root, m2root, m1_block_len=10, m2_block_len = 15, image_size=28, block_len_max=None, train=True, transform1=transforms.ToTensor(), transform2=transforms.ToTensor(), target_transform=None ):
        
        self.m1root = m1root
        self.m2root = m2root
        self.train = train
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_size = image_size
        self.m1_block_len = m1_block_len
        self.m2_block_len = m2_block_len
        self.block_len_max=block_len_max
        self.rnd = np.random.RandomState(0)
        
        if self.train:
            self.m1train_data, self.m1train_labels = torch.load(
                os.path.join(self.m1root,  'processed/training.pt'))
            self.m2train_data, self.m2train_labels = torch.load(
                os.path.join(self.m2root,  'processed/training.pt'))
        else:
            self.m1test_data, self.m1test_labels = torch.load(
                os.path.join(self.m1root,  'processed/test.pt'))
            self.m2test_data, self.m2test_labels = torch.load(
                os.path.join(self.m2root,  'processed/test.pt'))
            
        self.block_masks()
    
    def block_masks(self):
        d0_len = d1_len = self.image_size
        n_masks = len(self)
        
        self.m1_mask = [None] * n_masks
        self.m1_mask_loc = [None] * n_masks
        
        self.m2_mask = [None] * n_masks
        self.m2_mask_loc = [None] * n_masks
        
        for i in range(n_masks):
            if self.block_len_max is None:
                m1_d0_mask_len = m1_d1_mask_len = self.m1_block_len
                m2_d0_mask_len = m2_d1_mask_len = self.m2_block_len
            else:
                m1_d0_mask_len = self.rnd.randint(
                    self.m1_block_len, self.block_len_max)
                m1_d1_mask_len = self.rnd.randint(
                    self.m1_block_len, self.block_len_max)
                
                m2_d0_mask_len = self.rnd.randint(
                    self.m2_block_len, self.block_len_max)
                m2_d1_mask_len = self.rnd.randint(
                    self.m2_block_len, self.block_len_max)
                

            m1_d0_start = self.rnd.randint(0, d0_len - m1_d0_mask_len + 1)
            m1_d1_start = self.rnd.randint(0, d1_len - m1_d1_mask_len + 1)
            
            m2_d0_start = self.rnd.randint(0, d0_len - m2_d0_mask_len + 1)
            m2_d1_start = self.rnd.randint(0, d1_len - m2_d1_mask_len + 1)

            m1_mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            m2_mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            
            m1_mask[m1_d0_start:(m1_d0_start + m1_d0_mask_len),
                 m1_d1_start:(m1_d1_start + m1_d1_mask_len)] = 1
            
            m2_mask[m2_d0_start:(m2_d0_start + m2_d0_mask_len),
                 m2_d1_start:(m2_d1_start + m2_d1_mask_len)] = 1
            
            self.m1_mask[i] = m1_mask
            self.m2_mask[i] = m2_mask
            
            self.m1_mask_loc[i] = m1_d0_start, m1_d1_start, m1_d0_mask_len, m1_d1_mask_len
            self.m2_mask_loc[i] = m2_d0_start, m2_d1_start, m2_d0_mask_len, m2_d1_mask_len
        
    
    def __getitem__(self, index):
        if self.train:
            img1, target = self.m1train_data[index], self.m1train_labels[index].long()
            mask1 = self.m1_mask[index]
            mask2 = self.m2_mask[index]
            ind2_list = torch.nonzero(self.m2train_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2train_data[index2]
            assert self.m2train_labels[index2] == self.m1train_labels[index].long()
        else:
            img1, target = self.m1test_data[index], self.m1test_labels[index].long()
            mask1 = self.m1_mask[index]
            mask2 = self.m2_mask[index]
            ind2_list = torch.nonzero(self.m2test_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2test_data[index2]
            assert self.m2test_labels[index2] == target
            

        if self.transform1 is not None:
            img1 = Image.fromarray(img1.numpy(), mode='L')
            img1 = self.transform1(img1).numpy()
            
        if self.transform2 is not None:
            img2 = Image.fromarray(img2.squeeze().numpy(), mode='L')
            img2 = self.transform2(img2).numpy()
        
        return [img1 * mask1, img2 * mask2],  target
    
    def __len__(self):
        if self.train:
            return len(self.m1train_data)
        else:
            return len(self.m1test_data)
    
class MaskedMNIST(Dataset):
    def __init__(self, data_dir, train = True, image_size=28, random_seed=0, transform = None, target_transform = None):
        self.rnd = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        self.image_size = image_size
        if image_size == 28:
            self.data = MNIST(
                data_dir, train=train, download=False,
                transform=transforms.ToTensor())
        else:
            self.data = MNIST(
                data_dir, train=train, download=False,
                transform=transforms.Compose([
                    transforms.Resize(image_size), transforms.ToTensor()]))
        self.generate_masks()

    def __getitem__(self, index):
        image, label = self.data[index]
        mask = self.mask[index]
        return image * mask.float(), mask[None], index, label

    def __len__(self):
        return len(self.data)

    def generate_masks(self):
        raise NotImplementedError


class BlockMaskedMNIST(MaskedMNIST):
    def __init__(self, block_len=11, block_len_max=None, *args, **kwargs):
        self.block_len = block_len
        self.block_len_max = block_len_max
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        d0_len = d1_len = self.image_size
        n_masks = len(self)
        self.mask = [None] * n_masks
        self.mask_loc = [None] * n_masks
        for i in range(n_masks):
            if self.block_len_max is None:
                d0_mask_len = d1_mask_len = self.block_len
            else:
                d0_mask_len = self.rnd.randint(
                    self.block_len, self.block_len_max)
                d1_mask_len = self.rnd.randint(
                    self.block_len, self.block_len_max)

            d0_start = self.rnd.randint(0, d0_len - d0_mask_len + 1)
            d1_start = self.rnd.randint(0, d1_len - d1_mask_len + 1)

            mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            mask[d0_start:(d0_start + d0_mask_len),
                 d1_start:(d1_start + d1_mask_len)] = 1
            self.mask[i] = mask
            self.mask_loc[i] = d0_start, d1_start, d0_mask_len, d1_mask_len

class IndepMaskedMNIST(MaskedMNIST):
    def __init__(self, obs_prob=.2, obs_prob_max=None, *args, **kwargs):
        self.prob = obs_prob
        self.prob_max = obs_prob_max
        self.mask_loc = None
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        imsize = self.image_size
        n_masks = len(self)
        self.mask = [None] * n_masks
        for i in range(n_masks):
            if self.prob_max is None:
                p = self.prob
            else:
                p = self.rnd.uniform(self.prob, self.prob_max)
            self.mask[i] = torch.ByteTensor(imsize, imsize).bernoulli_(p)

    
class MultiMNIST(data.Dataset):
    def __init__(self, m1root, m2root, train=True, transform=None, target_transform=None ):
        self.m1root = m1root
        self.m2root = m2root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            self.m1train_data, self.m1train_labels = torch.load(
                os.path.join(self.m1root,  'processed/training.pt'))
            self.m2train_data, self.m2train_labels = torch.load(
                os.path.join(self.m2root,  'processed/training.pt'))
        else:
            self.m1test_data, self.m1test_labels = torch.load(
                os.path.join(self.m1root,  'processed/test.pt'))
            self.m2test_data, self.m2test_labels = torch.load(
                os.path.join(self.m2root,  'processed/test.pt'))
    
    def __getitem__(self, index):
        #torch.manual_seed(0)
        if self.train:
            img1, target = self.m1train_data[index], self.m1train_labels[index]
            ind2_list = torch.nonzero(self.m2train_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2train_data[index2]
            assert self.m2train_labels[index2] == self.m1train_labels[index]
        else:
            img1, target = self.m1test_data[index], self.m1test_labels[index]
            ind2_list = torch.nonzero(self.m2test_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2test_data[index2]
            assert self.m2test_labels[index2] == target
            
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.squeeze().numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return [img1, img2], target
    
    def __len__(self):
        if self.train:
            return len(self.m1train_data)
        else:
            return len(self.m1test_data)
        
class MultiMNISTv2(data.Dataset):
    def __init__(self, m1root, m2root, train=True, transform=None, target_transform=None ):
        self.m1root = m1root
        self.m2root = m2root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train == 'train':
            self.m1train_data, self.m1train_labels = torch.load(
                os.path.join(self.m1root,  'processed/training.pt'))
            self.m2train_data, self.m2train_labels = torch.load(
                os.path.join(self.m2root,  'processed/training.pt'))
        if self.train == 'val':
            self.m1test_data, self.m1test_labels = torch.load(
                os.path.join(self.m1root,  'processed/val.pt'))
            self.m2test_data, self.m2test_labels = torch.load(
                os.path.join(self.m2root,  'processed/val.pt'))
        if self.train == 'test':
            self.m1test_data, self.m1test_labels = torch.load(
                os.path.join(self.m1root,  'processed/test.pt'))
            self.m2test_data, self.m2test_labels = torch.load(
                os.path.join(self.m2root,  'processed/test.pt'))
    
    def __getitem__(self, index):
        #torch.manual_seed(0)
        if self.train == 'train':
            img1, target = self.m1train_data[index], self.m1train_labels[index]
            ind2_list = torch.nonzero(self.m2train_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2train_data[index2]
            assert self.m2train_labels[index2] == self.m1train_labels[index]
        else:
            img1, target = self.m1test_data[index], self.m1test_labels[index]
            ind2_list = torch.nonzero(self.m2test_labels == target)
            index2 = ind2_list[torch.randint(len(ind2_list), (1,)).long()].long()
            img2 = self.m2test_data[index2]
            assert self.m2test_labels[index2] == target
            
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.squeeze().numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return [img1, img2], target
    
    def __len__(self):
        if self.train == 'train':
            return len(self.m1train_data)
        else:
            return len(self.m1test_data)

    
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    
