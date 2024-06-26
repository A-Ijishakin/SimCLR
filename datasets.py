import torch 
import torchvision.transforms as tfs
import numpy as np
from glob import glob
from PIL import Image 
import os 
from tqdm import tqdm 
import pandas as pd  
from data_aug.gaussian_blur import GaussianBlur     

class CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode=0, classification=False, ffhq='', s=1, size=224, n_views=2, train=True):
        #filter for those in the training set
        self.datums = pd.read_csv('../celeba.csv')
        self.datums = self.datums[self.datums['set'] == mode]  
        self.ffhq = ffhq 
        #instantiate the base directory 
        self.base = '../img_align_celeba' 
        
        color_jitter = tfs.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s) 
        
        self.train = train 
        
        if train:
            self.base_transform = tfs.Compose([tfs.RandomResizedCrop(size=size),
                                                tfs.RandomHorizontalFlip(),
                                                tfs.RandomApply([color_jitter], p=0.8),
                                                tfs.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                tfs.ToTensor()]) 
            
        else:
            self.base_transform = tfs.Compose([tfs.CenterCrop((size, size)), 
                                                tfs.ToTensor()]) 
        self.n_views = n_views 
    def __len__(self): 
        return len(self.datums) - 1 
    
    def __getitem__(self, idx):
        path = '{}/{}'.format(self.base, 
                self.datums.iloc[idx]['id']) 
        
        x = Image.open(path) 
        
        # Crop the center of the image
        w, h = x.size 
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((256, 256))      
        
        if self.train:                    
            imgs = [self.base_transform(x).to(torch.float32) for i in range(self.n_views)]
        
        else:
            imgs = self.base_transform(x).to(torch.float32) 
        
        labels = torch.tensor(self.datums.iloc[idx].drop(['id', 'set']).values.astype(float))
        
        
        
        return {'imgs':imgs,  
                'index' : idx, 'path': path, 'labels': labels}   
            
    
class FFHQ_HDataset(torch.utils.data.Dataset):
    def __init__(self, s=1, size=224, n_views=2):
        self.base = '/home/rmapaij/HSpace-SAEs/datasets/FFHQ/images/*'
        self.images = glob(self.base) 
        
        
        color_jitter = tfs.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.base_transform = tfs.Compose([tfs.RandomResizedCrop(size=size),
                                              tfs.RandomHorizontalFlip(),
                                              tfs.RandomApply([color_jitter], p=0.8),
                                              tfs.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              tfs.ToTensor()])
        self.n_views = n_views 
        
        
        
    def __len__(self): 
        return len(self.images) - 1 
    
    def __getitem__(self, idx):
        path = self.images[idx] 
        
        x = Image.open(path) 

        # resize the image
        x = x.resize((256, 256))      
                            
        imgs = [self.base_transform(x).to(torch.float32) for i in range(self.n_views)]
        
        return {'imgs':imgs,  
                'index' : idx, 'path': path}   