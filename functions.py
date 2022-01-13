from torch.utils.data import Dataset
import os
from skimage import io
from PIL import Image
import numpy as np

#####################################################################################################################
class synthetic2_loader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
    
    def __len__(self):
        num = 60000
        return num
    
    def __getitem__(self, idx):
        file = ("%d" % (idx+1)) + '.jpg'
        img_name = os.path.join(self.path,file)
        
        image = io.imread(img_name)
        image = Image.fromarray(image).convert("L")
        image = np.array(image.getdata())
        
        if (idx+1) % 2 == 0:
            image = np.reshape(image, (600, 800))
            image = np.fliplr(image)
            
        image = np.reshape(image, (600, 800, 1))
        image = image.astype('uint8')
        image = Image.fromarray(image.reshape(600, 800), 'L')
        
        if self.transform:
            image = self.transform(image)
            
        return image


##################################################################################################################################
class real_loader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
    
    def __len__(self):
        num = 10700
        return num
    
    def __getitem__(self, idx):
        file = ("%d" % (idx+1)) + '.png'
        img_name = os.path.join(self.path,file)
        
        image = Image.open(img_name).convert("L")
        image = np.array(image.getdata())
        
        if (idx+1) % 2 == 0:
            image = np.reshape(image, (35, 55))
            image = np.fliplr(image)
        
        image = np.reshape(image, (35, 55, 1))
        image = image.astype('uint8')
        image = Image.fromarray(image.reshape(35, 55), 'L')
        
        if self.transform:
            image = self.transform(image)
            
        return image

