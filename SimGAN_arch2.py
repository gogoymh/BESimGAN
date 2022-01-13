import torch.nn as nn
#from collections import namedtuple

###########################################################################################################################
class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=32):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features)
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        convs = self.convs(x)
        sum = convs + x
        output = self.relu(sum)
        return output

class Generator(nn.Module):
    def __init__(self, block_num, in_features, nb_features=32):
        super(Generator, self).__init__()

        self.upchannel = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU()
        )
        
        blocks = []
        for i in range(block_num):
            blocks.append(ResnetBlock(nb_features, nb_features))
        
        self.resnet = nn.Sequential(*blocks)
        
        self.downchannel = nn.Sequential(
            nn.Conv2d(nb_features, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.upchannel(x)
        output = self.resnet(output)
        output = self.downchannel(output)
        return output

#D_Outputs = namedtuple('DiscriminatorOuputs', ['aux_img', 'out_cls'])

class Discriminator(nn.Module):
    def __init__(self): #, aux_cls = True
        super(Discriminator, self).__init__()
        
        #self.aux_cls = aux_cls
        
        self.downsample = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1), # 32*35*55
                nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
                nn.Linear(32*35*55, 200),
                nn.BatchNorm1d(200),
                nn.LeakyReLU(),
                
                nn.Linear(200, 32*35*55),
                nn.BatchNorm1d(32*35*55),
                nn.LeakyReLU()
        )
        # Upsampling
        self.upsample = nn.Sequential(
                nn.Conv2d(32, 1, 3, 1, 1) # 1*35*55
        )

        
    def forward(self, x):
        #print(x.shape)
        out = self.downsample(x)
        #print(out.shape)
        out = self.fc(out.view(out.size(0),-1))
        #print(out.shape)
        out = self.upsample(out.view(out.size(0), 32, 35, 55))
        #print(out.shape)
        #print('='*20)
        '''
        if self.aux_cls :
            aux_img = out
        
        out = self.fcn(x)
        out_cls = out.view(out.size(0), -1, 2)
        
        if self.aux_cls :
            return D_Outputs(aux_img, out_cls)
        '''
        return out


