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
        
        self.upchannel = nn.Sequential(
                nn.Conv2d(1, 256, 3, 2, 0), 
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                
                )
        
        blocks = []
        for i in range(11):
            blocks.append(ResnetBlock(256, 256))
        
        self.resnet = nn.Sequential(*blocks)
        
        self.fcn = nn.Sequential(
                nn.Conv2d(256, 2, 3, 2, 0)
        )
        
        
    def forward(self, x):
        out = self.upchannel(x)
        out = self.resnet(out)
        out = self.fcn(out)
        out = out.view(out.size(0), -1, 2)
        
        return out


