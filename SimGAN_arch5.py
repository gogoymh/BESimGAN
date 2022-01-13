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
        
        self.conv1 = nn.Conv2d(1,16,3,1,1) # 16*35*55
        
        self.encoder = nn.Sequential(
                nn.Conv2d(16,16,3,1,1), # 16*35*55
                #nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16,32,3,1,1), # 32*35*55
                #nn.BatchNorm2d(32),
                nn.ELU(),
                
                nn.Conv2d(32,32,3,2,0), # 32*17*27
                
                nn.Conv2d(32,32,3,1,1), # 32*17*27
                #nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32,64,3,1,1), # 64*17*27
                #nn.BatchNorm2d(64),
                nn.ELU(),
                
                nn.Conv2d(64,64,3,2,0), # 64*8*13
                
                nn.Conv2d(64,64,3,1,1), # 64*8*13
                #nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64,64,3,1,1), # 64*8*13
                #nn.BatchNorm2d(64),
                nn.ELU()
                
                )

        self.fc = nn.Sequential(
                nn.Linear(64*8*13, 1000),
                #nn.BatchNorm1d(1000),
                nn.LeakyReLU(),
                
                nn.Linear(1000, 64*8*13),
                #nn.BatchNorm1d(64*8*13),
                nn.LeakyReLU()
                )
        
        self.decoder = nn.Sequential(
                nn.Conv2d(64,64,3,1,1), # 64*8*13
                #nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64,64,3,1,1), # 64*8*13
                #nn.BatchNorm2d(64),
                nn.ELU(),
                
                nn.UpsamplingNearest2d(scale_factor=2), # 64*16*26
                
                nn.Conv2d(64,32,3,1,1), # 32*16*26
                #nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32,32,4,1,2), # 32*17*27
                #nn.BatchNorm2d(32),
                nn.ELU(),
                
                nn.UpsamplingNearest2d(scale_factor=2), # 32*34*54
                
                nn.Conv2d(32,16,3,1,1), # 16*34*54
                #nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16,16,4,1,2), # 16*35*55
                #nn.BatchNorm2d(16),
                nn.ELU()
                )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(16,1,3,1,1), # 16*35*55
                nn.Tanh()
                )
        
    def forward(self, x): # batch*1*35*55
        out = self.conv1(x) # batch*16*35*55
        out = self.encoder(out) # batch*48*8*13
        out = self.fc(out.view(out.size(0),-1)) # batch*48*8*13
        out = self.decoder(out.view(out.size(0), 64, 8, 13)) # batch*16*35*55
        out = self.conv2(out) # batch*1*35*55

        return out


