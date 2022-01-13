import argparse
import os
import csv
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.autograd import Variable
import torch
import timeit

######################################################################################################################
import sys
sys.path.append("C:/유민형/개인 연구/BESimGAN/scripts")
from SimGAN_arch4 import Generator, Discriminator
import functions as fn
from image_history_buffer import ImageHistoryBuffer

######################################################################################################################
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=59, help="number of experiment")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample_interval", type=int, default=10, help="number of sampling images")
#parser.add_argument("--delta", type=float, default=0.01, help="Scale factor of refine loss")

opt = parser.parse_args()
print(opt)

# Initialize generator and discriminator ############################################################################
generator = Generator(8, 1, 64).cuda()
discriminator = Discriminator().cuda()

# Configure data loader #############################################################################################
synthetic_data = fn.synthetic2_loader("C:/유민형/개인 연구/BESimGAN/Dataset/UnityEyes_Windows/imgs3",
                               transform=transforms.Compose([
                                       transforms.CenterCrop((180,250)),
                                       transforms.Resize((35,55)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,),(0.5,))]))

synthetic = DataLoader(synthetic_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

real_data = fn.real_loader("C:/유민형/개인 연구/BESimGAN/Dataset/real10",
                            transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))]))

real = DataLoader(real_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

# Optimizers #######################################################################################################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

# Criteria #########################################################################################################
self_regularization_loss = nn.L1Loss(size_average=False)
local_adv_loss = nn.CrossEntropyLoss()

# Train ###########################################################################################################
image_path = "C:/유민형/개인 연구/BESimGAN/results/images/exp%d" % opt.exp
if not os.path.isdir(image_path):
    os.mkdir(image_path)
    
loss_path = "C:/유민형/개인 연구/BESimGAN/results/losses/exp%d" % opt.exp
models_path = "C:/유민형/개인 연구/BESimGAN/results/models/exp%d" % opt.exp
if not os.path.isdir(models_path):
    os.mkdir(models_path)
if not os.path.isdir(loss_path):
    os.mkdir(loss_path)
tot_g_loss = torch.zeros((opt.sample_interval))
tot_d_loss = torch.zeros((opt.sample_interval))

image_history_buffer = ImageHistoryBuffer((0, 1, 35, 55), 120000, opt.batch_size)
print("="*120)

ones = Variable(torch.ones(1664).type(torch.LongTensor)).cuda()
zeros = Variable(torch.zeros(1664).type(torch.LongTensor)).cuda()

# pre-train #######################################################################################################
pre_start = timeit.default_timer()
for j in range(1000):
    syn_imgs = Variable(synthetic.__iter__().next()).float().cuda()
    
    # --------- Refiner --------- #            
    optimizer_G.zero_grad()
            
    ref_imgs = generator(syn_imgs)
                
    D_ref_imgs = discriminator(ref_imgs).view(-1, 2)

    # Adv            
    g_loss_adv = local_adv_loss(D_ref_imgs, ones)
            
    # Similarity
    g_loss_reg = torch.mean(torch.abs(ref_imgs - syn_imgs))
                
    g_loss = g_loss_adv + g_loss_reg#_scale
            
    g_loss.backward(retain_graph=True)
    optimizer_G.step()
    
    if (j+1) % 10 == 0:
        print("Refiner pre-trained: %d times| g_loss: %f" % ((j+1), g_loss.item()))
print("=" * 120)
for j in range(20):
    syn_imgs = Variable(synthetic.__iter__().next()).float().cuda()
    real_imgs = Variable(real.__iter__().next()).float().cuda()
    ref_imgs = generator(syn_imgs)
    
    # --------- Discriminator --------- #        
    optimizer_D.zero_grad()
        
    D_ref_imgs2 = discriminator(ref_imgs).view(-1, 2)
    D_real_imgs = discriminator(real_imgs).view(-1, 2)
        
    d_loss = local_adv_loss(D_ref_imgs2, zeros) + local_adv_loss(D_real_imgs, ones)
        
    d_loss.backward()
    optimizer_D.step()
    
    if (j+1) % 10 == 0:
        print("Discriminator pre-trained: %d times| d_loss: %f" % ((j+1), d_loss.item()))
pre_stop = timeit.default_timer()

pre = [g_loss.item(), d_loss.item(), (pre_stop-pre_start)]
f = open(os.path.join(loss_path, "pretrain_loss.txt"), 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
wr.writerow("G Loss, D Loss, Time")
wr.writerow(pre)
f.close()
print("="*120)
# train ###########################################################################################################
start = timeit.default_timer()
for epoch in range(opt.n_epochs):
    for i in range(len(synthetic)):

        syn_imgs = Variable(synthetic.__iter__().next()).float().cuda()
        real_imgs = Variable(real.__iter__().next()).float().cuda()
        
        for j in range(50):

            # --------- Refiner --------- #            
            optimizer_G.zero_grad()
            
            ref_imgs = generator(syn_imgs)
                
            D_ref_imgs = discriminator(ref_imgs).view(-1, 2)
                    
            # Adv            
            g_loss_adv = local_adv_loss(D_ref_imgs, ones)
            
            # Similarity
            g_loss_reg = torch.mean(torch.abs(ref_imgs - syn_imgs))
            #g_loss_reg = self_regularization_loss(ref_imgs, syn_imgs)
            #g_loss_reg_scale = torch.mul(g_loss_reg, opt.delta)
                
            g_loss = g_loss_adv + g_loss_reg#_scale
            
            g_loss.backward(retain_graph=True)
            optimizer_G.step()

        # --------- Discriminator --------- #        
        optimizer_D.zero_grad()
        
        # buffer        
        half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
        image_history_buffer.add_to_image_history_buffer(ref_imgs.cpu().detach().numpy())

        if len(half_batch_from_image_history):
            torch_type = torch.from_numpy(half_batch_from_image_history)
            ref_imgs[:opt.batch_size // 2] = torch_type
            ref_imgs = Variable(ref_imgs).float().cuda()
        
        D_ref_imgs2 = discriminator(ref_imgs).view(-1, 2)
        D_real_imgs = discriminator(real_imgs).view(-1, 2)
        
        d_loss = local_adv_loss(D_ref_imgs2, zeros) + local_adv_loss(D_real_imgs, ones)
        
        d_loss.backward()
        optimizer_D.step()

        # --------------
        # Log Progress
        # --------------
        print("-" * 120)
        print(
            "[Epoch %d/%d] [Step %d/%d] [D loss : %f] [G loss: %f]"
            % (epoch, opt.n_epochs, (i+1), len(synthetic), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(synthetic) + i + 1
        
        tot_g_loss[(batches_done-1) % opt.sample_interval] = g_loss.item()
        tot_d_loss[(batches_done-1) % opt.sample_interval] = d_loss.item()
        
        if batches_done % opt.sample_interval == 0:
            stop = timeit.default_timer()
            time = [stop-start]
            
            save_image(syn_imgs.data[:12], os.path.join(image_path, "%d_0sample.png" % batches_done), nrow=3, normalize=True)
            
            save_image(ref_imgs.data[:12], os.path.join(image_path, "%d_1refined.png" % batches_done), nrow=3, normalize=True)
            #save_image(D_ref_imgs.data[:12], os.path.join(image_path, "%d_2autoencoder.png" % batches_done), nrow=3, normalize=True)
            
            save_image(real_imgs.data[:12], os.path.join(image_path, "%d_3real.png" % batches_done), nrow=3, normalize=True)            
            #save_image(D_real_imgs.data[:12], os.path.join(image_path, "%d_4autoencoder.png" % batches_done), nrow=3, normalize=True)
            
            if batches_done > 0:
                f = open(os.path.join(loss_path, "%d_loss.txt" % batches_done), 'w', encoding='utf-8', newline='')
                wr = csv.writer(f, delimiter='\t')
                wr.writerow("G Loss")
                wr.writerow(tot_g_loss.tolist())
                wr.writerow("D Loss")
                wr.writerow(tot_d_loss.tolist())
                wr.writerow("Time")
                wr.writerow(time)
                f.close()
            
            start = timeit.default_timer()
            
        if batches_done % 100 == 0:
            torch.save({'epoch': batches_done,
                        'model_state_dict': generator.state_dict(),
                        'optimizer_state_dict': optimizer_G.state_dict(),
                        'loss': g_loss}, os.path.join(models_path, "%d_generator" % batches_done))
            torch.save({'epoch': batches_done,
                        'model_state_dict': discriminator.state_dict(),
                        'optimizer_state_dict': optimizer_D.state_dict(),
                        'loss': d_loss}, os.path.join(models_path, "%d_discriminator" % batches_done))
            
            







