import argparse
import os
import csv
import torchvision.transforms as transforms
from torchvision.utils import save_image

#import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.autograd import Variable
import torch
import timeit

######################################################################################################################
import sys
sys.path.append("C:/유민형/개인 연구/BESimGAN/scripts")
from SimGAN_arch5 import Generator, Discriminator
import functions as fn
#from image_history_buffer import ImageHistoryBuffer

######################################################################################################################
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=63, help="number of experiment")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample_interval", type=int, default=10, help="number of sampling images")
parser.add_argument("--delta", type=float, default=0.01, help="Scale factor of refine loss")

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
#self_regularization_loss = nn.L1Loss(size_average=False)

# BEGAN hyper parameters ###########################################################################################
gamma = 0.9
lambda_k = 0.001
k = 0.0

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

#image_history_buffer = ImageHistoryBuffer((0, 1, 35, 55), 120000, opt.batch_size)
print("="*120)
start = timeit.default_timer()
for epoch in range(opt.n_epochs):
    for i in range(len(synthetic)):

        syn_imgs = Variable(synthetic.__iter__().next()).float().cuda()
        real_imgs = Variable(real.__iter__().next()).float().cuda()
        
        # --------- Refiner --------- #
        
        optimizer_G.zero_grad()
         
        ref_imgs = generator(syn_imgs)
            
        D_ref_imgs = discriminator(ref_imgs)
        #D_syn_imgs = discriminator(syn_imgs)    
        
        #ref_imgs = generator(D_syn_imgs)
        #D_ref_imgs = discriminator(ref_imgs)
        
        # BEGAN
        #g_loss_began = self_regularization_loss(D_ref_imgs, ref_imgs)
        g_loss_began = torch.mean(torch.abs(D_ref_imgs - ref_imgs))
        
        # Similarity
        #g_loss_reg = self_regularization_loss(ref_imgs, syn_imgs)
        g_loss_reg = torch.mean(torch.abs(ref_imgs - syn_imgs))
        g_loss_reg_scale = torch.mul(g_loss_reg, opt.delta)
            
        g_loss = g_loss_began + g_loss_reg_scale
        
        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        
        # --------- Discriminator --------- #
        
        optimizer_D.zero_grad()
        
        # BEGAN
        D_real_imgs = discriminator(real_imgs)
        D_ref_imgs2 = discriminator(ref_imgs.detach())
        D_syn_imgs2 = discriminator(syn_imgs)

        #d_loss_real = self_regularization_loss(D_real_imgs, real_imgs)
        #d_loss_fake = self_regularization_loss(D_ref_imgs2, ref_imgs.detach())
        d_loss_real = torch.mean(torch.abs(D_real_imgs - real_imgs))
        d_loss_fake = torch.mean(torch.abs(D_ref_imgs2 - ref_imgs.detach()))
        d_loss_reg = torch.mean(torch.abs(D_syn_imgs2 - syn_imgs))
        
        d_loss = d_loss_real - k * (d_loss_fake + d_loss_reg)

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # ----------- Parameter ----------- #
        diff = torch.sum(gamma * d_loss_real - (d_loss_fake + d_loss_reg))

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).item()

        # --------------
        # Log Progress
        # --------------
        print("-" * 120)
        print(
            "[Epoch %d/%d] [Step %d/%d] [D loss : %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, opt.n_epochs, (i+1), len(synthetic), d_loss.item(), g_loss.item(), M, k)
        )

        batches_done = epoch * len(synthetic) + i + 1
        
        tot_g_loss[(batches_done-1) % opt.sample_interval] = g_loss.item()
        tot_d_loss[(batches_done-1) % opt.sample_interval] = d_loss.item()
        
        if batches_done % opt.sample_interval == 0:
            stop = timeit.default_timer()
            time = [stop-start]
            
            save_image(syn_imgs.data[:12], os.path.join(image_path, "%d_0sample.png" % batches_done), nrow=3, normalize=True)
            #save_image(D_syn_imgs.data[:12], os.path.join(image_path, "%d_1D_syn.png" % batches_done), nrow=3, normalize=True)
            
            save_image(ref_imgs.data[:12], os.path.join(image_path, "%d_2refined.png" % batches_done), nrow=3, normalize=True)
            save_image(D_ref_imgs.data[:12], os.path.join(image_path, "%d_3D_ref.png" % batches_done), nrow=3, normalize=True)
            
            save_image(real_imgs.data[:12], os.path.join(image_path, "%d_4real.png" % batches_done), nrow=3, normalize=True)            
            save_image(D_real_imgs.data[:12], os.path.join(image_path, "%d_5D_real.png" % batches_done), nrow=3, normalize=True)
            
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
                        'loss': g_loss}, os.path.join(models_path, "%d_generator.pkl" % batches_done))
            torch.save({'epoch': batches_done,
                        'model_state_dict': discriminator.state_dict(),
                        'optimizer_state_dict': optimizer_D.state_dict(),
                        'loss': d_loss}, os.path.join(models_path, "%d_discriminator.pkl" % batches_done))
    
            







