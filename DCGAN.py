from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#랜덤 시드 번호 설정
manualSeed = 999;
print("Random Seed : ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Inputs
dataroot = 'data/celeba'               #data 경로
workers = 2                            #Dataloader로 데이터를 로드하기 위해 사용되는 스레드 수
train_batch = 128                       
image_size = 64                        #Input Image Size
nc = 3                                 #Input Image Channel Size
nz = 100                               #Input Vector Size
ngf = 64                               #feature map size after Generator
ndf = 64                               #feature map size after discriminator
train_ep = 5                           
lr = 0.0002                            
beta1 = 0.5                            #Adam optimizer hyper-parameter
ngpu = 1                               #num of gpu

#set option(input resize, crop, normalize)
dataset = dset.ImageFolder(root=dataroot,
                           transform= transforms.Compose([
                              transforms.Resize(image_size),
                              transforms.CenterCrop(image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))                             
                           ]))

#data를 batch size로 분할
dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch, shuffle=True, num_workers=workers)

#gpgpu
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0)else "cpu")
real_batch = next(iter(dataloader))     #batch 이동

#이미지 보여주기 options
plt.figure(figsize=(10, 10))            
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

#Initialize Weight
def initialize_weights(m):
    classname = m.__classs__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  #평균 0, 표준편차 0.02의 정규분포를 가진 random weight
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)  #평균 1, 표준편차 0.02의 정규분포를 가진 random weight
        nn.init.constant_(m.bias.data, 0)           

#Generator
class Generator(nn.Module):
    def __init(self, ngpu): #set network
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input : uniform namdom vector Z, output : ngf*8
            nn.ConvTranspose2d(in_channels = nz, out_channels = ngf*8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            #input ngf*8, output = ngf*4
            nn.ConvTranspose2d(in_channels = ngf*8, out_channels = ngf*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.Relu(True),

            #input ngf*4, output = ngf*2 
            nn.ConvTranspose2d(in_channels = ngf*4, out_channels = ngf*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.Relu(True),

            #input ngf*2, output = ngf
            nn.ConvTranspose2d(in_channels = ngf*2, out_channels = ngf, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.Relu(True),

            #input ngf, output = nc
            nn.ConvTranspose2d(in_channels = ngf, out_channels = nc, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

#Generator Structure 확인
netG = Generator(ngpu).to(device)
if(device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(initialize_weights)
print(netG)

#Discriminator
class Discriminator(nn.Module):
    def __init(self, ngpu): #set network
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input : nc, output : ndf
            nn.ConvTranspose2d(in_channels = nc, out_channels = ndf, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.LeakyReLU(0.2,inplace=True),

            #input ngf*8, output = ngf*4
            nn.ConvTranspose2d(in_channels = ndf, out_channels = ngf*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            #input ngf*4, output = ngf*2 
            nn.ConvTranspose2d(in_channels = ngf*2, out_channels = ngf*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            #input ngf*2, output = ngf
            nn.ConvTranspose2d(in_channels = ngf*4, out_channels = ndf*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            #input ngf, output = nc
            nn.ConvTranspose2d(in_channels = ndf*8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.Sigmoid()
        )

        def forward(self, input):
            return self.main(input)

#Discriminator Structure 확인
netD = Discriminator(ngpu).to(device)
if(device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(initialize_weights)
print(netD)

#Loss Function
loss = nn.BCELoss()

#Generator와 Discriminator의 Optimizer
optG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))

#initialize option
uniform_vector = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

#Training
img_list = []
lossG = []
lossD = []
iters = 0

print("Start Training")

for epoch in range(train_ep):
    for i, data in enumerate(dataloader, 0):

        #train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ), real_label, device=device)

        output = optD(real_cpu).view(-1)
        errD_real = loss(output,label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with all-fake batch
        noise = torch.randn(b_size,nz,1,1,device=device)
        fake = netG(noise)
        label.fill_(fake_label)

        output = optD(fake.detach()).view(-1)
        errD_fake = loss(output,label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optD.step()

        # Update G
        netG.zero_grad()
        label.fill_(real_label)
    
        output = netD(fake).view(-1)
        errG = loss(output,label)
        errG.backward()
        D_G_z2 = output.mean().item()
    
        optG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, train_ep, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            lossG.append(errG.item())
            lossD.append(errD.item())
            if(iters % 500 == 0) or ((epoch == train_ep-1) and (i == len(dataloader)-1)):
                with torch.no_grad:
                    fake = netG(uniform_vector).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters +=1

#Results
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(lossG,label="G")
plt.plot(lossD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
