
import torch.nn as nn
class Object(object):
    pass

opt = Object()
opt.n_epochs = 10000 # number of epochs of training
opt.batch_size = 64 # size of the batches
opt.lr = 0.0002 # adam: learning rate
opt.b1 = 0.5 # adam: decay of first order momentum of gradient
opt.b2 = 0.997 # adam: decay of first order momentum of gradient
opt.n_cpu = 8 # number of cpu threads to use during batch generation
opt.latent_dim = 5 # dimensionality of the latent space
opt.img_size = 64 # size of each image dimension
# opt.img_size = 32 # size of each image dimension
opt.channels = 3 # number of image channels
opt.sample_interval = 100 # interval between image sampling
img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.l1, self.conv_blocks = self.conv_blocks1()
        self.l1, self.conv_blocks = self.conv_blocks2()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.layer1_depth, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def conv_blocks1(self):
        # ----------------------
        # adjustable parameters
        self.init_size = opt.img_size // 4
        self.layer1_depth = 32
        self.layer2_depth = 16
        self.layer3_depth = 16
        self.layer4_depth = 16
        # ----------------------

        l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.layer1_depth * self.init_size ** 2))
        # output dim: (self.layer1_depth, self.init_size, self.init_size)
        return l1, nn.Sequential(
            # Layer 1
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.LeakyReLU(0.8, inplace=True),
            # output dim: (self.layer1_depth, scale_factor * self.init_size, scale_factor * self.init_size)
            nn.Conv2d(self.layer1_depth, self.layer2_depth, 3, stride=1, padding=1),
            # output dim: (self.layer2_depth, h, w)
            # where h=w=(scale_factor * self.init_size)

            # Layer 2
            nn.BatchNorm2d(self.layer2_depth, 0.8),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            # output dim: (self.layer2_depth, 2*h, 2*w)
            nn.Conv2d(self.layer2_depth, self.layer3_depth, 3, stride=1, padding=1),
            # output dim: (self.layer3_depth, 2*h, 2*w)

            # Layer 3
            nn.BatchNorm2d(self.layer3_depth, 0.8),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Conv2d(self.layer3_depth, self.layer4_depth, 3, stride=1, padding=1),
            # output dim: (self.layer4_depth, 2*h, 2*w)

            # Layer 4
            nn.BatchNorm2d(self.layer4_depth, 0.8),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Conv2d(self.layer4_depth, opt.channels, 3, stride=1, padding=1),
            # output dim: (opt.channels, 2*h, 2*w)

            # Output Layer
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def conv_blocks2(self):
        # ----------------------
        # adjustable parameters
        self.init_size = opt.img_size // 4
        self.layer1_depth = 1024
        self.layer2_depth = 512
        self.layer3_depth = 256
        self.layer4_depth = 16
        self.layer5_depth = 64
        # ----------------------

        l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.layer1_depth * self.init_size ** 2))
        # output dim: (self.layer1_depth, self.init_size, self.init_size)
        return l1, nn.Sequential(
            # Layer 1
            nn.Dropout2d(),
            # nn.BatchNorm2d(self.layer1_depth, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.layer1_depth, self.layer2_depth, 4, stride=2, padding=1),
            # output dim: (self.layer2_depth, h, w)
            # where h=w=(self.init_size-1)*stride-2*padding+(kernel_size-1)+1

            # Layer 2
            nn.Dropout2d(),
            nn.BatchNorm2d(self.layer2_depth, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.layer2_depth, self.layer3_depth, 4, stride=2, padding=1, bias=False),
            # nn.ConvTranspose2d(self.layer2_depth, opt.channels, 4, stride=2, padding=1),
            # output dim: (self.layer3_depth, hh, ww)
            # where hh=ww=(h-1)*stride-2*padding+(kernel_size-1)+1

            # Layer 3
            nn.Dropout2d(),
            nn.BatchNorm2d(self.layer3_depth, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.Conv2d(self.layer3_depth, self.layer4_depth, 3, stride=1, padding=1, bias=False),
            # nn.ConvTranspose2d(self.layer3_depth, opt.channels, 1, stride=1, padding=0, bias=False),
            # output dim: (self.layer4_depth, hh, ww)

            # Layer 4
            nn.Dropout2d(),
            nn.BatchNorm2d(self.layer4_depth, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.ConvTranspose2d(self.layer4_depth, opt.channels, 1, stride=1, padding=0, bias=False),
            # output dim: (opt.channels, hh, ww)


            # Output Layer
            nn.Tanh(),
            # nn.Sigmoid(),
        )