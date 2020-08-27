import torch
import torch.nn as nn
import copy

class Object(object):
    pass

opt = Object()
opt.n_epochs = 50000 # number of epochs of training
opt.batch_size = 32 # size of the batches
opt.lr = 0.001 # adam: learning rate
opt.b1 = 0.6 # adam: decay of first order momentum of gradient
opt.b2 = 0.999 # adam: decay of first order momentum of gradient
opt.n_cpu = 8 # number of cpu threads to use during batch generation
opt.latent_dim = 100 # dimensionality of the latent space
opt.img_size = 64 # size of each image dimension
opt.channels = 3 # number of image channels
opt.sample_interval = 500 # interval between image sampling
img_shape = (opt.channels, opt.img_size, opt.img_size)
opt.embedding_dim = 768 # embedding's dimension, BERT:768,  google-word2vec:300
opt.reduced_embedding = 100
opt_g = copy.copy(opt)
opt_d = copy.copy(opt)
opt_d.lr = 0.0001
opt_d.b1 = 0.4
opt_d.b2 = 0.99

class CGenerator(nn.Module):
    def __init__(self):
        super(CGenerator, self).__init__()
        self.dim = opt.latent_dim

        self.linear_block = nn.Sequential(nn.Linear(opt.embedding_dim, opt.reduced_embedding), nn.ReLU())
        
        self.l1, self.conv_blocks = self.conv_blocks2()
          
    def forward(self, image, embeddings):
        embeddings = self.linear_block(embeddings)
        cat_input = torch.cat([image, embeddings], 1)
        #print("G - cat input size: ", cat_input.size())
        out = self.l1(cat_input)
        #print("G - 1st fc size: ", cat_input.size())
        out = out.view(out.shape[0], self.layer1_depth, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        #print("Generated Image Size: ", img.size())
        #img = img.view(img.size(0), 4*64*64)
        return img
   
    def conv_blocks2(self):
        # ----------------------
        # adjustable parameters
        self.scale = 0.5
        self.init_size = opt.img_size // 4
        self.layer1_depth = int(1024 * self.scale)
        self.layer2_depth = int(512 * self.scale)
        self.layer3_depth = int(256 * self.scale)
        self.layer4_depth = int(128 * self.scale)
        self.layer5_depth = int(64 * self.scale)
        # ----------------------

        l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.reduced_embedding, self.layer1_depth * self.init_size ** 2), 
                           nn.BatchNorm1d(self.layer1_depth * self.init_size ** 2,0.9), nn.ReLU())
        # output dim: (self.layer1_depth, self.init_size, self.init_size)
        
        conv_block = nn.Sequential(
            # Layer 1
            #nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(self.layer1_depth, self.layer2_depth, 3, stride=2, padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(self.layer2_depth, 0.9),
            nn.ReLU(),
            # Layer 2        
            nn.ConvTranspose2d(self.layer2_depth, self.layer3_depth, 3, stride=2, padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(self.layer3_depth, 0.9),
            nn.ReLU(),
            # output dim: (self.layer3_depth, hh, ww)
            # where hh=ww=(h-1)*stride-2*padding+(kernel_size-1)+1

            # Layer 3        
            nn.Conv2d(self.layer3_depth, self.layer4_depth, 3, stride= 2, padding=1,padding_mode="zeros"),
            nn.BatchNorm2d(self.layer4_depth, 0.9),
            nn.ReLU(),
            # Layer 4            
            nn.ConvTranspose2d(self.layer4_depth, opt.channels, 3, stride= 2, padding= 1, padding_mode="zeros"),

            # Output Layer
            nn.Sigmoid()
        )
        
        return l1, conv_block