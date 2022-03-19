#PVDNet.py is our Video Deblurring Framework
import torch
import torch.nn as nn
import collections
import numpy as np

class Network(nn.Module):

    def __init__(self, PV_input_dim):
        super(Network, self).__init__()
#1..input
        #start feature transform layer
        #pixel volume
        #PV_dec_ch = 64
      #layer type  size  stride in         out regularization activation
        #input            V_t-1
        #conv      3x3    (1,1) V_t-1    64    batchnorm     relu
        #conv      3x3    (1,1) 64          64    batchnorm     relu
        #conv      3x3    (1,1) 64          64    batchnorm     relu
        #sum1             conv, Input
        #
        self.PV_dec = nn.Sequential(
            #1 input=PV_input_dim output=PV_dec_chkernel=3 stride=1 padding=1
            nn.Conv2d(PV_input_dim, 64, 3, stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #2 input=64 output=64 kernel=3 stride=1 padding=1
            nn.Conv2d(64, 64, 3, stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #added extra - check
            nn.Conv2d(64, 64, 3, stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
#2..input
        #frame size
        #image i_t-1, i_t, i_t+1  = 3 + 3 + 3 = 9
        # input = 9
        #img_dec_ch = 32
      #layer type  size  stride in         out regularization activation
        #concat    i_t-1, i_t, i_t+1  ( 3 + 3 + 3 = 9)
        #conv      3x3    (1,1) 9           32    batchnorm     relu
        #conv      3x3    (1,1) 32          32    batchnorm     relu
        #conv      3x3    (1,1) 32          32    batchnorm     relu
        #sum2             conv, concat
        #
        #concat           sum1,sum2
        #
        self.img_dec = nn.Sequential(
            #1 input=9 output=32 kernel=3 stride=1 padding=1
            nn.Conv2d(9, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
                      nn.ReLU(),
            #2 input=32 output=32
            nn.Conv2d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #added extar check
            nn.Conv2d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #end feature transform layer

        ##########################################3
#3.     #Encoder
        #############################################
        #1st intermediate after concatenating pixel vol and image = 64+32 = 96 -- input
        # downsample input=96->output=64
      #layer type  size  stride in         out regularization activation
        #conv11      5x5    (1,1) 96       64    -             -
        #conv21      5x5    (1,1) 64       64    batchnorm     relu
        #conv22      5x5    (1,1) 64       64    batchnorm     relu
        #conv23      5x5    (1,1) 64       64    -             -
        self.d0 = nn.Conv2d(96, 64, 5, stride = 1, padding = 2)
        #conv21
        #conv22
        #conv23
        self.d1 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride = 2, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
        )

        #2nd intermediate - input 64->128 output( 64*2=128)
      #layer type  size  stride in         out regularization activation
        #conv31      5x5    (2,2) 64       128   batchnorm     relu
        #conv32      5x5    (1,1) 128      128   batchnorm     relu
        #conv33      5x5    (1,1) 128      128   batchnorm     relu
        self.temp = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride = 2, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
                ################################################3


        ###################################
#4..    #   Residual Block
        ##################################33
        #ResBlocks -- 24 X 196 input -> 128 output
        #change 128 to 196
        #res channels = 196
      #layer type  size  stride in         out regularization activation
        #conv      5x5    (1,1) 128         128   batchnorm     relu
        #conv      5x5    (1,1) 128         128   batchnorm     relu
        #
        #conv      5x5    (1,1) 128         128   batchnorm     relu
        self.temp = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride = 2, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        ################################################3


        ###################################
#4..    #   Residual Block
        ##################################33
        #ResBlocks -- 24 X 196 input -> 128 output
        #change 128 to 196
        #res channels = 196
      #layer type  size  stride in         out regularization activation
        #conv      3x3    (1,1) 128         128   batchnorm     relu
        #conv      3x3    (1,1) 128         128   batchnorm     relu
        #
        #conv      3x3    (1,1) 128         128   batchnorm     relu
        #
        self.RBs = nn.ModuleList([
            nn.Sequential(
                #1
                nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                #2
                nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU()) for i in range(24)#12)
            ])
        
        #end of residual
        self.RB_end = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        #ResBlocks -- 24 end

        ################################
#5..    #decoder
        ################################
        #change 128 to 196
      #layer type  size  stride in         out regularization activation
        #conv41    3x3    (1,1) 128        64  -              -
        #add              conv41, conv33
        #
        self.dconv1 = nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1)
      #layer type  size  stride in         out regularization activation
        #dconv1    3x3    (1,1) 64         64  -              -
        #add              dconv1, conv23
        #conv51    3x3    (1,1) 64          64    batchnorm     relu
        #conv52    3x3    (1,1) 64          64    batchnorm     relu
        self.dconv1_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        #4
      #layer type  size  stride in         out regularization activation
        #dconv2    4x4    (1,1) 64         64     -              -
        #add              dconv2, conv11          -             relu
        #conv62    3x3    (1,1) 64          64    batchnorm     relu
        self.dconv2 = nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1)
        self.dconv2_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
      #layer type  size  stride in         out regularization activation
        #conv63    5x5    (1,1) 64         3   -              -
        self.end = nn.Conv2d(64, 3, 5, stride = 1, padding = 2)
        #add              conv63, input         -              -

###########################################################

    def forward(self, PV, I_prev, I_curr, I_next):
        #decode pixel volume
        n_dec = self.PV_dec(PV)
        #image decode
        n_img = self.img_dec(torch.cat((I_curr, I_prev, I_next), axis = 1))
        #concatenate the image and pixel volume - feature transform
        n = torch.cat((n_dec, n_img), axis = 1)

        #input 96->128
        d0 = self.d0(n)
        #
        d1 = self.d1(d0)
        temp = self.temp(d1)

        #
        n = temp.clone()

        #Residual Blocks - 24
        for i in range(24):#12):
            nn = self.RBs[i](n)
            n = n + nn

        #RB end
        n = self.RB_end(n)
        #RB + 2nd
        n = n + temp

        n = self.dconv1_end((self.dconv1(n) + d1))
        n = self.dconv2_end((self.dconv2(n) + d0))
        n = self.end(n)
        n = I_curr + n

        return n
