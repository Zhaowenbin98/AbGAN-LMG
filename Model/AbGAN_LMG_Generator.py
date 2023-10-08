import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channels):
        super(ResidualBlock,self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=1,padding=1,bias=True)
        self.deconv = nn.ConvTranspose1d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=2,output_padding=1,dilation=2,bias=True)
        self.deconv_1 = nn.ConvTranspose1d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=2,output_padding=1,dilation=2,bias=True)


    def forward(self,protein_ft):

        identity = self.deconv(protein_ft)
        protein_ft = self.activation(protein_ft)
        conv_ft_2 = self.deconv_1(protein_ft)
        conv_ft_2 = self.activation(conv_ft_2)
        conv_ft_2 = self.conv(conv_ft_2)
        conv_ft_2 += identity
        
        return conv_ft_2

class SelfAttention(nn.Module):
    def __init__(self,in_channel,out_channels):
        super(SelfAttention,self).__init__()
        self.ch_sqrt = int(math.sqrt(out_channels))
        self.conv_k = nn.Conv1d(in_channels=in_channel, out_channels=self.ch_sqrt, kernel_size=1, stride=1)
        self.conv_q = nn.Conv1d(in_channels=in_channel, out_channels=self.ch_sqrt, kernel_size=1, stride=1)
        self.conv_v = nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,protein_ft):

        x=protein_ft
        conv_k=self.conv_k(protein_ft)
        conv_q=self.conv_q(protein_ft)
        conv_v=self.conv_v(protein_ft)
        conv_q=torch.transpose(conv_q,1,2)
        conv_qk=torch.matmul(conv_q,conv_k)
        conv_qk=self.softmax(conv_qk)
        conv=torch.matmul(conv_v,conv_qk)
        conv=torch.add(conv,x)

        return conv




class AbGANGenerator(nn.Module):
    def __init__(self, **param_dict):
        super(AbGANGenerator, self).__init__()
        self.param_dict = param_dict
        self.batch =param_dict['batch_size']
        self.max_length = param_dict['max_length']
        self.label_dim = param_dict['label_dim']
        self.amino_num = param_dict['amino_num']
        self.kernel_size = param_dict['kernel_size']        
        self.stride = param_dict['stride']
        self.temperature = param_dict['temperature']
        self.noise_dim = param_dict['noise_dim']
        self.reshape_size_1 = self.max_length/self.stride**5
        self.reshape_size_2 = self.max_length*self.amino_num/self.reshape_size_1
        self.filter_conv = self.stride**4 * self.amino_num
        self.dense = nn.Linear(self.label_dim+self.noise_dim,self.max_length*self.amino_num)
        self.activation = nn.LeakyReLU(0.2)
        self.batchnom_0 =nn.BatchNorm1d(num_features=int(self.max_length*self.amino_num))
        self.batchnom_1 =nn.BatchNorm1d(num_features=int(self.filter_conv))
        self.batchnom_2 =nn.BatchNorm1d(num_features=int(self.filter_conv/2))
        self.batchnom_3 =nn.BatchNorm1d(num_features=int(self.filter_conv/4))
        self.batchnom_4 =nn.BatchNorm1d(num_features=int(self.filter_conv/8))
        self.batchnom_5 =nn.BatchNorm1d(num_features=int(self.amino_num))           
        self.conv_0 = ResidualBlock(in_channel=int(self.reshape_size_2), out_channels=int(self.filter_conv))
        self.conv_1 = ResidualBlock(in_channel=int(self.filter_conv), out_channels=int(self.filter_conv/2))
        self.conv_2 = ResidualBlock(in_channel=int(self.filter_conv/2), out_channels=int(self.filter_conv/4))
        self.conv_3 = ResidualBlock(in_channel=int(self.filter_conv/4), out_channels=int(self.filter_conv/8))
        self.conv_4 = ResidualBlock(in_channel=int(self.filter_conv/8), out_channels=int(self.filter_conv/16))
        self.selfatt = SelfAttention(in_channel=int(self.amino_num), out_channels=int(self.amino_num))
        self.softmax = nn.Softmax()

    def forward(self,label,noise):
        batch_size,input_dim = label.size()
        input = torch.cat((noise,label),1)
        input = self.dense(input)
        input = self.activation(input)
        input = self.batchnom_0(input)
        input = torch.reshape(input,[batch_size,int(self.reshape_size_1),int(self.reshape_size_2)])
        input = input.transpose(1,2)
        input = self.conv_0(input)
        input = self.activation(input)
        input = self.batchnom_1(input)
        input = self.conv_1(input)
        input = self.activation(input)
        input = self.batchnom_2(input)
        input = self.conv_2(input)
        input = self.activation(input)
        input = self.batchnom_3(input)
        input = self.conv_3(input)
        input = self.activation(input)
        input = self.batchnom_4(input)
        input = self.conv_4(input)
        input = self.selfatt(input) 
        input = self.activation(input)
        input = self.batchnom_5(input)
        input = input.transpose(1,2)
        output = F.gumbel_softmax(input,hard=False,tau=self.temperature,dim=-1)
        
        return output