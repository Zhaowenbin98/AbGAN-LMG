import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channels):
        super(ResidualBlock,self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, dilation=1,padding=1,bias=True)
        self.conv2 =nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=2,dilation=2,bias=True)
        self.conv2_1 =nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=2,dilation=2,bias=True)

    def forward(self,protein_ft):

        identity = self.conv2(protein_ft)
        conv_ft_2 = self.conv1(protein_ft)
        conv_ft_2 = self.activation(conv_ft_2)
        conv_ft_2 = self.conv2_1(conv_ft_2)        
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


class Project(nn.Module):
    def __init__(self,dim_0,label_dim):
        super(Project,self).__init__()
        self.dense = nn.Linear(dim_0,label_dim)
        self.dense1 =nn.Linear(label_dim,1)
    def forward(self,input,label):
        input_f = torch.flatten(input,start_dim=1,end_dim=2)
        input = self.dense(input_f)
        dot =torch.matmul(input.unsqueeze(2).transpose(1,2),label.unsqueeze(2))
        dot = dot.squeeze()
        dot = dot.unsqueeze(1)
        input = self.dense1(input)
        output = torch.add(dot,input)
        return output

class AbGANDiscriminator(nn.Module):
    def __init__(self, **param_dict):
        super(AbGANDiscriminator, self).__init__()
        self.param_dict = param_dict
        self.batch =param_dict['batch_size']
        self.max_length = param_dict['max_length']
        self.label_dim = param_dict['label_dim']
        self.amino_num = param_dict['amino_num']
        self.kernel_size = param_dict['kernel_size']        
        self.stride = param_dict['stride']
        self.project_size = self.max_length/self.stride**5
        self.filter_conv = self.stride*self.amino_num
        self.project_0 = Project(dim_0=self.max_length*self.amino_num,label_dim=self.label_dim)
        self.project_1 = Project(dim_0=int(self.amino_num*self.max_length/2),label_dim=self.label_dim)
        self.project_2 = Project(dim_0=int(self.project_size*self.max_length),label_dim=self.label_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.selfatt = SelfAttention(in_channel=int(self.amino_num), out_channels=int(self.amino_num))
        self.conv_0 = ResidualBlock(in_channel=int(self.amino_num), out_channels=int(self.filter_conv))
        self.conv_1 = ResidualBlock(in_channel=int(self.filter_conv), out_channels=int(self.filter_conv))
        self.conv_2 = ResidualBlock(in_channel=int(self.filter_conv), out_channels=int(self.filter_conv*2))
        self.conv_3 = ResidualBlock(in_channel=int(self.filter_conv*2), out_channels=int(self.filter_conv*2))
        self.conv_4 = ResidualBlock(in_channel=int(self.filter_conv*2), out_channels=int(self.max_length))

    def forward(self,label,sequence):

        sequence = sequence.transpose(1,2)
        input = self.conv_0(sequence)
        input = self.activation(input)
        output_0 = self.project_0(input,label)
        input = self.conv_1(input)
        input = self.activation(input)
        input = self.conv_2(input)      
        input = self.activation(input)
        output_1 = self.project_1(input,label)
        input = self.conv_3(input)
        input = self.activation(input)
        input = self.conv_4(input)
        input = self.activation(input)
        output_2 = self.project_2(input,label)
        output = output_0 + output_1 + output_2

        return output 