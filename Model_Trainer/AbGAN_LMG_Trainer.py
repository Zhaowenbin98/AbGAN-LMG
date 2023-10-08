#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import random
import torch
import os.path as osp
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from Dataset.AbGAN_LMG_Dataset_fasta import AbGAN_Dataset
from Model.AbGAN_LMG_Discriminator import AbGANDiscriminator
from Model.AbGAN_LMG_Generator import AbGANGenerator
from Dataset.tools.data_proc import Antibody_SecondaryStructure,Antibody_amino_acids_fasta

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--seed',type=int ,default= 10)
parser.add_argument('--batch_size',type=int ,default= 64)
parser.add_argument('--max_length',type=int ,default= 128)
parser.add_argument('--label_dim',type=int ,default= 768)
parser.add_argument('--noise_dim',type=int ,default= 768)
parser.add_argument('--amino_num',type=int ,default= 21)
parser.add_argument('--kernel_size',type=int ,default= 3)
parser.add_argument('--stride',type=int ,default= 2)
parser.add_argument('--temperature',type=float ,default= 1.0)
parser.add_argument('--lr_g',type=float ,default= 1e-4)
parser.add_argument('--lr_d',type=float ,default= 1e-4)
parser.add_argument('--d_train_step',type=int ,default= 1)
parser.add_argument('--g_train_step',type=int ,default= 1)
parser.add_argument('--epochs',type=int ,default= 1000)
parser.add_argument('--epoch_to_save',type=int ,default= 1)
parser.add_argument('--model_select',default= "AbLang")
parser.add_argument('--train_dataset_save_path',default= "Dataset/train_data/Dataset_Train.txt")
parser.add_argument('--train_dataset_labels_save_path',default= "Dataset/train_data/Dataset_Train_Tokenizer.txt")
parser.add_argument('--val_dataset_save_path',default= "Dataset/train_data/Dataset_Val.txt")
parser.add_argument('--val_dataset_labels_save_path',default= "Dataset/train_data/Dataset_Val_Tokenizer.txt")
args = parser.parse_args()

#Data Preparation
sequences_path_train = args.train_dataset_save_path
if args.model_select =="BERT2DAb":
    Antibody_SecondaryStructure(args.train_dataset_save_path,args.train_dataset_labels_save_path,args.model_select)
else:
    Antibody_amino_acids_fasta(args.train_dataset_save_path,args.train_dataset_labels_save_path)
label_path_train = args.train_dataset_labels_save_path

sequences_path_val = args.val_dataset_save_path
if args.model_select =="BERT2DAb":
    Antibody_SecondaryStructure(args.val_dataset_save_path,args.val_dataset_labels_save_path,args.model_select)
else:
    Antibody_amino_acids_fasta(args.val_dataset_save_path,args.val_dataset_labels_save_path)
label_path_val = args.val_dataset_labels_save_path

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_train = AbGAN_Dataset(sequence = self.param_dict['sequence_train'],
                                    label_path = self.param_dict['sequence_train_label'],
                                    device = self.device,
                                    max_len= self.param_dict['max_length'],
                                    model_select = self.param_dict['model_select'])
        self.dataset_val = AbGAN_Dataset(sequence = self.param_dict['sequence_val'],
                                    label_path = self.param_dict['sequence_val_label'],
                                    device = self.device,
                                    max_len= self.param_dict['max_length'],
                                    model_select = self.param_dict['model_select'])
        self.dataloader_train = DataLoader(self.dataset_train,batch_size = self.param_dict['batch_size'],shuffle=True,drop_last=True)
        self.dataloader_val = DataLoader(self.dataset_val,batch_size = self.param_dict['batch_size'],shuffle=True,drop_last=True)
        self.file_name = __file__.split('\\')[-1].replace('.py', '')
        self.writer=SummaryWriter(f"Model_Save/Logs/Logs_{args.model_select}")
        self.trainer_info = 'Model_Save/Seed={}_Batch={}_{}'.format(self.param_dict['seed'], self.param_dict['batch_size'],args.model_select)
        self.build_model()

    def build_model(self):
        #Model Building
        self.gen_model = AbGANGenerator(**self.param_dict).to(self.device)
        self.dis_model = AbGANDiscriminator(**self.param_dict).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=self.param_dict['lr_g'],betas=(0, 0.9))
        self.dis_optimizer = torch.optim.Adam(self.dis_model.parameters(), lr=self.param_dict['lr_d'],betas=(0, 0.9))
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, step_size=1000, gamma=0.98)
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, step_size=1000, gamma=0.98)
        #Statistics Training Parameters
        total_num_gen = sum(p.numel() for p in self.gen_model.parameters())
        trainable_num_gen = sum(p.numel() for p in self.gen_model.parameters() if p.requires_grad)
        total_num_dis = sum(p.numel() for p in self.dis_model.parameters())
        trainable_num_dis = sum(p.numel() for p in self.dis_model.parameters() if p.requires_grad)
        print(f'generater model parameters: total: {total_num_gen}, trainable: {trainable_num_gen}')
        print(f'discriminator model parameters: total: {total_num_dis}, trainable: {trainable_num_dis}')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
    
    def iteration(self, epoch, data_loader, is_training=True):
        Loss_G=0
        Loss_D=0
        #Training Model  
        if is_training:
            self.gen_model.train()
            self.dis_model.train()
            for sequence,label in data_loader:
                noise = torch.randn((self.param_dict['batch_size'],self.param_dict['noise_dim'])).to(self.device) 
                sequence = sequence.type(torch.float32)
                sequence = sequence.to(self.device)
                label = np.squeeze(label).to(self.device)
                #Training Discriminator Module
                for i in range(self.param_dict['d_train_step']):
                    self.dis_optimizer.zero_grad()
                    generate_sequences = self.gen_model(label,noise).detach()
                    pred_true  = self.dis_model(label,sequence)
                    pred_false   = self.dis_model(label,generate_sequences)
                    #Training Discriminator Loss and Backward Propagation
                    r = torch.rand(size=(self.param_dict['batch_size'],1,1)).to(self.device)
                    x = (r * sequence + (1 - r) * generate_sequences).requires_grad_(True)
                    pred_dis_avg  = self.dis_model(label,x)
                    fake = torch.ones_like(pred_dis_avg)
                    g =torch.autograd.grad(outputs=pred_dis_avg,inputs=x,grad_outputs=fake,create_graph=True,retain_graph=True)[0]
                    gp =((g.norm(2,dim=1)-1)** 2).mean()
                    d_loss = (-torch.mean(pred_true) + torch.mean(pred_false)).to(self.device) + 10*gp
                    d_loss.backward()
                    self.dis_optimizer.step()
                    Loss_D += d_loss

                #Training Generator Module
                for i in range(self.param_dict['g_train_step']):
                    noise = torch.randn((self.param_dict['batch_size'],self.param_dict['noise_dim'])).to(self.device)          
                    generate_sequences_G = self.gen_model(label,noise)
                    pred_gen  = self.dis_model(label,generate_sequences_G)
                    #Training Generator Loss and Backward Propagation
                    g_loss = (-torch.mean(pred_gen)).to(self.device)
                    self.gen_optimizer.zero_grad()
                    g_loss.backward()
                    self.gen_optimizer.step()
                    Loss_G += g_loss
            #Record Loss Changes
            Loss_D = Loss_D*self.param_dict['batch_size']/len(self.dataset_train)
            Loss_G = Loss_G*self.param_dict['batch_size']/len(self.dataset_train)
            self.writer.add_scalar("train_d_loss",Loss_D.item(),global_step=epoch)
            self.writer.add_scalar("train_g_loss",Loss_G.item(),global_step=epoch)

            self.gen_scheduler.step()
            self.dis_scheduler.step()

        #Evaluating Model 
        else:
            self.gen_model.eval()
            self.dis_model.eval()

            for sequence,label in data_loader:
                noise = torch.randn((self.param_dict['batch_size'],self.param_dict['noise_dim'])).to(self.device)                
                sequence = sequence.type(torch.float32)
                sequence = sequence.to(self.device)
                label = np.squeeze(label).to(self.device)
                #Evaluating Discriminator Module   
                generate_sequences = self.gen_model(label,noise).detach()
                pred_true = self.dis_model(label,sequence)
                pred_false = self.dis_model(label,generate_sequences)
                r = torch.rand(size=(self.param_dict['batch_size'],1,1)).to(self.device)
                x = (r * sequence + (1 - r) * generate_sequences).requires_grad_(True)
                pred_dis_avg  = self.dis_model(label,x)
                fake = torch.ones_like(pred_dis_avg)
                g =torch.autograd.grad(outputs=pred_dis_avg,inputs=x,grad_outputs=fake,create_graph=True,retain_graph=True)[0]
                gp =((g.norm(2,dim=1)-1)** 2).mean()
                d_loss = (-torch.mean(pred_true) + torch.mean(pred_false)).to(self.device) + 10*gp
                Loss_D += d_loss
                #Evaluating Generator Module 
                noise = torch.randn((self.param_dict['batch_size'],self.param_dict['noise_dim'])).to(self.device)            
                generate_sequences_G = self.gen_model(label,noise)
                pred_gen = self.dis_model(label,generate_sequences_G)
                g_loss = (-torch.mean(pred_gen)).to(self.device)
                Loss_G += g_loss
                
            Loss_D = Loss_D*self.param_dict['batch_size']/len(self.dataset_val)
            Loss_G = Loss_G*self.param_dict['batch_size']/len(self.dataset_val)
            self.writer.add_scalar("val_d_loss",Loss_D.item(),global_step=epoch)
            self.writer.add_scalar("val_g_loss",Loss_G.item(),global_step=epoch)

        return Loss_G , Loss_D 

    def train(self,epochs,epoch_to_save):
        int_ld =100
        it = tqdm(range(0,epochs+1), total=epochs, initial=0)
        for epoch in it:
            LG_train , LD_train = self.iteration(epoch, self.dataloader_train,
                           is_training=True)
            LG_val , LD_val  = self.iteration(epoch, self.dataloader_val,
                           is_training=False)
            print(f'Epoch {epoch} Complete.')
            print(f'Train: Loss_G:{LG_train}, Loss_D:{LD_train}')
            print(f'Val: Loss_G:{LG_val}, Loss_D:{LD_val}')
            LD_train_abs = abs(LD_train)

            if epoch  > epoch_to_save and LD_train_abs < int_ld:
                save_complete_model_path = osp.join(self.trainer_info + '_Generator_complete.pkl')
                torch.save(self.gen_model,save_complete_model_path)
                same_model_param_path = osp.join(self.trainer_info + '_Generator_param.pkl')
                torch.save(self.gen_model.state_dict(), same_model_param_path)
                save_complete_model_path = osp.join(self.trainer_info + '_Discriminator_complete.pkl')
                torch.save(self.dis_model,save_complete_model_path)
                same_model_param_path = osp.join(self.trainer_info + '_Discriminator_param.pkl')
                torch.save(self.dis_model.state_dict(), same_model_param_path)
                int_ld = LD_train_abs


if __name__ == '__main__':

    param_dict = {
                    'seed': args.seed,
                    'batch_size': args.batch_size,
                    'max_length': args.max_length,
                    'label_dim': args.label_dim,
                    'amino_num': args.amino_num,
                    'noise_dim':args.noise_dim,
                    'kernel_size': args.kernel_size,
                    'stride': args.stride,
                    'temperature': args.temperature,
                    'lr_g': args.lr_g,
                    'lr_d': args.lr_d,
                    'd_train_step':args.d_train_step,
                    'g_train_step':args.g_train_step,
                    "sequence_train":args.train_dataset_save_path,
                    "sequence_train_label": args.train_dataset_labels_save_path,
                    "sequence_val":args.val_dataset_save_path,
                    "sequence_val_label": args.val_dataset_labels_save_path,
                    "model_select":args.model_select,
                }
            
    trainer = Trainer(**param_dict)
    trainer.train(epochs=args.epochs,epoch_to_save=args.epoch_to_save)