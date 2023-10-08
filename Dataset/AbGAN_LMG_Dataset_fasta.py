#!/usr/bin/env
# coding:utf-8
import torch
from Bio import SeqIO
import torch.nn.functional as F
from torch.utils.data import Dataset
from Dataset.tools.constants import amino_map_idx
from transformers import BertModel,BertTokenizer,AutoTokenizer,AutoModel
from Model.LanguageModels.AbLang.H.AbLang_roberta_model import RobertaForMaskedLMV2,RobertaModelV2

BERT2DAb_tokenizer_path = 'Model/LanguageModels/AntiBERTSS/vocab/seg_by_secondstructure_H_wordpiece/vocab.txt'
BERT2DAb_model_path = 'Model/LanguageModels/AntibodyPretrainedModel_ep9'

ProtBERT_tokenizer_path = 'Model/LanguageModels/ProtBert/vocab.txt'
ProtBERT_model_path = 'Model/LanguageModels/ProtBert'

ESM2_tokenizer_path = 'Model/LanguageModels/esm2_t30_150M_UR50D'
ESM2_model_path = 'Model/LanguageModels/esm2_t30_150M_UR50D'

AntiBERTy_tokenizer_path = 'Model/LanguageModels/AntiBERTy/vocab.txt'
AntiBERTy_model_path = 'Model/LanguageModels/AntiBERTy'

AbLang_tokenizer_path = 'Model/LanguageModels/AbLang/H/'
AbLang_model_path = 'Model/LanguageModels/AbLang/H'

class AbGAN_Dataset(Dataset):
    def __init__(self,sequence,label_path,max_len,model_select,device,num=1):
        self.sequence = self.Seq_Padding_OneHot(sequence,max_len,num)
        self.label = self.AntiBERTSS_Label(label_path,device,model_select,num)
        self._sequence = torch.tensor(self.sequence)
        self._label = torch.tensor(self.label)

    def __getitem__(self, index):
        return self._sequence[index],self._label[index]
    
    def __len__(self):
        return len(self._sequence)
    
    def Seq_Padding_OneHot(self,sequences,max_lens,num):
        data_seq_list=[]
        with open (sequences,"r") as r:
            data = r.readlines()
        for sequence in data:
            sequence = sequence.replace('\n','')
            sequence += '-'*(max_lens-len(sequence))
            sequence = sequence[:max_lens]
            seq=torch.tensor([amino_map_idx[aa] for aa in sequence])
            one_hot = F.one_hot(seq,21)
            one_hot = one_hot.clone().detach().cpu().numpy()
            data_seq_list.append(one_hot)

        data_seq_list*=num

        return data_seq_list

    def AntiBERTSS_Label(self,sequences,device,model_select,num):
        if model_select=="AbLang":
            tokenizer_H = AutoTokenizer.from_pretrained(AbLang_tokenizer_path,do_lower_case=False)
            model_H = RobertaModelV2.from_pretrained(AbLang_model_path).to(device)
        if model_select=="AntiBERTy":
            tokenizer_H = BertTokenizer.from_pretrained(AntiBERTy_tokenizer_path,do_lower_case=False)
            model_H = BertModel.from_pretrained(AntiBERTy_model_path).to(device)
        if model_select=="ProtBERT":
            tokenizer_H = BertTokenizer.from_pretrained(ProtBERT_tokenizer_path,do_lower_case=False)
            model_H = BertModel.from_pretrained(ProtBERT_model_path).to(device)
        if model_select=="BERT2DAb":
            tokenizer_H = BertTokenizer.from_pretrained(BERT2DAb_tokenizer_path,do_lower_case=False)
            model_H = BertModel.from_pretrained(BERT2DAb_model_path).to(device)            
        if model_select=="ESM2-150M":
            model_H = AutoModel.from_pretrained(ESM2_model_path,trust_remote_code=True).to(device)
            tokenizer_H = AutoTokenizer.from_pretrained(ESM2_tokenizer_path,do_lower_case=False)
        data_label_list = []
        with open (sequences,"r") as r:
            data = r.readlines()
        for i in range(0,len(data)):
            sequence = data[i].replace('\n','')

            input=tokenizer_H.encode(sequence)
            input_id = torch.tensor([input]).to(device)
            outputs = model_H(input_id)
            last = torch.mean(outputs.last_hidden_state,dim=1)
            last = last.clone().detach().cpu().numpy()
            data_label_list.append(last)

        data_label_list*=num

        del model_H
        return data_label_list