import sys
import os
sys.path.append(os.getcwd())
from Dataset.tools.data_proc_utils import *
import pandas as pd
from Bio import SeqIO

def Antibody_SecondaryStructure(seg_by_no_data_path,seg_by_secondstructure_data_path,name1):

    models_folder = "/public_data/zhaowb/AbGAN_Web/src/statics/code/AbGAN-LMG/Dataset/tools/second_structure_prediction/tools/proteinUnet/data/models"
    seg_by_no_to_seg_by_secondstructure(seg_by_no_data_path=seg_by_no_data_path,
                                        seg_by_secondstructure_data_path=seg_by_secondstructure_data_path,
                                        models_folder=models_folder,
                                        chunksize = 5000,
                                        GPU='gpu:2',
                                        test = False,
                                        is_multipleproccess = False,
                                        name=name1)

def Antibody_amino_acids(sequence,seg_by_secondstructure_data_path):
    with open(seg_by_secondstructure_data_path,"w") as w:    
        sequence = sequence.replace('\n','')
        for amino in sequence:
                w.write(amino+" ")

def Antibody_amino_acids_fasta(fasta_path,seg_by_secondstructure_data_path):
    with open(fasta_path,"r") as r:
        sequences = r.readlines()    
    with open(seg_by_secondstructure_data_path,"w") as w:
        for sequence in sequences:    
            sequence = sequence.replace('\n','')
            for amino in sequence:
                    w.write(amino+" ")
            w.write("\n")
           


def select_model(model):
    if model=="ESM2-150M":
        dim=640
    elif model=="ProtBERT":
        dim=1024
    elif model=="AntiBERTy":
        dim=512
    elif model=="BERTAb2D" or "AbLang":
        dim=768    
    model_param=f"Model_Save/Model/AbGAN_{model}.pkl"
    return model_param,dim