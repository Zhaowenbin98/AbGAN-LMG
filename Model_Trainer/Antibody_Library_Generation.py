#!/usr/bin/env
# coding:utf-8
import sys
import os
sys.path.append(os.getcwd())
import torch.nn.functional as F
import numpy as np
import random
import torch
import os.path as osp
import argparse
import random
from collections import defaultdict
from operator import itemgetter

from Dataset.tools.constants import idx_map_amino,top_k_top_p_filtering
from torch.utils.data.dataloader import DataLoader
from Dataset.AbGAN_LMG_Dataset import AbGAN_Dataset
from Model.AbGAN_LMG_Generator import AbGANGenerator
from Dataset.tools.data_proc import Antibody_SecondaryStructure,Antibody_amino_acids,select_model

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--generate_num',type=int ,default= 2000)
parser.add_argument('--top_num',type=int ,default= 10)
parser.add_argument('--model',default= "BERT2DAb")
parser.add_argument('--target_sequences_VH',default="QMQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVSS")
parser.add_argument('--target_sequences_VL',default="EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQHYGSSRGWTFGQGTKVEIK")
parser.add_argument('--id',default=100000)
args = parser.parse_args()

seed = random_integer = random.randint(1, 100)
model_param,dim=select_model(args.model)

sequence_count = defaultdict(int)
param_dict = {
                    'seed': seed,
                    'batch_size': 64,
                    'max_length': 128,
                    'label_dim': dim,
                    'amino_num': 21,
                    'noise_dim': dim,
                    'h_dim':36,
                    'kernel_size': 3,
                    'stride': 2,
                    'temperature': 1.0,
                }

if not os.path.exists(f"./Result/{args.id}"):
    os.makedirs(f"./Result/{args.id}")

with open(f"./Result/{args.id}/TargetAntibody.txt","w") as w:
        w.write(str(args.target_sequences_VH)+'\n')

sequences_path_train = f"./Result/{args.id}/TargetAntibody.txt"
if args.model =="BERT2DAb":
    Antibody_SecondaryStructure(f"./Result/{args.id}/TargetAntibody.txt",f"./Result/{args.id}/TargetAntibody_Tokenizer.txt",args.model)
else:
    Antibody_amino_acids(args.target_sequences_VH,f"./Result/{args.id}/TargetAntibody_Tokenizer.txt")
label_path_train = f"./Result/{args.id}/TargetAntibody_Tokenizer.txt"


torch.manual_seed(param_dict['seed'])
torch.cuda.manual_seed_all(param_dict['seed'])
np.random.seed(param_dict['seed'])
random.seed(param_dict['seed'])
torch.backends.cudnn.deterministic = False

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dataset_train = AbGAN_Dataset(sequences = args.target_sequences_VH,
                                    label_path = label_path_train,
                                    device = device,
                                    max_len= param_dict['max_length'], 
                                    model_select = args.model,
                                    num = 10000000)
dataloader_train = DataLoader(dataset_train,batch_size = param_dict['batch_size'],shuffle=False,drop_last=False)

gen_model = AbGANGenerator(**param_dict).to(device)
model_file_path = osp.join(model_param)
gen_model.load_state_dict(torch.load(model_file_path,map_location='cuda:3'))
gen_model.eval()     

fasta_num = 1
max_sequences = args.generate_num 
sequence_list = [{"sequence": str(args.target_sequences_VH), "label": str(args.target_sequences_VL)}]  

with open(f"./Result/{args.id}/TargetAntibody_Library_{args.id}.fasta", 'w') as w:
    for sequence, label in dataloader_train:
        batch_size, sequence_length, input_dim = sequence.size()
        noise = torch.randn((batch_size, param_dict['noise_dim'])).to(device)
        label = np.squeeze(label).to(device)
        generate_sequences = gen_model(label, noise)
        generate_sequences = top_k_top_p_filtering(generate_sequences, top_p=1)

        for i in range(int(batch_size)):
            generate_sequences_library = torch.multinomial(generate_sequences[i, :, :], num_samples=1)
            generate_sequences_library = generate_sequences_library.clone().detach().cpu().numpy()
            seq = [idx_map_amino[int(generate_sequences_library[j])] for j in range(128)]
            seq_1 = ''.join(seq)
            seq_1 = str(seq_1).replace('-', '')
            sequence_count[seq_1] += 1

            duplicate_index = None
            for index, existing_sequence in enumerate(sequence_list):
                if existing_sequence["sequence"] == seq_1:
                    duplicate_index = index
                    break


            if duplicate_index is not None:
                sequence_list[duplicate_index] = {"sequence": seq_1, "label": str(args.target_sequences_VL)}
                continue
            else:
                sequence_list.append({"sequence": seq_1, "label": str(args.target_sequences_VL)})
                fasta_num += 1


            if fasta_num >= max_sequences:
                break

        if fasta_num >= max_sequences:
            break

    num = 0

    for entry in sequence_list:

        w.write(f">Seq{num}_VH\n")
        w.write(f"{entry['sequence']}\n")
        w.write(f">Seq{num}_VL\n")
        w.write(f"{entry['label']}\n")
        num +=1

    sorted_sequences = sorted(sequence_count.items(), key=itemgetter(1), reverse=True)[:args.top_num]

    topk_num = 0

    with open(f"./Result/{args.id}/TargetAntibody_Library_top-k{args.id}.fasta", 'w') as fasta_file:

        fasta_file.write(f">Seq{topk_num}_VH\n{args.target_sequences_VH}\n")
        fasta_file.write(f">Seq{topk_num}_VL\n{args.target_sequences_VL}\n")

        topk_num += 1
        
        for sequence, count in sorted_sequences:
            fasta_file.write(f">Seq{topk_num}_VH\n{sequence}\n")
            fasta_file.write(f">Seq{topk_num}_VL\n{args.target_sequences_VL}\n")
            topk_num += 1

