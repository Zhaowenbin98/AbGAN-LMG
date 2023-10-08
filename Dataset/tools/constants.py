import torch 
import math
import numpy as np

amino_map_idx = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    '-': 20,
}

idx_map_amino = {
    0:"A",
    1:"R",
    2:"N",
    3:"D",
    4:"C",
    5:"Q",
    6:"E",
    7:"G",
    8:"H",
    9:"I",
    10:"L",
    11:"K",
    12:"M",
    13:"F",
    14:"P",
    15:"S",
    16:"T",
    17:"W",
    18:"Y",
    19:"V",
    20:'-',
}

amino_acid_alphabet = 'ARNDCEQGHILKMFPSTWYV'

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=int(0), min_tokens_to_keep=1):

        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits


def cacShannonEnt(sequence):
    shannonEnt_result=[]
    for i  in range(0,64):
        shannonEnt_list_all = []
        shannonEnt_list = []
        for j in range(0,128):
            sequence_prob = sequence[i,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()
            shannonEnt = 0
            
            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt -= num * math.log(num, 2)
            shannonEnt_list.append(shannonEnt)

        shannonEnt_ =  np.array(shannonEnt_list)
        shannonEnt_result.append(shannonEnt_)


    return shannonEnt_result

def cacShannonEnt_CDR(sequence,data,k):
    shannonEnt_result_h1=[]
    shannonEnt_result_h2=[]
    shannonEnt_result_h3=[]
    shannonEnt_result_h4=[]
    for m  in range(0,64):
 
        i = data[m+k*64]   
        if i =="no antibody":
            continue
        i1 = int(i.split(',')[0])
        i2 = int(i.split(',')[1])
        i3 = int(i.split(',')[2])
        i4 = int(i.split(',')[3])
        i5 = int(i.split(',')[4])
        i6 = int(i.split(',')[5])

        shannonEnt_result_h1_1=[]
        shannonEnt_result_h2_2=[]
        shannonEnt_result_h3_3=[]
        shannonEnt_result_h4_4=[]

        for j in range(int(i1),int(i2)):
            sequence_prob = sequence[m,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()
            shannonEnt_h1 = 0
            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt_h1 -= num * math.log(num, 2)
            shannonEnt_result_h1_1.append(shannonEnt_h1)
        shannonEnt_h1_ =  np.mean(np.array(shannonEnt_result_h1_1),axis=0)

        for j in range(int(i3),int(i4)):

            sequence_prob = sequence[m,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()
            shannonEnt_h2 = 0
            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt_h2 -= num * math.log(num, 2)
            shannonEnt_result_h2_2.append(shannonEnt_h2)
        shannonEnt_h2_ =  np.mean(np.array(shannonEnt_result_h2_2),axis=0)

        for j in range(int(i5),int(i6)):
            sequence_prob = sequence[m,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()

            shannonEnt_h3 = 0

            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt_h3 -= num * math.log(num, 2)

            shannonEnt_result_h3_3.append(shannonEnt_h3)
        shannonEnt_h3_ =  np.mean(np.array(shannonEnt_result_h3_3),axis=0)

        shannonEnt_result_h1.append(shannonEnt_h1_)
        shannonEnt_result_h2.append(shannonEnt_h2_)
        shannonEnt_result_h3.append(shannonEnt_h3_)

        for j in range(int(i2),int(i3)):
            sequence_prob = sequence[m,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()

            shannonEnt_h4 = 0

            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt_h4 -= num * math.log(num, 2)

            shannonEnt_result_h4_4.append(shannonEnt_h4)

        for j in range(int(i4),int(i5)):
            sequence_prob = sequence[m,j,:]
            sequence_prob = sequence_prob.clone().detach().cpu().numpy()

            shannonEnt_h4 = 0

            for num in sequence_prob:
                if num == 0.0000e+00:
                    num = 1e-50
                shannonEnt_h4 -= num * math.log(num, 2)

            shannonEnt_result_h4_4.append(shannonEnt_h4)
        shannonEnt_h4_ =  np.mean(np.array(shannonEnt_result_h4_4),axis=0)      
        shannonEnt_result_h4.append(shannonEnt_h4_)
              
    shannonEnt_result_h1 = np.array(shannonEnt_result_h1)
    shannonEnt_result_h2 = np.array(shannonEnt_result_h2)
    shannonEnt_result_h3 = np.array(shannonEnt_result_h3)
    shannonEnt_result_h4 = np.array(shannonEnt_result_h4) 
    
    return shannonEnt_result_h1,shannonEnt_result_h2,shannonEnt_result_h3,shannonEnt_result_h4
