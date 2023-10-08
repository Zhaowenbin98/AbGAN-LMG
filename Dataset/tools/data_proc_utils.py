import os
from multiprocessing import Pool
import time
import pandas as pd
import numpy
import pandas as pd
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import h5py

config= tf.compat.v1.ConfigProto(log_device_placement=True) 
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

#1==================================================
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)
            
            
def async_kd_tokenizer(filename, worker_id, num_workers):
    with open(filename, 'r') as f:
        size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
        print(f'size {size}')
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        print(f'offset {offset}')
        if offset > 0:
            safe_readline(f)    # drop first incomplete line
        lines = []
        line = f.readline()
        while line:
            line = line.replace(' ','')
            line = line.strip('\n')
            if not line:
                line = f.readline()
                continue
            lines.append(line)
            if f.tell() > end:
                break
            line = f.readline()
        return lines
    
    
def encode_file(path, workers):
    assert os.path.exists(path)
    results = []
    workers_thread = []
    pool = Pool(processes=workers)
    for i in range(workers):
        w = pool.apply_async(
            async_kd_tokenizer,
            (path, i, workers),
        )
        workers_thread.append(w)
    pool.close()
    pool.join()
    for w in workers_thread:
        result = w.get() 
        results += result
    return results    

#2==========================================
def sampling(percentage,pre_data,past_data):
    pre_dat = pd.read_csv(pre_data,header=None)
    print(pre_dat.head())
    pre_dat = pre_dat.sample(frac=percentage)
    print(pre_dat.head())
    with open(past_data,'w') as f:
        for line in range(len(pre_dat)):
            f.write(str(pre_dat.iloc[line].values[0])+'\n')
    print(len(pre_dat))


#3==============================================
# define problem properties
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
ANGLE_NAMES_LIST = ["PHI", "PSI", "THETA", "TAU"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

EXP_NAMES_LIST = ["ASA", "CN", "HSE_A_U", "HSE_A_D"]
EXP_MAXS_LIST = [330.0, 131.0, 76.0, 79.0]  # Maximums from our dataset
UPPER_LENGTH_LIMIT = 1024

def mul1(rows):
    idx,s=rows
    if "X" not in s[0]:
        return {'idx':str(idx),'s':s[0]}
    
    
def mul2(df):
    idx_space_list =[]
    seqs = ''
    csv_data = df
    for s in range(csv_data.shape[0] - 1):
        a = csv_data.iloc[s,1]
        b = csv_data.iloc[s+1,1]
        if a == b :
            continue
        else:
            idx_space_list.append(s)
    for a in range(csv_data.shape[0]):
        if a not in idx_space_list:
            seqs = seqs + csv_data.iloc[a,0]
        else:
            seqs = seqs + csv_data.iloc[a,0]
            seqs = seqs +' '
    return seqs

def read_fasta(fasta_files_dict_list):
    protein_names_list = []
    sequences_list = []
    for file in fasta_files_dict_list:
        if file != None :
            protein_names = []
            sequences = []
            name = file['idx']
            protein_names.append(name)
            sequences.append(file['s'])
            protein_names_list.append(protein_names)
            sequences_list.append(sequences)
    return protein_names_list, sequences_list


def fill_array_with_value(array: np.array, length_limit: int, value):
    array_length = len(array)

    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


def save_prediction_to_csv(resnames: str, pred_c: np.array,name):
    
    sequence_length = len(resnames)

    output_df = pd.DataFrame()
    output_df["resname"] = [r for r in resnames]

    def get_ss(one_hot):
        return [["C", "H", "E", "T", "G", "S", "I", "B"][idx] for idx in np.argmax(one_hot, axis=-1)]

    output_df["Q3"] = get_ss(pred_c[0][0])[:sequence_length]
    a = get_ss(pred_c[0][0])[:sequence_length]
    b = get_ss(pred_c[1][0])[:sequence_length]
    output_df["Q8"] = get_ss(pred_c[1][0])[:sequence_length]


    return output_df


def mul4(idx,residue_lists,pred_c):
    resnames = residue_lists[idx][0]
    pred_c = [pred_c[0][idx:idx+1], pred_c[1][idx:idx+1,:,:]]

    sequence_length = len(resnames)

    output_df = pd.DataFrame()
    output_df["resname"] = [r for r in resnames]

    def get_ss(one_hot):
        return [["C", "H", "E", "T", "G", "S", "I", "B"][idx] for idx in np.argmax(one_hot, axis=-1)]

    output_df["Q3"] = get_ss(pred_c[0][0])[:sequence_length]
    output_df["Q8"] = get_ss(pred_c[1][0])[:sequence_length]

    return output_df

def main(ensemble_c,fasta_files_dict_list):
    protein_names, residue_lists = read_fasta(fasta_files_dict_list)

    sequence_list =[]
    for resnames in residue_lists:


        sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames[0]], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)

        sequence_list.append(sequence)

    print(f"Generating prediction...")
    start = time.time()
    pred_c = ensemble_c.predict(np.array(sequence_list))
    tf.compat.v1.keras.backend.clear_session()
    print('推理时间：',time.time()-start)
    return residue_lists,protein_names,pred_c
    

def seg_by_no_to_seg_by_secondstructure(seg_by_no_data_path,seg_by_secondstructure_data_path,models_folder,chunksize,GPU,test,is_multipleproccess,name):
    with tf.device(f'/device:{GPU}'):
        ensemble_c = load_model(os.path.join(models_folder, "unet_c_ensemble"),compile=False)
        seg_by_no_data_chunk = pd.read_csv(seg_by_no_data_path,header=None,chunksize=chunksize)
        with open(seg_by_secondstructure_data_path,'w') as  c:
            for n,chunk in enumerate(seg_by_no_data_chunk) :
                sta = time.time()
                print(f'已完成{n*len(chunk)}条序列')
                

                #===============================================
                if is_multipleproccess == True:
                    # 多进程
                    pool1 = Pool()
                    res1 = pool1.map(mul1,chunk.iterrows())
                    pool1.close()
                    pool1.join()
                else :
                    #单进程
                    res1 = []
                    for i in chunk.iterrows():
                        res1.append(mul1(i))

                
                
                residue_lists,protein_names,pred_c = main(ensemble_c = ensemble_c,fasta_files_dict_list = res1 )




                df_list = []
                for idx in [i for i in range(len(protein_names))]:
                    df_list.append(save_prediction_to_csv(residue_lists[idx][0],
                                                          [pred_c[0][idx:idx+1], pred_c[1][idx:idx+1,:,:]],name))                

                #===============================================
                if is_multipleproccess == True:
                    #多进程
                    pool2 = Pool()
                    res2=pool2.map(mul2,df_list)
                    pool2.close()
                    pool2.join()
                else:
                    #单进程
                    res2 = []
                    for i in df_list:
                        res2.append(mul2(i))

                for d in res2:
                    c.write(d+'\n')
                print('完成1个chuank的时间:',time.time()-sta)
                
                if test ==  True:
                   break

#4
def split_txt():
    file_count=1
    url_list=[]

    with open('/tmp/luoxw/data_antibody/data/proprocessed_seg_by_no_all_seq_L.txt') as f:
        count1 = len(f.readlines())  
        LIMIT = count1 // 2

    with open('/tmp/luoxw/data_antibody/data/proprocessed_seg_by_no_all_seq_L.txt') as f:    
        for line in f:
            url_list.append(line)
            if len(url_list)<LIMIT:
                continue
            file_name='/tmp/luoxw/data_antibody/data/proprocessed_seg_by_no_50%_seq_L'+str(file_count)+'.txt'
            with open(file_name,'w') as file:
                for url in url_list[:-1]:
                    file.write(url)
                file.write(url_list[-1].strip())
                url_list=[]
                file_count+=1

    if url_list:
        file_name='/tmp/luoxw/data_antibody/data/proprocessed_seg_by_no_50%_seq_L'+str(file_count)+'.txt'
        with open(file_name,'w') as file:
            for url in url_list:
                file.write(url)

    print('done')

#5
def XXXX(pre_data,past_data):
    with open(pre_data,'r') as f:
        with open(past_data,'w') as w:
            for i in f.readlines():
                if 'X' not in i:
                    w.write(i)


#6
def merge_txt(input_data,output_data):
    with open(output_data,'w') as w:
        for i in input_data:
            with open(i,'r') as f:
                for s in f.readlines():
                    w.write(s)


#7
def seg_by_region_to_seg_by_no(input_data):
    result = []
    with open(input_data,'r') as f:
        seq = f.readlines()
        for i in seq:
            result.append(i.replace(' ','').strip('\n'))
    return result


#8
def split_data(input_data_path,output_data_path_train,output_data_path_test):
    raw_datasets = []
    with open(input_data_path,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            raw_datasets.append(line)
    train_dataset, eva_dataset = train_test_split(raw_datasets, test_size =0.001,random_state=19930606)
    with open(output_data_path_train,'w') as f:
        for s in train_dataset:
            f.write(s)
            f.write('\n')
    
    with open(output_data_path_test,'w') as f:
        for s in eva_dataset:
            f.write(s)
            f.write('\n') 


def fun1_for_Datasets(l):
    return len(tokenizer.encode(l, add_special_tokens=True))


def fun2(l):
    l = tokenizer.encode(l, add_special_tokens=True)
    pad = [-999]*(max_length_token-len(l))
    l.extend(pad) 
    return l

def seq_to_tokenid(input_txt,vocab_path,h5_path,save_tokenid_h5 = True,save_mask_h5 = False):
    global tokenizer
    tokenizer = BertTokenizer(vocab_path,do_lower_case=False)
    start =time.time()
    if save_tokenid_h5 == True:
        with open(input_txt,encoding='utf-8') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]#secondstructure
        print(lines[0:10])

        #多进程
        pool1 = Pool()
        result1 = pool1.map(fun1_for_Datasets,lines)
        pool1.close()
        pool1.join()
        print(result1[0:10])
        global max_length_token 
        max_length_token = max(result1)
        del result1,pool1
        print('最长的token:',max_length_token)

        num_length = 3000000
        # times = len(result1)
        pool2 = Pool()
        for i in range(0,len(lines),num_length):
            res = lines[i:i+num_length]
            result2 = pool2.map(fun2,res)
            print(i)
            if i == 0:
                b=h5py.File(h5_path,"w")
                tokenid = b.create_dataset('tokenid',data = result2,maxshape=(None,max_length_token),dtype=np.int32,chunks=True,compression='gzip', compression_opts=0)#
            else:
                b=h5py.File(h5_path,"a")
                tokenid = b['tokenid']
                le = tokenid.shape[0]
                tokenid.resize([le+len(result2),max_length_token])
                tokenid[le:le+len(result2),:] = result2
                print(tokenid.shape)
        
        # del result1

        print(time.time()-start)

    if save_mask_h5 == True:
        dataset = antibody_normalDataset(
                # tokenizer = self.tokenizer,
                file_path = np_tokenid 
                # block_size = self._get_max_seq_length(raw_data_path=self.train_data),  # maximum sequence length
            )
        max_length = max_length_token + 10

        data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15,pad_to_multiple_of = max_length,
            )

        loader = DataLoader(dataset=dataset,batch_size=1,collate_fn=data_collator)
        ite = iter(loader)

        
        np_tokenid = np.zeros((len(dataset),3,max_length),dtype=numpy.int32)

        for i in range(len(dataset)):
            its = next(ite)
            print(i)
            np_tokenid[i,0,:] = its['input_ids'][0,:]
            np_tokenid[i,1,:] = its['attention_mask'][0,:]
            np_tokenid[i,2,:] = its['labels'][0,:]
        
        print(np_tokenid.shape)
        f.create_dataset('tokenid_mask',data=np_tokenid)
        # np.save(output_npy_mask,np_tokenid)

    for fkey in b.keys():
        print(fkey)
        print(b[fkey].name)
        print(b[fkey].shape)
        # print(b[fkey].value)
    print(b['tokenid'][50,:])

    b.close()


    

    



