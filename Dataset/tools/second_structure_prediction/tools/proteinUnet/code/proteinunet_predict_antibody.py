import argparse
import pandas as pd
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import time



# def parse():
#     parser = argparse.ArgumentParser(description='Run ProteinUnet prediction')

#     parser.add_argument('--input',
#                         action="store",
#                         dest="input",
#                         help='FASTA filepath')
#     return parser.parse_args()

fasta_files_path = '/home/luoxw/antibody_doctor_project/second_structure_prediction/tools/proteinUnet/data/fasta_files_1'
file_lists = '/home/luoxw/antibody_doctor_project/second_structure_prediction/tools/proteinUnet/data/file_lists'
models_folder = "/home/luoxw/antibody_doctor_project/second_structure_prediction/tools/proteinUnet/data/models"
output_folder = "/home/luoxw/antibody_doctor_project/second_structure_prediction/tools/proteinUnet/results"

# define problem properties
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
ANGLE_NAMES_LIST = ["PHI", "PSI", "THETA", "TAU"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

EXP_NAMES_LIST = ["ASA", "CN", "HSE_A_U", "HSE_A_D"]
EXP_MAXS_LIST = [330.0, 131.0, 76.0, 79.0]  # Maximums from our dataset
UPPER_LENGTH_LIMIT = 1024


def read_list(file_name):
    """
    read a text file to get the list of elements
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text

def read_fasta(file_lists,fasta_files_path):
    protein_names_list = []
    sequences_list = []
    for file in os.listdir(fasta_files_path):
        protein_names = []
        sequences = []
        with open(os.path.join(fasta_files_path,file), 'r') as reader:
            name = reader.readline()
            while name.startswith((">", ";")):
                protein_names.append(name[1:].strip())
                sequences.append(reader.readline())
                name = reader.readline()
        protein_names_list.append(protein_names)
        sequences_list.append(sequences)
        
    return protein_names_list, sequences_list


def fill_array_with_value(array: np.array, length_limit: int, value):
    array_length = len(array)

    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


def save_prediction_to_csv(resnames: str, pred_c: np.array, pred_r: np. array, output_path: str):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving predictions to {output_folder}")
    sequence_length = len(resnames)

    output_df = pd.DataFrame()
    output_df["resname"] = [r for r in resnames]

    def get_ss(one_hot):
        return [SS_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    output_df["Q3"] = get_ss(pred_c[0][0])[:sequence_length]
    output_df["Q8"] = get_ss(pred_c[1][0])[:sequence_length]

    for i in range(4):
        sin_cos_list = pred_r[i][0][:sequence_length]
        sin_cos_list = sin_cos_list * 2 - 1
        arctan = np.rad2deg(np.arctan2(sin_cos_list[:, 0], sin_cos_list[:, 1]))
        output_df[ANGLE_NAMES_LIST[i]] = arctan

        exp_list = pred_r[i + 4][0][:sequence_length] * EXP_MAXS_LIST[i]
        output_df[EXP_NAMES_LIST[i]] = exp_list

    output_df.to_csv(output_path)


def main():
    protein_names, residue_lists = read_fasta(file_lists,fasta_files_path)
    
    ensemble_c = load_model(os.path.join(models_folder, "unet_c_ensemble"))
    ensemble_r = load_model(os.path.join(models_folder, "unet_r_ensemble"))

    sequence_list =[]
    for resnames in residue_lists:
        
        # print(resnames)
        # if len(resnames) > UPPER_LENGTH_LIMIT:
        #     print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues!")
        #     continue

        sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames[0]], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)

        sequence_list.append(sequence)

    # print(sequence_list)

    print(f"Generating prediction...")
    # start = time.time()
    pred_c = ensemble_c.predict(np.array(sequence_list))
    pred_r = ensemble_r.predict(np.array(sequence_list))
    # print(time.time()-start)
    # print(pred_c,pred_r)
    # print(pred_c[0].shape,'\n',pred_c[1].shape,'\n',pred_r[0].shape)
    
    for idx in range(len(protein_names)):
        save_prediction_to_csv(residue_lists[idx][0], [pred_c[0][idx:idx+1], pred_c[1][idx:idx+1,:,:]],[x[idx:idx+1,:,:] for x in pred_r ], os.path.join(output_folder, protein_names[idx][0] + ".csv"))


if __name__ == '__main__':
    main()
