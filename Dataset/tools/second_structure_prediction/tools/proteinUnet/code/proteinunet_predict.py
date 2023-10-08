import argparse
import pandas as pd
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model



def parse():
    parser = argparse.ArgumentParser(description='Run ProteinUnet prediction')

    parser.add_argument('--input',
                        action="store",
                        dest="input",
                        help='FASTA filepath')
    return parser.parse_args()


models_folder = "../data/models"
output_folder = "../results"

# define problem properties
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
ANGLE_NAMES_LIST = ["PHI", "PSI", "THETA", "TAU"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

EXP_NAMES_LIST = ["ASA", "CN", "HSE_A_U", "HSE_A_D"]
EXP_MAXS_LIST = [330.0, 131.0, 76.0, 79.0]  # Maximums from our dataset
UPPER_LENGTH_LIMIT = 1024


def read_fasta(filepath: str):
    protein_names = []
    sequences = []
    with open(filepath, 'r') as reader:
        name = reader.readline()
        while name.startswith((">", ";")):
            protein_names.append(name[1:].strip())
            sequences.append(reader.readline())
            name = reader.readline()
    return protein_names, sequences


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
    protein_names, residue_lists = read_fasta(parse().input)
    
    ensemble_c = load_model(os.path.join(models_folder, "unet_c_ensemble"))
    ensemble_r = load_model(os.path.join(models_folder, "unet_r_ensemble"))

    for protein_name, resnames in zip(protein_names, residue_lists):
        print(protein_name,resnames)
        if len(resnames) > UPPER_LENGTH_LIMIT:
            print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues!")
            continue

        sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)
        print(sequence)
        print(f"Generating prediction...")
        pred_c = ensemble_c.predict(np.array([sequence,sequence,sequence]))
        pred_r = ensemble_r.predict(np.array([sequence,sequence,sequence]))

        print(pred_c[0].shape,'\n',pred_c[1].shape,'\n',pred_r[0].shape)


        save_prediction_to_csv(resnames, pred_c, pred_r, os.path.join(output_folder, protein_name + ".csv"))


if __name__ == '__main__':
    main()
