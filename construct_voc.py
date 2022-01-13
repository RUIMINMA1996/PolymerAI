"""
© Copyright 2021
RUIMIN MA
"""
import sys
from rdkit import Chem
import re

def canonicalize_smiles_from_files(fname):
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 10 == 0:
                print('{} lines processed.'.format(i))
            smiles = line.split(" ")[0]
            try:
                mol = Chem.MolFromSmiles(smiles)
                smiles_list.append(Chem.MolToSmiles(mol))
            except:
                pass
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list

def construct_vocabulary(smiles_list):
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]
    print("Number of characters: {}".format(len(add_chars)))
    with open('data/voc', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def write_smiles_to_file(smiles_list, fname):
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_files(smiles_file)
    print('Constructing vocabulary...')
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")