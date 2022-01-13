"""
© Copyright 2021
RUIMIN MA
"""

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem, rdBase
from collections import Counter
import numpy as np
rdBase.DisableLog('rdApp.error')
import joblib

class activity_model():
    def __init__(self):
        self.reg = joblib.load("regressor/rng_100.pkl")

    def __call__(self, smile):
    
        mol = Chem.MolFromSmiles(smile)
        if mol: # RDKit validation
            smi_ = Chem.MolToSmiles(mol)
            smile_list = [char for char in smi_]
            smile_dict = Counter(smile_list)
            number_of_star = smile_dict['*']
            if number_of_star == 2: # number of star == 2
                atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                for an in atomic_number:
                    try:
                        if mol.GetAtomWithIdx(an).GetSymbol() == "*":
                            atom = mol.GetAtomWithIdx(an)
                            number_of_bonds = len(atom.GetNeighbors()[-1].GetBonds())
                            if number_of_bonds == 2: # number of bonds to "*" == 1
                                try:
                                    embeddings = activity_model.polymer_embeddings(smi_)
                                    score = self.reg.predict(embeddings)
                                    if float(score) >= 0.40:
                                        return float(score)
                                    else:
                                        return 0.000
                                except:
                                    pass
                    except:
                        pass

        return 0.000

        # mol = Chem.MolFromSmiles(smile)
        # if mol:
        #     smile = Chem.MolToSmiles(mol)
        #     try:
        #         embeddings = activity_model.polymer_embeddings(smile)
        #         score = self.reg.predict(embeddings)
        #         if float(score) >= 0.40:
        #             return float(score)
        #         else:
        #             return 0.0000000001
        #     except:
        #         pass
        # return 0.0

    @classmethod
    def polymer_embeddings(cls, smile):
        sentences = []
        model = word2vec.Word2Vec.load('regressor/POLYINFO_PI1M.pkl')
        sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(smile), 1))
        sentences.append(sentence)
        PE_model = [DfVec(x) for x in sentences2vec(sentences, model, unseen='UNK')]
        PE = np.array([x.vec.tolist() for x in PE_model])
        return PE

def get_scoring_function(scoring_function):
    scoring_function_classes = [activity_model]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    # for k, v in kwargs.items():
    #     if k in scoring_function_class.kwargs:
    #         setattr(scoring_function_class, k, v)

    return Singleprocessing(scoring_function=scoring_function_class)


class Singleprocessing():
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()

    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)