
# Polymer Generator
A generative model trained on PI1M, which can produce polymers with high validness.


## Notes
Latested update: 12/8/2020
## Version control
* Python 3.6
* PyTorch 1.6.0
* RDKit 2020.03.1.0
* Scikit-Learn 0.23.2

## Usage

To train an agent that can be used to generate polymers via imitation learning:

* First filter the p-SMILES and construct a vocabulary from the remaining sequences. 
` python construct_voc.py PI1M_test.csv`   - Will generate data/mols_filtered.smi and data/voc.

* Then use `python imitation_learning.py` to train the polymer generator. 

