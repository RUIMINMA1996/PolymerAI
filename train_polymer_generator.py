"""
© Copyright 2021
RUIMIN MA
"""
import torch
from voc import Vocabulary
from exp import MolData
from torch.utils.data import DataLoader
from rnn import RNN
from tqdm import tqdm
from rdkit import Chem
from utils import Variable, decrease_learning_rate
import matplotlib.pyplot as plt


def main(restore_from=None, visualize=False):
    # read vocbulary from a file
    voc = Vocabulary(init_from_file="data/voc")

    # create a dataset from a smiles file
    moldata = MolData("data/mols_filtered.smi", voc)
    data = DataLoader(moldata, batch_size=10, shuffle=True, drop_last=True,
                     collate_fn=MolData.collate_fn)

    agent = RNN(voc)

    # can restore from a saved RNN
    if restore_from:
        agent.rnn.load_state_dict(torch.load(restore_from, map_location=torch.device('cpu')))

    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=0.001)
    torch.autograd.set_detect_anomaly(True)
    valid_ratios = list()
    for epoch in range(1, 2):
        for step, batch in tqdm(enumerate(data), total=len(data)):
            # sample from DataLoader
            seqs = batch.long()

            # calculate loss
            log_p, _ = agent.likelihood(seqs)
            loss = - log_p.mean()
            # print(loss)

            # calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # every n steps we decrease learning rate and print out some information, n can be customized
            if step % 5 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("#" * 50)
                tqdm.write("Epoch {:3d} step {:3d} loss: {:5.2f}\n".format(epoch, step, loss.data))
                seqs, likelihood, _ = agent.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                    valid_ratio = 100 * valid / len(seqs)
                    valid_ratios.append(valid_ratio)
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("#" * 50 + "\n")
                    torch.save(agent.rnn.state_dict(), "data/Prior.ckpt")
        torch.save(agent.rnn.state_dict(), "data/Prior.ckpt")
    if visualize:
        plt.plot(range(len(valid_ratios)), valid_ratios, color='red', linewidth=5)
        plt.savefig('/Users/ruiminma/Desktop/validratio.png', bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    main(restore_from='data/Prior.ckpt', visualize=True)