"""
© Copyright 2021
RUIMIN MA
"""

import torch
import pickle
import pandas as pd
import numpy as np
import time
import os
from shutil import copyfile

from rnn import RNN
from voc import Vocabulary
from score_function import get_scoring_function
from utils import Variable


def unique(arr):
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def train_agent(restore_agent_from='data/Prior.ckpt',
                scoring_function='activity_model',
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=1, sigma=60):

    voc = Vocabulary(init_from_file="data/voc")
    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt'))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt', map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)
    scoring_function = get_scoring_function(scoring_function=scoring_function)
    step_score = [[], []]
    print("Model initialized, starting training...")

    for step in range(n_steps):
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = []
        for seq in seqs.cpu().numpy():
            smiles.append(voc.decode(seq))
        score = scoring_function(smiles)

        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        loss = loss.mean()

        regularization = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print out information during the training
        print("Agent    Prior    Target    Score        SMILES")
        for i in range(10):
            print("{:6.2f}  {:6.2f}  {:6.2f}  {:6.2f}    {}".format(agent_likelihood[i],
                                                                    prior_likelihood[i],
                                                                    augmented_likelihood[i],
                                                                    score[i],
                                                                    smiles[i]))

        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))


    if not save_dir:
        save_dir = 'experiments/baseline/agent_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('agent_baseline.py', os.path.join(save_dir, "agent_baseline.py"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'agent_baseline.ckpt'))

    step_score_data = pd.DataFrame({
        'Step': step_score[0],
        'Score': step_score[1]
    })
    step_score_data.to_csv(os.path.join(save_dir, "step_score.csv"), index=None)

    seqs, agent_likelihood, entropy = Agent.sample(256)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    score = scoring_function(smiles)
    with open(os.path.join(save_dir, "sampled"), 'w') as f:
        f.write("SMILES  Score  PriorLogP\n")
        for s, sc, pri in zip(smiles, score, prior_likelihood):
            f.write("{}  {:5.2f}  {:6.2f}\n".format(s, sc, pri))

if __name__ == '__main__':
    train_agent()