#-*-coding:utf-8-*-

r"""
attention注意力机制层
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()

USE_CUDA = True if torch.cuda.is_available() else False

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method  # 选择不同的score计算方式
        self.hidden_size = hidden_size  # 隐藏层大小

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            nn.init.xavier_normal_(self.attn.weight)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            nn.init.xavier_normal_(self.attn.weight)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: [max_len, batch_size]
        :param encoder_outputs: [max_len, batch_size, hidden_size]
        :return:
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)).to(config.device)  # B x S

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output).squeeze(0)
            energy = hidden.squeeze(0).dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1)).squeeze(0)
            energy = self.v.squeeze(0).dot(energy)
            return energy