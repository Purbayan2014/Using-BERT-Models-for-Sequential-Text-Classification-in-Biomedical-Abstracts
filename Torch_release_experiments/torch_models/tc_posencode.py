from json import load
import re
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Utility.voice_engine import vc_arch
from torch_utils.tc_utils import TC_UTILS

class Model3(nn.Module):
    """
        TORCH LSTM model with positional encodings
    """
    def __init__(self, voc_size, hd_dim, num_lyrs, ln_output, num_classes, embed_dm, pad_idx = 0, pre_embed = None):
        self.tc_utils = TC_UTILS()
        #  init the embeddings
        if pre_embed is None:
            self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dm)
        else:
            pre_embed = torch.from_numpy(pre_embed).float()
            self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dm, _weight=pre_embed, padding_idx=pad_idx)
        # creating the lstm layers
        self.lstm = nn.LSTM(embed_dm, hd_dim, num_layers=num_lyrs, batch_first=True, bidirectional=True)
        # forward propagation layers
        self.fcd1 = nn.Linear(2*hd_dim, ln_output) # for dealing with text
        self.fcd2 = nn.Linear(31, 64) # for dealing with line nos
        self.fcd3 = nn.Linear(31, 64) # for dealing with total line
        self.fcd4 = nn.Linear((64+64+ln_output), num_lyrs) # concatenating all those in the a single total layer as the final
        self.drop_fn = nn.Dropout(0.3)


    def forward(self, x):
        """
        Method for forward propagation for model 2
        Args:
            x : inputs
        """
        x_inputs, len_sequences, line_numbers, total_lines = x
        x_inputs = self.embed(x_inputs)

        # outputs from RNN
        res, backward_props = self.lstm(x_inputs)
        X = self.tc_utils.last_relevant(hd_states=res, seq_lens=len_sequences)
        # forward props result
        X = F.relu(self.fcd1(X))
        X_1 = self.fcd2(line_numbers)
        X_2 = F.relu(self.fcd3(total_lines))

        # concateting the layers
        x_concat_lyr = torch.cat((X, X_1, X_2), dim=1)
        x_concat_lyr = self.drop_fn(x_concat_lyr)
        x_concat_lyr = self.fcd2(x_concat_lyr)
        return x_concat_lyr
