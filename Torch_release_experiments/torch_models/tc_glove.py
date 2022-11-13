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

class Model2(nn.Module):
    """Torch LSTM model with embeddings
    """
    def __init__(self, voc_size,hd_dim, num_lyrs, ln_output, num_classes, embed_dm, pad_idx = 0,pre_embed=None):
        super(Model2, self).__init__()
        self.tc_utils = TC_UTILS()
        # glove embeddings initializations
        if pre_embed is None:
            self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dm)
        else:
            pre_embed = torch.from_numpy(pre_embed).float()
            self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dm, _weight=pre_embed, padding_idx=pad_idx)

        # LSTM lyrs 
        self.lstm = nn.LSTM(embed_dm, hd_dim, num_layers=num_lyrs, batch_first=True, bidirectional=True)
        # forward propagation layers
        self.fcd1 = nn.Linear(2*hd_dim, ln_output)
        self.fcd2 = nn.Linear(ln_output, num_classes)
        self.drop_fn = nn.Dropout(0.3)

    def forward(self, x):
        """
        Method for forward propagation for torch Model 1

        Args:
            x : inputs
        """
        x_inputs, len_sequences = x
        x_inputs = self.embed(x_inputs)

        # outputs from RNN
        res , backward_props = self.lstm(x_inputs)
        X = self.tc_utils.last_relevant(hd_states=res, seq_lens=len_sequences)
        # forward props results 
        X = F.relu(self.fcd1(X))
        X = self.drop_fn(X)
        X = self.fcd2(X)
        return X