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
from py_help import torch_helper as tc_helper

class Model1(nn.Module):
    """Torch LSTM model with embeddings
    """
    def __init__(self, voc_size,hd_dim, num_lyrs, ln_output, num_classes, embed_dm, pad_idx = 0):
        super(Model1, self).__init__()

        # embeddings 
        self.embed = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dm)
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
        X = tc_helper().last_relavent(states=res, seq_lens=len_sequences)
        # forward props results 
        X = self.fcd1(x)
        X = self.drop_fn(x)
        X = self.fcd2(x)
        return X