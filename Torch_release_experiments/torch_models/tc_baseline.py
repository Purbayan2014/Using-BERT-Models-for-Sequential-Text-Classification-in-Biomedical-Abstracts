import re
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class tc_baseline(object):
    """
    First Baseline Model of torch/release branch
    """

    def __init__(self, model, device, loss_func, optimizer,scheduler, dump_path) -> None:
        self.model = model
        self.device = device 
        self.loss_func = loss_func
        self.optimizer =  optimizer
        self.scheduler = scheduler
        self.dump_path = dump_path

    def training_architecture(self, dloader):
        """The training architecture for the torch baseline model

        Args:
            dloader (object) : Dataloader
        Returns:
        loss (int) : loss evaluated while training the model
        accuracy (int) : accuracy evaluated while training the model        
        """

        self.model.train()
        loss = 0.0
        accuracy = 0.0

        # iterating over the batches of training data
        for idx, btch in tqdm(enumerate(dloader), total=len(dloader)):
            # setting up a common baseline device
            btch  = [data.to(self.device) for data in btch]
            # inputs and outputs
            x, y = btch[:-1], btch[-1]
            # setting up the optimizer for resetting the gradients
            self.optimizer.zero_grad()
            # forward propagation 
            fwd = self.model(x)
            # loss func for the model
            ls_fn = self.loss_func(fwd, y)
            # backward propagation
            ls_fn.backward()
            # now update the weights after the backward prop
            self.optimizer.step()

            # metrics
            loss += (ls_fn.detach().item() - loss) / (idx + 1)
            accuracy += self.cal_acc(fwd, y)
        
        return loss, accuracy

    def training_engine(self, epochs, patience, train_dl, val_dl):
        """
        Training engine for the torch baseline model

        Args:
            epochs (int) : 
        """

    


