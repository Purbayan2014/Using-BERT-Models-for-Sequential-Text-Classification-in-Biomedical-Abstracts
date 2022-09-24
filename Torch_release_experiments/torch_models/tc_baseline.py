from json import load
import re
import numpy as np
from tqdm import tqdm
from symbol import flow_stmt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Utility.voice_engine import vc_arch

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
            epochs (int) : The nos of epochs the model has to be trained for 
            patience (int) : This parameter counts the nos of validation checks to wait if there are no improvement
            train_dl (dataloader object) : The training dataloader 
            val_dl (dataloader object) :  The tesing dataloader
        Returns : 
            appt_model : Returns the best model after evaluating the custom parameters that has been provided
        """
        appt_val_loss = np.inf
        for ep in range(epochs):
            print(f"<----------------------Training in EPOCH: {ep+1} ------------------------>")
            vc_arch(f"<----------------------Training in EPOCH: {ep+1} ------------------------>")
            training_loss , training_accuracy = self.training_architecture(dloader = train_dl)
            validation_loss , validation_accuracy = self.evaluation_architecture(dloader = val_dl)
            self.scheduler.step(validation_loss)

            # implementation of the early stopping in torch models
            if validation_loss < appt_val_loss:
                appt_val_loss =  validation_loss
                appt_model = self.model
                if self.dump_path is not None:
                    print(f'Dumping the model into the system in {self.dump_path}')
                     # reseting the patience 
                    _patience = patience
                else:
                    _patience -= 1
                if not _patience:
                    print("Patience state is zero !!!!! \n\n Stopping early")
                    vc_arch("Patience state is zero !!!!! \n\n Stopping early")
                    break

                training_accuracy = training_accuracy / len(train_dl)
                validation_accuracy = validation_accuracy / len(val_dl)

                # Logging the parameters
                print(
                    f"Training loss : {training_loss:,.3f},\t"
                    f"Training Accuracy : {training_accuracy:,.3f},\t"
                    f"Validation loss :  {validation_loss:,.3f}, \t"
                    f"Validation Accuracy : {validation_accuracy:,.3f}, \t"
                    f"Learning-rate : {self.optimizer.param_groups[0]['learning-rate']:,.3f}, \t"
                    f"Patience : {patience}"
                    "\n"
                    )
            return appt_model
        
    def evaluation_architecture(self, dloader):
        """
        The evaluation architecture for the torch baseline model

        Args:
            dloader (dataloader object) : The dataloader object 

        Returns : 
            loss, accurracy , true_values [list] , predicted_values [list] -- > evaluation metrics generated from the
            architecture 
        """    
        self.model.eval()
        loss = 0.0
        accuracy = 0.0
        true_values , predicted_values = [], []

        # iteration over the validation batches
        with torch.inference_model():
            for idx , btch in tqdm(enumerate(dloader), total=len(dloader)):

                # setting up a common device architecture
                btch = [data.to(self.device) for data in btch]
                x, y_true  = btch[:-1], btch[-1]
                # forward propagation 
                fwd = self.model(x)
                loss_fn = self.loss_func(fwd)
                # metrics 
                loss += (loss_fn - loss) / (idx + 1)
                accuracy += self.cal_acc(fwd, y_true)
                # grabbing the results
                prd = F.softmax(fwd).cpu().numpy()
                predicted_values.extend(prd)
                true_values.extend(y_true.cpu().numpy())
            
        return loss, accuracy, np.vstack(true_values), np.vstack(predicted_values)

    
    def prediction_architecture(self, dloader):
        """
        The prediction architecture for the torch baseline model

        Args:
            dloader (dloader object) : The dataloader object
        
        Returns : 
            predicted_values [list] = List of predicted values 
        """
        self.model.eval()
        predicted_vals = []

        # iteration over the validation batches 
        with torch.inference_model():
            for idx, btch in tqdm(enumerate(dloader),  total=len(dloader)):

                # forward propagation with inputs 
                x, y = btch[:-1], btch[-1]
                fwd = self.model(x)

                # grabbing the outputs
                predicted_val = F.softmax(fwd).cpu().numpy()
                predicted_vals.extend(predicted_val)

        return np.vstack(predicted_vals)

    
    def cal_acc(self, preds, target_labels):
        """
        Calculation of the accuracy of the baseline model of torch

        Args:
            Predicted values and Targetted labels
        
        Returns : 
            training_accuracy 
        """
        pred_class = torch.softmax(pred, dim=1).argmax(dim=1)
        training_accuray = (pred_class == target_labels).sum().item()/len(preds)
        return training_accuray





    


