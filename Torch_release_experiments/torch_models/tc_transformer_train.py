from json import load
import re
import numpy as np
from tqdm.notebook import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from voice_engine import vc_arch



def training_engine_and_evaluation_architecture(model, tdloader, device, patience,optimizer, loss_cat, vdloader, scheduler):
    """
    Training engine architecture for torch transformer model
    
    """
    epochs = 10
    patience = 3
    dump_path = 'model_dumps/transformer.pt'

    res = {
        'training-loss' : [],
        'validation-loss' : [],
        'training-acurracy': [],
        'validation-accuracy' : []
    }

    global best_validation_loss
    best_validation_loss = np.inf
    for ep in range(epochs):
        print(f'------------------Training in epoch {ep+1}--------------------------------->')
        # vc_arch(f'------------------Training in epoch {ep+1}--------------------------------->')
        training_loss = 0
        validation_loss = 0
        training_accuracy = 0
        validation_accuracy = 0
        model.train()
        for bcth in tqdm(tdloader, total=len(tdloader)):
            text_sq , ln_nums , total_lns, lbls = bcth
            input_id, attn_mask = text_sq['input_ids'].to(device), text_sq['attention_mask'].to(device)
            ln_nums, total_lns, lbls = ln_nums.to(device), total_lns.to(device), lbls.to(device)
            _ins = {'input_ids':input_id, 'attention_mask': attn_mask}
            optimizer.zero_grad()
            ans = model.forward(_ins, ln_nums, total_lns)
            loss = loss_cat(ans, lbls)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pred_class = torch.softmax(ans, dim=1).argmax(dim=1)
            training_accuracy += (pred_class == lbls).sum().item()/len(ans)

        # evaluation on validation data
        model.eval()
        for btch in tqdm(vdloader, total=len(vdloader)):
            text_sq , ln_nums , total_lns, lbls = btch
            input_id, attn_mask = text_sq['input_ids'].to(device), text_sq['attention_mask'].to(device)
            ln_nums, total_lns, lbls = ln_nums.to(device), total_lns.to(device), lbls.to(device)
            _ins = {'input_ids':input_id, 'attention_mask': attn_mask}
            ans = model(_ins, ln_nums, total_lns)
            loss = loss_cat(ans, lbls)
            validation_loss += loss.item()
            pred_class = torch.softmax(ans, dim=1).argmax(dim=1)
            validation_accuracy += (pred_class == lbls).sum().item()/len(ans)

        # metrics 
        training_loss = training_loss / len(tdloader)
        validation_loss = validation_loss / len(vdloader)

        # accuracy 
        training_accuracy = training_accuracy / len(tdloader)
        validation_accuracy =  validation_accuracy / len(vdloader)

        scheduler.step(validation_loss)

        # TODO :: ADD callbacks and early stopping implementation in transformers
        if validation_loss < best_validation_loss :
            best_validation_loss =  validation_loss
            print('The model is being saved please wait')
            # vc_arch('The model is being saved please wait')
            torch.save(model.state_dict(), dump_path)
            _patience = patience
        else: 
            _patience -= 1
            if not _patience: 
                print('Stopping early the limit for patience has been reached')
                # vc_arch('Stopping early the limit for patience has been reached')
                break

        res['training-loss'].append(training_loss)
        res['validation-loss'].append(validation_loss)
        res['training-acurracy'].append(training_accuracy)
        res['validation-accuracy'].append(validation_accuracy)

        print(f"The learning rate used for this model : {optimizer.param_groups[0]['lr']:.2E}")
        # Logging the parameters
        print(
            f"Training loss : {training_loss:.6f},\t"
            f"Training Accuracy : {training_accuracy:.2f},\t"
            f"Validation loss :  {validation_loss:.6f}, \t"
            f"Validation Accuracy : {validation_accuracy:.2f}, \t"
        )


