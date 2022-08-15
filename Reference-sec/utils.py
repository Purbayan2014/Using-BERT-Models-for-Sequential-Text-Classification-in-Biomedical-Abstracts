#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 12:27:16 2022

       /\         markins@archcraft
      /  \        os     Archcraft
     /\   \       host   G3 3579
    /      \      kernel 5.18.14-arch1-1
   /   ,,   \     uptime 45m
  /   |  |  -\    pkgs   1558
 /_-''    ''-_\   memory 2062M / 15844M

@author: markins
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime
import zipfile
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def loss_plotter(history):
    loss = history.history["loss"]
    val_loss =  history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    
    epochs = range(len(history
                       .history["loss"]))
    
    plt.plot(epochs, loss, label="Training data")
    plt.plot(epochs, val_loss, label="Testing data")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label="Training data")
    plt.plot(epochs, val_acc, label="Testing data")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    

def image_preprocessor(filename, img_shape=224, scale=True):
    """ 
    Reads the image and turns into a tensor and reshapes it
    (224,224,3)
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    
    # resizing 
    img = tf.image.resize(img, [img_shape, img_shape])
    
    if scale :
        return img / 255.
    else :
        return img
    

def pretty_matrix(y_true,y_pred,
                  classes=None, figsize=(10,10), text_size=15,
                  norm=False, savefig=False):
    """ 
    Just a pretty confusion matrix 
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    
    # plot them
    fig, ax = plt.subplot(figsize=figsize)
    cax = ax.natshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # edge cases for multi classes
    if classes:
        labels = classes
    else :
        labels = np.arange(cm.shape[0])
        
    
    ax.set(title="Pretty confusion matrix",
           xlabel="Prediction labels",
           ylabel="True labels",
           xticks=np.arange(n_classes),
           y_ticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    # threshold 
    threshold = (cm.max() + cm.min()) / 2
    
    # TODO --- plot the text on the cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i,j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i , f"{cm[i,j]}",
                     horizontalalignment="center",
                     color="white" if cm[i,j] > threshold else "black",
                     size=text_size)
            
    if savefig:
        fig.save("pretty_confusion_matrix.png")
            
def pred_image_plt_plot(model, filename, class_names):
    img = image_preprocessor(filename)

    pred = model.predict(tf.expand_dims(img, axis=0))

    # prediction
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
        
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    

def create_tensorboard_callback(dir_name, exp_name):
    """ 
    Generate tensorboard callbacks log
    """
    log_dir = dir_name + "/" + exp_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Gathering the log data and saving in : {log_dir}")
    return tensorboard_callback


def histor_cmp(org_hs, nw_hs, init_epochs=5):
    """
    Compares the history of tf models
    """    
    
    acc = org_hs.history["accuracy"]
    loss = org_hs.history["loss"]
    
    val_acc = acc + nw_hs.history["val_accuracy"]
    val_loss = loss + nw_hs.history["val_loss"]
    
    total_acc = acc + nw_hs.history["accuracy"]
    total_loss = loss + nw_hs.history["loss"]
    
    total_val_Ac = val_acc + nw_hs.history["val_accuracy"]
    total_val_loss = val_loss + nw_hs.history["loss"]
    
    # TODO -- plot
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label="Training accuracy")
    plt.plot(total_val_Ac, label="valiation accuracy")
    plt.plot([init_epochs-1, init_epochs-1],
             plt.ylim(), label="Starting the Fine Tuning") # TODO reshift the plot around the epochs
    plt.legend(loc='lower right')
    plt.title("Training and Validation Accuracy")
    
    plt.subplot(2,1,2)
    plt.plot(total_loss, label="Training data")
    plt.plot(total_val_loss, label="Validation loss")
    plt.plot([init_epochs-1, init_epochs-1],
             plt.ylim(), label="Starting the Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel('epoch')
    plt.show()
    
    
def unzip_data(filename):
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()
    
  
def walker_dt_dir(dir_path):
    """
    Traverses and generates the log about the file and filenames
    """
    for dir_path , dir_name, filename in os.walk(dir_path):
        print(f"There are {len(dir_name)} directoriesand {len(filename)} images/files in '{dir_path}' .")
        
def evaluate_bin_class_model(y_true, y_pred):
    """
    Evaluating metrics for binary classification
    """
    model_acc = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1 , _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_res = {"accuracy" : model_acc,
                 "precision" : model_precision,
                 "recall" : model_recall,
                 "f1" : model_f1}
    return model_res