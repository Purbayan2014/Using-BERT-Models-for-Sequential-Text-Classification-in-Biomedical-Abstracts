from operator import le
import re
import numpy as np
import json as json_utils
from collections import Counter
from more_itertools import take 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class torch_helper:
    """Common utility class for torch experiments
    """
    
    def __init__(self):
        self.lb_encoder = self.lb_encoder()
        self.ct_tokenzr = self.ct_tokenzr()
#         self.CustomDataset = self.CustomDataset(text_seq="text_seq", l_num=56, tot_ln=6656, target=451, toknzer=object)

    def get_lines(self,filename):
        """Returns the file contents

        Args:
            filename (string): Name of the file
        """
        with open(filename, mode="r") as data:
            return data.readlines()

    def split_char_embedding(self, string):
      """
      Method used to split the characters of the string for
      initiating the character level embedding

      Args : 
         string (str) : The sentence whose characters needs to be splited
        
      Returns : 
        chars (str) : All the characters after splitting
      """
      return " ".join(list(string))
        
    def last_relavent(self,states,seq_lens):
        """
        Gathers last and the relavent data from the hidden states based on the
        sequence length provded
        
        Args:
            states : Hidden states
            seq_lens : Sequence length of the data
            
        Returns:
            res (tensor) : A sequence of combined tensors of new dimension
        """
        seq_lens = seq_lens.long().detach().cpu().numpy() - 1
        out = []
        for batch_index, column_index in enumerate(seq_lens):
            out.append(states[batch_index, column_index])
        return torch.stack(out)


    def nltk_preprocessor(self, sentence,stopwords=1):
        """preprocessing the data based on nltk STOPWORDS

        Args:
            sentence (string): The string or the sentence that is to be passed 

        Returns:
            sentence (string): The pre proceesed result from the function 
        """

        sentence = sentence.lower()
        # get rid of the stop words
        pt = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        sentence = pt.sub("", sentence)
        # paranthesis cases 
        sentence = re.sub(r"\([^)]*\)", "", sentence)
        # handling the spaces and the filters
        sentence = re.sub(r"([-;;.,!?<=>])", r" \1", sentence)
        sentence = re.sub(r"[^A-Za-z0-9]", " ", sentence) # removing all cases for non alpha numeric characters 
        sentence = re.sub(" +", " ", sentence)
        sentence = sentence.strip()

        return sentence

    def load_and_prep_image(self,filename, img_shape=224, scale=True):
      """
      Reads in an image from filename, turns it into a tensor and reshapes into
      (224, 224, 3).
      Parameters
      ----------
      filename (str): string filename of target image
      img_shape (int): size to resize target image to, default 224
      scale (bool): whether to scale pixel values to range(0, 1), default True
      """
      # Read in the image
      img = tf.io.read_file(filename)
      # Decode it into a tensor
      img = tf.image.decode_jpeg(img)
      # Resize the image
      img = tf.image.resize(img, [img_shape, img_shape])
      if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
      else:
        return img

    def make_confusion_matrix(self,y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
      """Makes a labelled confusion matrix comparing predictions and ground truth labels.
      If classes is passed, confusion matrix will be labelled, if not, integer class values
      will be used.
      Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).

      Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
      Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=(15, 15),
                              text_size=10)
      """  
      # Create the confustion matrix
      cm = confusion_matrix(y_true, y_pred)
      cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
      n_classes = cm.shape[0] # find the number of classes we're dealing with

      # Plot the figure and make it pretty
      fig, ax = plt.subplots(figsize=figsize)
      cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
      fig.colorbar(cax)

      # Are there a list of classes?
      if classes:
        labels = classes
      else:
        labels = np.arange(cm.shape[0])

      # Label the axes
      ax.set(title="Confusion Matrix",
              xlabel="Predicted label",
              ylabel="True label",
              xticks=np.arange(n_classes), # create enough axis slots for each class
              yticks=np.arange(n_classes), 
              xticklabels=labels, # axes will labeled with class names (if they exist) or ints
              yticklabels=labels)

      # Make x-axis labels appear on bottom
      ax.xaxis.set_label_position("bottom")
      ax.xaxis.tick_bottom()

      # Set the threshold for different colors
      threshold = (cm.max() + cm.min()) / 2.

      # Plot the text on each cell
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
          plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)
        else:
          plt.text(j, i, f"{cm[i, j]}",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)

      # Save the figure to the current working directory
      if savefig:
        fig.savefig("confusion_matrix.png")

    def dump_tf_callback(self, directory, task_name):
      """
        Method used to dump the tensorboard callback for each tensorflow experiments
        
        Args:
         directory (str) : The directory where the callbacks are to be dumped
         tas (str) : The task that has to be executed
      """
      lg_dir = directory + "/" + task_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      tf_callback = tf.keras.callback.tensorboard(
        lg_dir = lg_dir
      )
      print(f'Saving the tensorboard callback to log files : {lg_dir}')
      return tf_callback

    def plt_ls_curves(self, history):
      """
        Method used to plot the loss curves for training and validation metrics

        Args 
          history : History object of the tensorflow model
      """
      loss = history.history["loss"] 
      validation_loss = history.history["val_loss"]
      accuracy = history.history["accuracy"]
      validation_accuracy = history.history["val_accuracy"]
      epochs = range(len(history.history["loss"]))
      # Plot loss
      plt.plot(epochs, loss, label='training_loss')
      plt.plot(epochs, validation_loss, label='val_loss')
      plt.title('Loss')
      plt.xlabel('Epochs')
      plt.legend()

      # Plot accuracy
      plt.figure()
      plt.plot(epochs, accuracy, label='training_accuracy')
      plt.plot(epochs, validation_accuracy, label='val_accuracy')
      plt.title('Accuracy')
      plt.xlabel('Epochs')
      plt.legend();
    
    def compare_historys(self, original_history, new_history, initial_epochs=5):
      """
      Compares two TensorFlow model History objects.
      
      Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after original_history)
        initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
      """
    
      # Get original history measurements
      acc = original_history.history["accuracy"]
      loss = original_history.history["loss"]

      val_acc = original_history.history["val_accuracy"]
      val_loss = original_history.history["val_loss"]

      # Combine original history with new history
      total_acc = acc + new_history.history["accuracy"]
      total_loss = loss + new_history.history["loss"]

      total_val_acc = val_acc + new_history.history["val_accuracy"]
      total_val_loss = val_loss + new_history.history["val_loss"]

      # Make plots
      plt.figure(figsize=(8, 8))
      plt.subplot(2, 1, 1)
      plt.plot(total_acc, label='Training Accuracy')
      plt.plot(total_val_acc, label='Validation Accuracy')
      plt.plot([initial_epochs-1, initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
      plt.legend(loc='lower right')
      plt.title('Training and Validation Accuracy')

      plt.subplot(2, 1, 2)
      plt.plot(total_loss, label='Training Loss')
      plt.plot(total_val_loss, label='Validation Loss')
      plt.plot([initial_epochs-1, initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
      plt.legend(loc='upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')
      plt.show()

    def pred_and_plot(self, model, filename, class_names):
      """
      Imports an image located at filename, makes a prediction on it with
      a trained model and plots the image with the predicted class as the title.
      """
      # Import the target image and preprocess it
      img = self.load_and_prep_image(filename)

      # Make a prediction
      pred = model.predict(tf.expand_dims(img, axis=0))

      # Get the predicted class
      if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
      else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

      # Plot the image and predicted class
      plt.imshow(img)
      plt.title(f"Prediction: {pred_class}")
      plt.axis(False);


    def seq_padder(self,seq, mx_len=0):
        """
        Method to specify how much sequence padding is required

        Args:
            seq  (str) : The sequence to be paddded
            mx_len (int) :  How much max length to be considered while padding the sequences
        
        Returns:
            pd_seq [list]
        """
        mx_len = max(mx_len, max(len(s) for s in seq))
        pd_seq = np.zeros((len(seq), mx_len))
        for idx, s in enumerate(seq):
            pd_seq[idx][:len(s)] = s
        return pd_seq

    def data_splitter(self, X, y, t_sz):
        """Splits the dataset into training,testing and validation data 

        Args:
            X (int),y (int) , t_sz (int) : size of the training dataset 

        Returns : 
            x_train,y_train,x_val,y_val,x_test,y_test
        """

        x_train,x_,y_train,y_ = train_test_split(X, y, train_size = t_sz, stratify = y)
        x_val, x_test, y_val, y_test = train_test_split(x_, y_, train_size = 0.5 , stratify = y_) 
        return x_train, x_val, x_test, y_train, y_val, y_test


    
    def pre_processor(self,filename):
        """Returns a dictionary list with all the information regarding each line in the file

        Args:
            filename (string): Name of the file

        Returns:
            A dictionary containing the line number, target , content of the line and the total nos of lines 
        """
        in_lines = self.get_lines(filename)
        abs_lines = ""
        abs_samples = []
        
        for line in in_lines:
            if line.startswith("###"):
                abs_id = line 
                abs_lines = ''
            elif line.isspace():
                abs_lines_split = abs_lines.splitlines()
            
                for abs_ln_num , ab_line in enumerate(abs_lines_split):
                    line_resp = {}
                    target_split = ab_line.split("\t")
                    line_resp['target'] = target_split[0]
                    line_resp['text'] = target_split[1].lower()
                    line_resp['line_number'] = abs_ln_num
                    line_resp['total_lines'] = len(abs_lines_split) - 1
                    abs_samples.append(line_resp)
            else:
                abs_lines +=  line
        return abs_samples

    class lb_encoder(object):
        """Encodes each labels with  a tag label [basically each train labels will be encoded] and generates a json file as a result
        
        """
        def __init__(self,target_classes={}):
            """init function

            Args:
                target_classes (dict, required): _description_. Defaults to {}, this parameter will take the nos of classes 
                                                to be encoded along with the nos of items for each class types
            """
            self.target_classes = target_classes
            self.encoded_classes = {item:index for index, item in self.target_classes.items()}
            self.total_classes = list(self.target_classes.keys())
            
        
        @classmethod
        def json_loader(cls,filename):
            """Loads the file that is to be passed for label enconding into a json format

            Args:
                filename (_type_): Name of the file
                cls (_type_, required):  First args to be passed in a classmethod 
            """
            with open(filename, "r") as data:
                kwargs = json_utils.load(data=data)
            return cls(**kwargs)
        
        def lb_decoder(self, targets):
            """Decodes the labelled classes 

            Args:
                targets (dict): encoded classes to be simplified
                
            Returns:
                response (list) : A list of decoded classes for the encoded labels passed
            """
            response = []
            for key, value in  enumerate(targets):
                response.append(self.encoded_classes[value])
            return response
        
        def lb_encoder(self, targets):
            """One hot label  encoding  of the target class

            Args:
                targets (_type_): target classes to be encoded

            Returns:
                encoded_response: encoded class labels
            """
            encoded_response = np.zeros((len(targets)), dtype=int)
            for key, value in enumerate(targets):
                encoded_response[key] = self.target_classes[value]
            return encoded_response
                
        def lb_fit(self,targets):
            """Encodes all the unique classes with their respective indices

            Args:
                targets (_type_): target classes to be encoded
            """
            un_classes = np.unique(targets)
            for key, value_ in enumerate(un_classes):
                self.target_classes[value_] = key
            self.encoded_classes = {item: index for index, item in self.target_classes.items()}
            self.total_classes = list(self.target_classes.keys())
            return self  
                            
            
        def save(self, filename):
            """Generates json file with label encoding 

            Args:
                filename (str): Name of the file
            """
            with open(filename, "w") as data:
                file_data = {'target_to_respective_encoding' : self.target_classes}
                json_utils.dump(file_data, filename, indent=4, sort_keys=False)
        
        def __len__(self):
            return len(self.target_classes)
        
        def __str__(self):
            return f"<lb_encoder(nos_classes={len(self)})>"
    
#     class CustomDataset(Dataset):
#         """Generates custom tokenized preprocessed dataset
#         """
        
        
#         def __init__(self, text_seq="text_seq", l_num=56, tot_ln=6656, target=451, toknzer=object):
#             self.text_seq = text_seq
#             self.l_num = l_num
#             self.tot_ln = tot_ln
#             self.target = target
#             self.toknzer = toknzer
            
#         def collation(self, data):
#             """Preprocessing on a batch of dataset

#             Args:
#                 data (ndarray): A batch of dataset in an array format
#             """
#             # grabbing the input
#             data,txt = np.array(data),txt[0]
#             ln_nums,total_lns,target = data[:,1], data[:,2],data[:,3]
#             # one hot encoding
#             ln_nums,total_lns = tf.one_hot(ln_nums, depth=20), tf.one_hot(total_lns, depth=24)
#             # tokenizing the inputs
#             toknzed_res = self.toknzer(txt.tolist(), return_tensors='pt', max_length=128, padding='max_length', truncated=True)
#             ln_nums = torch.tensor(ln_nums.numpy())
#             total_lns = torch.tensor(total_lns.numpy())
#             target = torch.LongTensor(target.astype(np.int32))
            
#             return toknzed_res,ln_nums,total_lns, target
        
#         def create_datald(self, batch_size, shuffle=False,drop_last=False):
#             dloader = DataLoader(dataset=self, batch_size=batch_size, collate_fn=self.collation, shuffle=shuffle, drop_last=drop_last, pin_memory=True)
#             return dloader
        
#         def __len__(self):
#             return len(self.target)
        
#         def __str__(self) -> str:
#             return f"<Custom_DataSet(N={len(self.target)})>"
        
#         def __getitem__(self, pos):
#             return [self.text_seq[pos], self.line_nos[pos],self.tot_ln[pos],self.target[pos]]

    class ct_tokenzr(object):
        """Generates custom tokens from data sets
        """
        def __init__(self, ch_lvl=True, nos_tkns=0, pad_tkn="<PAD>", oov_tkn="<UNK>", tkn_to_idx=None):
            """Initialize tokenizer

            Args:
                ch_lvl (boolean, optional): Enable character level tokenization or not. 
                nos_tkns (int, optional): Number of tokens . Defaults to None.
                pad_tkn (str, optional): Custom padding for the tokens . Defaults to "<PAD>".
                oov_tkn (str, optional): Overriding the value of tokens . Defaults to "<UNK>".
                tkn_to_idx (int, optional): Number of tokens to be converted to indexes. Defaults to None.
            """
            self.ch_lvl = ch_lvl
            self.sep = "" if self.ch_lvl else " "
            if nos_tkns: nos_tkns -= 2
            self.nos_tkns = nos_tkns
            self.pad_tkn = pad_tkn
            self.oov_tkn = oov_tkn
            if not tkn_to_idx: tkn_to_idx = {pad_tkn: 0, oov_tkn: 1}
            self.tkn_to_idx = tkn_to_idx
            self.idx_to_tkn = {v: k for k,v, in self.tkn_to_idx.items()}
        
        
            
        @classmethod
        def load(cls, filename):
            """Loads the tokens from the given file

            Args:
                filename (str): Name of the file to load

            Returns:
                Keyworded_dictionary (dict) 
            """
            with open(filename, "r") as f:
                kwargs = json.load(filename=filename)
            return cls(**kwargs)

        def save(self, filename):
            """
            Saves the tokenzied contents into a json dump

            Args:
                filename (str): Name of the file to load
            
            Returns:
                returns the json dumps
            """                       

            with open(filename, "w")  as f:
                data = {
                    "ch_level" : self.ch_lvl,
                    "oov_token" : self.oov_tkn,
                    "tkn_to_index" : self.tkn_to_idx
                }
                json.dumps(data, filename, indent=4, sort_keys=False)
        
        def txt_fitter(self, txt):
            """
            Fits the tokens based on the txt being passed

            Args:
                txt (str) : The actual text being passed
            
            """
            if not self.ch_lvl:
                txt = [t.split(" ") for t in txt]
            all_tkns = [tkn for t in txt for tkn in t]
            cnts = Counter(all_tkns).most_common(self.nos_tkns)
            self.min_tkn_frq = cnts[-1][1]
            for tkn, cn in cnts:
                idx = len(self)
                self.tkn_to_idx[tkn] = idx
                self.idx_to_tkn[idx] =  tkn
            return self

        def seq_txt(self, seq):
            """converts the token sequences to texts
            Args:
                seq (str) : The textual sequences
            
            Returns: 
                Text (list)
            """
            txt = []
            for se in seq:
                t = []
                for idx in se:
                    t.append(self.idx_to_tkn.get(idx, self.oov_tkn))
                txt.append(self.sep.join([tk for tk in t]))
            return txt
        
        def txt_seq(self, txt):
            """Converts the texts to token sequences
            
            Args:
                txt (str) : The textual sequences
            
            Returns:
            token_seq (list)
            """
            token_seq = []
            for t in txt:
                if not self.ch_lvl:
                    t = t.split(" ")
                seq = []
                for tkn in t:
                    seq.append(self.tkn_to_idx.get(
                        tkn, self.tkn_to_idx[self.oov_tkn]
                    ))
                token_seq.append(np.asarray(seq))
            return token_seq

        
        def __len__(self):
            return len(self.tkn_to_idx)

        def __str__(self) -> str:
            return f"<ct_tokenzr(nos_tokens={len(self)})>"

            
            
            
            
            
        
        
           
        
        
