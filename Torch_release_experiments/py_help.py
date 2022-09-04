import re
import numpy as np
import json as json_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch

class torch_helper:
    """Common utility class for torch experiments
    """
    
    def __init__(self):
        self.lb_encoder = self.lb_encoder()
#         self.CustomDataset = self.CustomDataset(text_seq="text_seq", l_num=56, tot_ln=6656, target=451, toknzer=object)

    def get_lines(self,filename):
        """Returns the file contents

        Args:
            filename (string): Name of the file
        """
        with open(filename, mode="r") as data:
            return data.readlines()
        
    def last_relavent(self,states,seq_lens):
        """
        Gathers last and the relavent data
        
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
            self.encoded_classes = {item:index for index, item in enumerate(self.target_classes)}
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
                response.append(self.encoded_classes[key])
            return response
        
        def lb_encoder(self, targets):
            """One hot label  encoding  of the target class

            Args:
                targets (_type_): target classes to be encoded

            Returns:
                encoded_response: encoded class labels
            """
            encoded_response = np.zeros((len(targets)), dtype=np.int64)
            for key, value in enumerate(targets):
                encoded_response[key] = self.encoded_classes[value]
            return encoded_response
                
        def lb_fit(self,targets):
            """Encodes all the unique classes with their respective indices

            Args:
                targets (_type_): target classes to be encoded
            """
            un_classes = np.unique(targets)
            for key, value_ in enumerate(un_classes):
                self.target_classes[value_] = key
            self.encoded_classes = {item:index for index, item in enumerate(self.target_classes)}
            self.total_classes = list(self.target_classes.keys())
            return self  
                            
            
        def save(self, filename):
            """Generates json file with label encoding 

            Args:
                filename (str): Name of the file
            """
            with open(filename, "w") as data:
                file_data = {'target_to_respective_encoding' : self.encoded_classes}
                json_utils.dump(file_data, filename, indent=4, sort_keys=False)
        
        def __length__(self):
            return len(self.target_classes)
        
        def __str__(self):
            return f"<lb_encoder={len(self)}>"
    
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
            
            
            
            
            
        
        
           
        
        
