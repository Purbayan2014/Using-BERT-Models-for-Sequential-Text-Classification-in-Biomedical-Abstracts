import numpy as np
import json as json_utils

class torch_helper:
    """Common utility class for torch experiments
    """
    
    def __init__(self):
        self.lb_encoder = self.lb_encoder()

    def get_lines(self,filename):
        """Returns the file contents

        Args:
            filename (string): Name of the file
        """
        with open(filename, mode="r") as data:
            return data.readlines()
    
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
        
        def lb_decoder(self, targets):
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
        
        