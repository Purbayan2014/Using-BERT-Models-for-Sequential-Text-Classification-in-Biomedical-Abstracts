import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast, AdamW,BertModel,BertTokenizerFast


class CustomBertModel(nn.Module):
	def __init__(self, classes, freeze=True):
		super(CustomBertModel, self).__init__()
		self.model = BertModel.from_pretrained('bert-base-cased')
		if freeze :
			for param in self.model.parameters():
				param.requires_grad = False
		# Forward layers
		self.forward_text = nn.Linear(768, 512)
		self.forward_line_nos = nn.Linear(20, 64)
		self.total_ln = nn.Linear(24,64)
		self.forward_final = nn.Linear((64+64+512), 128)
		self.classification = nn.Linear(128, classes)
		self.dropout = nn.Dropout(0.3)

	def forward(self,inputs, line_nos, total_lns):
		model_output = self.model(**inputs)
		layer1 = model_output['pooler_output']

		# forward layers output
		layer1 = F.relu(self.forward_text(layer1))
		layer2 = F.relu(self.forward_line_nos(line_nos))
		layer3 = F.relu(self.total_ln(total_lns))
		x = torch.cat((layer1,layer2,layer3), dim=1)
		x = self.dropout(x)
		x = F.relu(self.forward_final(x))
		x = self.dropout(x)
		x = self.classification(x)
		return x