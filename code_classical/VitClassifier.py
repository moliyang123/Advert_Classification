import torch
from torch import nn
from transformers import ViTImageProcessor, ViTModel
import functions

"""
The vitClassifier.
@Author Tianlin Yang
@Reference:https://huggingface.co/facebook/dino-vitb16
"""

class vitClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(vitClassifier, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        self.model = ViTModel.from_pretrained('facebook/dino-vitb16')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 39)  # output for topics
        self.linear2 = nn.Linear(768, 30)  # output for sentiments
        self.relu = nn.ReLU()

    def forward(self, tensor, device):
        inputs = self.processor(images=functions.preprocessImage(tensor), return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        # get pooled output
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        # dropout some nerual incase over-fitting, use for CrossEntropy loss
        dropout_output = self.dropout(pooled_output)
        output1 = self.linear1(dropout_output)
        output2 = self.linear2(dropout_output)
        final_layer1 = self.relu(output1)
        final_layer2 = self.relu(output2)
        return final_layer1, final_layer2, pooled_output
    
    #freeze some not important layer to let model faster
    def freezeLayer(self, freeze_list):
        p = 0
        all_layers = self.model.encoder.layer
        for layer in all_layers:
            if p in freeze_list:
                for param in layer.parameters():
                    param.requires_grad = False
            p += 1