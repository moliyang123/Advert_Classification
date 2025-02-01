from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification
import functions

"""
The vitSClassifier.
@Author Tianlin Yang
@Reference:https://huggingface.co/google/vit-base-patch16-224
"""

class vitSClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(vitSClassifier, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(1000, 39)  # output for topics
        self.linear2 = nn.Linear(1000, 30)  # output for sentiments
        self.linear3 = nn.Linear(1000, 768)
        self.relu = nn.ReLU()

    def forward(self, tensor, device):
        inputs = self.processor(images=functions.preprocessImage(tensor), return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        # get logits classification result
        logits = outputs.logits
        # dropout some nerual incase over-fitting, use for CrossEntropy loss
        dropout_output = self.dropout(logits)
        output1 = self.linear1(dropout_output)
        output2 = self.linear2(dropout_output)
        output3 = self.linear3(logits)
        final_layer1 = self.relu(output1)
        final_layer2 = self.relu(output2)
        return final_layer1, final_layer2, output3
    
    #freeze some not important layer to let model faster
    def freezeLayer(self, freeze_list):
        p = 0
        all_layers = self.model.vit.encoder.layer
        for layer in all_layers:
            if p in freeze_list:
                for param in layer.parameters():
                    param.requires_grad = False
            p += 1