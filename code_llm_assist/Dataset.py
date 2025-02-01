from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
import functions
import torch
from transformers import BertTokenizer
import os
import re

"""
The dataset class.
@author Tianlin Yang
"""

class Dataset():
    def __init__(self, dataframe):
        self.labels = dataframe['labels']
        self.llm_texts = dataframe['llm_texts']
        self.texts = dataframe['texts']
        self.sentiments = dataframe['sentiments']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.labels)
    
    def get_imagename(self, i):
        filename = os.path.basename(self.texts[i][0:-4])
        return filename

    def get_batch_labels(self, i):
        # Fetch a batch of topics
        train_label_path = functions.readLabelPklPath(self.labels[i])
        return functions.readPkl(train_label_path), self.labels[i]
    
    def get_batch_sentiments(self, i):
        # Fetch a batch of sentiments
        s_list = []
        for n in range (len(self.sentiments[i])):
            train_sentiment_path = functions.readSentimentPklPath(self.sentiments[i][n])
            sentiment = functions.readPkl(train_sentiment_path)
            s_list.append(sentiment)
        
        return s_list, self.sentiments[i]
    
    def get_batch_texts(self, i):
        text = functions.readText(self.texts[i])
        tokenText = self.tokenizer(text, padding='max_length',max_length=512, truncation=True, return_tensors="pt")
        return tokenText

    def get_batch_llm_texts(self, i):
        result = functions.read_llm_texts(self.llm_texts[i])
        tensor = []
        for i in result:
            tokenText = self.tokenizer(i, padding='max_length',max_length=512, truncation=True, return_tensors="pt")
            tensor.append(tokenText)
        return tensor

    def __getitem__(self, i):
        batch_texts = self.get_batch_texts(i)
        batch_llm_texts = self.get_batch_llm_texts(i)
        batch_labels, label_num = self.get_batch_labels(i)
        batch_sentiments, sentiment_num = self.get_batch_sentiments(i)
        imagename = self.get_imagename(i)
        return batch_texts, batch_llm_texts, batch_labels, batch_sentiments, label_num, sentiment_num, imagename