import os
from collections import Counter
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import csv
import pickle

"""
Functions can use multiple times, some are not used in the model because related function were deleted or replaced.
@Author Tianlin Yang
"""

def save_to_csv(file_path, content):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['filenname', 'Predict', 'Groundtruth', 'Correct'])
        writer.writerow(content)

def getFileNames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

def mostCommonItem(items):
    counter = Counter(items)
    most_common_item = counter.most_common(1)[0][0]
    return most_common_item

def preprocessImage(tensor):
    tensor = tensor.squeeze(0)
    # turn grey image to colorful image
    if(tensor.shape[0] == 1):
        tensor = torch.cat([tensor] * 3, dim=0)
    elif(tensor.shape[0] == 4):
        tensor = tensor[:3]
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image

#image can be png and jpg
def markImageLabels(filename, num, labels, labelJson, missing, f, unk):
    key = num+"/"+filename[0:-4]
    if (key + ".jpg") in labelJson:
        keyjpg = labelJson[key + ".jpg"]
        l = None
        if(f == 1):
            flat_list = []
            [flat_list.extend(sublist) for sublist in keyjpg] # extend all sublist to flat_list
            unique_flat_list = list(set(flat_list)) # make all item unique in list
            int_list = [convert_to_int(element, unk) - 1 for element in unique_flat_list] # make it to label
            labels.append(torch.tensor(int_list))
        else:
            l = mostCommonItem(keyjpg)
            labels.append(torch.tensor(convert_to_int(l, unk) - 1))
        return 0
        
    elif (key + ".png") in labelJson:
        keypng = labelJson[key + ".png"]
        l = None
        if(f == 1):
            flat_list = []
            [flat_list.extend(sublist) for sublist in keypng] # extend all sublist to flat_list
            unique_flat_list = list(set(flat_list))  # make all item unique in list
            int_list = [convert_to_int(element, unk) - 1 for element in unique_flat_list] # make it to label
            labels.append(torch.tensor(int_list))
        else:
            l = mostCommonItem(keypng)
            labels.append(torch.tensor(convert_to_int(l, unk) - 1))
        return 0
    else:
        missing.append(filename) # a checking method to find if here is missing file can't find in Json
        return 1

# a few original dataset is string, we don't have time to label it manually, so just set it unk.
def convert_to_int(string, unk):
    try:
        number = int(string)
        return number
    except ValueError:
        if(unk == 39):
            return unk
        elif(unk == 31):
            pass
    
def readFile(f, path, list, c):
    if(c == 'txt'):
        file_path = path + f[0:-4] + '.txt'
    elif(c == 'image'):
        file_path = path + f
    list.append(file_path)

# read text with RE
def readText(path):
    finallstr = ""
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        pattern = re.compile(r'Line:+.*}')
        matches = re.findall(pattern, content)
        for match in matches:
            pattern2 = re.compile(r"'([^']+)'")
            matches2 = re.findall(pattern2, match)
            for match2 in matches2:
                if(match2[-1] == '-'): # advert change line a lot so we delete hyphen for bert read easily.
                    finallstr = finallstr + match2[0:-1]
                else: # add space between different words when change line
                    finallstr = finallstr + match2 + " "
    return finallstr

# check sentiment accuracy
def sentimentAccuracy(predict, label, device):
    rounded_label = label.to(torch.int)
    label_num = torch.sum(rounded_label) # because rounded_label all is 1, so sum is num
    __, index = torch.topk(predict, label_num) #select top k highest label
    predict_result = torch.zeros(30).to(device)
    predict_result[index] = 1 #set top k highest label 1
    rounded_predict = predict_result.to(torch.int)
    logical_and_result = rounded_label & rounded_predict # get the overlap
    correct = torch.sum(logical_and_result) # get how many correct
    return correct/label_num

def labelNumToSentence(num):
    path = "image_result/Sentiments_List.txt"
    topic_list = {}
    with open(path, 'r',encoding="utf-8") as file:
        for line in file:
            topic_list[int(line.split('"')[0])] = line.split('"')[1]
    return topic_list[num]

# Loss function
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, input1, input2):
        if isinstance(input2, list):
            cosine_loss_all = 0
            for i in input2:
                i = torch.stack(i).squeeze(1).to("cuda")
                cosine_similarity = torch.nn.functional.cosine_similarity(input1, i, dim=0)
                cosine_loss = 1 - cosine_similarity
                cosine_loss_all += cosine_loss
            
            cosine_loss_all = cosine_loss_all/len(input2) # return average loss for all groundtruth and predict
            return cosine_loss_all
        else:
            cosine_similarity = torch.nn.functional.cosine_similarity(input1, input2, dim=0)
            cosine_loss = 1 - cosine_similarity
            return cosine_loss
    
def readLabelPklPath(num):
    label_path = "label_embedding_bert/"+str(num.item()+1)+".pkl"
    return label_path

def readSentimentPklPath(num):
    sentiment_path = "sentiment_embedding_bert/"+str(num.item()+1)+".pkl"
    return sentiment_path

# select the most similiar topic
def getSimiliarLabel(vector1, criterion, device):
    a_list = []
    for k in range(39):
        labels = "label_embedding_bert/"+str(k+1)+".pkl"
        vector2 = torch.tensor(readPkl(labels)).to(device)
        similarity = criterion(vector1, vector2)
        a_list.append(similarity)
    answer_for_this_question = torch.argmin(torch.stack(a_list))
    return answer_for_this_question

def get_top_k_min_indices(lst, k):
    min_indices = sorted(range(len(lst)), key=lambda i: lst[i])[:k]
    return min_indices

# get top k sentiment according to groundtruth's length
def getSimiliarSentiment(vector1, criterion, device, length):
    a_list = []
    for k in range(30):
        labels = "sentiment_embedding_bert/"+str(k+1)+".pkl"
        vector2 = torch.tensor(readPkl(labels)).to(device)
        similarity = criterion(vector1, vector2)
        a_list.append(similarity)
    a_list = torch.stack(a_list)
    answer_for_this_question = get_top_k_min_indices(a_list, length)
    return torch.tensor(answer_for_this_question)

def readPkl(path):
    with open(path, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list

def append_to_csv(filename, videoname, pred, label, vector):
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        if not file_exists:
            csv_writer.writerow(['Video_Name', 'Pred', 'Label', 'Vector'])
        csv_writer.writerow([videoname, pred, label, vector])

def compute_overlap_rate(tensor1, tensor2):
    intersection = len(set(tensor1.tolist()) & set(tensor2.tolist()[0])) # tensor2 shape is like [[]]
    overlap_rate = intersection / len(tensor1.tolist())
    return overlap_rate

def read_llm_texts(path):
    with open(path, 'r', encoding='utf-8') as file:
        sentence = file.read()
    keywords = ["Describe the main visual elements and messaging in this image.\nASSISTANT: ", 
                "Describe the main visual elements and messaging in this image in one line.\nASSISTANT: ", 
                "Briefly describe the key visuals and text of this image for a visually impaired listener.\nASSISTANT: ",
                "Detect branding, slogans, or call-to-action phrases that indicate this content is an advertisement.\nASSISTANT: ",
                "Is the input an advertisement\? Respond with YES/NO. Followed by what is being promoted if YES or followed by the justification of the answer if NO.\nASSISTANT: ",
                "Extract the following information:Brand Name/Product: Clearly state the brand or product being advertised.Main Message: Summarize the central message or offer of the ad.Visual Elements: Describe significant images, colors, and visual style.Textual Content: Read out any text, including headlines, slogans, and call-to-actions.Tone and Mood: Convey the overall feel of the ad, such as humorous, serious, or inspirational.\nASSISTANT: ",
                "Time: "]
    segments = re.split(f'{keywords[0]}|{keywords[1]}|{keywords[2]}|{keywords[3]}|{keywords[4]}|{keywords[5]}|{keywords[6]}',sentence)[1:-1]
    return segments