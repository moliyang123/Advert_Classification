from torch.utils.data import Subset, DataLoader
import torch
from sklearn.model_selection import KFold
import BertClassifier
import VitClassifier
import VitSClassifier
import final
import functions
from sklearn.metrics import classification_report
import json
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import os
import Dataset
import subtrain
import torch.optim.lr_scheduler as lr_scheduler

"""
main training
@Author Tianlin Yang
"""

# The folder 0-10 saving pics, num is loop range
num = 11;
path = "../image/"
# loading label
labelpath = "Topics.json"
sentimentspath = "Sentiments.json"

labelJson = None
sentimentsJson = None

#label and sentiment json
with open(labelpath, 'r') as file:
    labelJson = json.load(file)
with open(sentimentspath, 'r') as file:
    sentimentsJson = json.load(file)

# loading dataset
images = []
labels = []
texts = []
sentiments = []
missing = []

#combine all parts of image folder 0-10
for i in range (num):
    if(i == 0 or i == 1 or i == 2 or i == 3 or i == 10 ):
        n = str(i)
        pathImage = path+n+"/"
        pathText = path+"advimage/"+n+"/"
        filenames = functions.getFileNames(path)
        imageFilenames = functions.getFileNames(pathImage)
        for f in imageFilenames:
            judge1 = functions.markImageLabels(f, n, labels, labelJson, missing, 0, 39)
            judge2 = functions.markImageLabels(f, n, sentiments, sentimentsJson, missing, 1, 31)
            if(judge1 != 1 and judge2 != 1): # some image may not have sentiment label
                functions.readFile(f, pathImage, images,"image")
                functions.readFile(f, pathText, texts,"txt")
df = pd.DataFrame({'texts': texts,'images': images, 'labels': labels, 'sentiments': sentiments})

print(df)
# df['label_int'] = df['labels'].astype(int)
# Caculate label distribution
# label_counts = df['label_int'].value_counts()
# print(label_counts.sort_index())

np.random.seed(2023)

# split dataset.
df_trainAndVal, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])

df_trainAndVal,  df_test= df_trainAndVal.reset_index(drop=True), df_test.reset_index(drop=True)

# begin training
modelImage = VitClassifier.vitClassifier()
modelText = BertClassifier.BertClassifier()
modelSemantic = VitSClassifier.vitSClassifier()

model = final.finalLayer(2304,1152,768)

def train(model, modelImage, modelText, modelSemantic, train_data, test_data, learning_rate, epochs):
    train = Dataset.Dataset(train_data)
    test = Dataset.Dataset(test_data)
    test_dataloader = DataLoader(test, batch_size=1, shuffle=False) #don't shuffle because we need create csv file
    
    # whether can use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = functions.CosineLoss()

    # optimizer for every model
    optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizerImage = Adam(modelImage.parameters(), lr=learning_rate)
    optimizerText = Adam(modelText.parameters(), lr=learning_rate)
    optimizerSemantic = Adam(modelSemantic.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        modelImage = modelImage.cuda()
        modelText = modelText.cuda()
        modelSemantic = modelSemantic.cuda()
        criterion = criterion.cuda()

    # freeze 0-7 layer for all transformers
    modelImage.freezeLayer([0,1,2,3,4,5,6,7])
    modelText.freezeLayer([0,1,2,3,4,5,6,7])
    modelSemantic.freezeLayer([0,1,2,3,4,5,6,7])

    #use dict to save best loss
    best_loss_dict = {}
    best_loss_dict["best_val_loss_image"] = float('inf')
    best_loss_dict["best_val_loss_text"] = float('inf')
    best_loss_dict["best_val_loss_semantic"] = float('inf')

    if os.path.exists("train_log.txt"):
        os.remove("train_log.txt")
    if os.path.exists("main_train_log.txt"):
        os.remove("main_train_log.txt")
    
    # do sub module training
    for epoch_num in range(epochs):
        subtrain.subtrain(train, "semantic", modelSemantic, criterion, optimizerSemantic, device, test_dataloader, best_loss_dict, epoch_num)
        subtrain.subtrain(train, "text", modelText, criterion, optimizerText, device, test_dataloader, best_loss_dict, epoch_num)
        subtrain.subtrain(train, "image", modelImage,criterion, optimizerImage, device, test_dataloader, best_loss_dict, epoch_num)
        
    # fusion model training
    modelText.load_state_dict(torch.load('trained_text.pth'))
    modelImage.load_state_dict(torch.load('trained_image.pth'))
    modelSemantic.load_state_dict(torch.load('trained_semantic.pth'))

    # set best val loss
    best_val_loss = float('inf')
    
    for epoch_num in range(epochs):
        print("Epoch:" + str(epoch_num + 1))
        with open("main_train_log.txt", "a") as file:
            file.write("Epoch:" + str(epoch_num + 1))
            file.write("\n")

        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True)

        train_dataloaders = []
        val_dataloaders = []

        # split for Cross-validation
        for train_indices, val_indices in kf.split(train):
   
            train_samples = Subset(train, train_indices)
            val_samples = Subset(train, val_indices)

            train_dataloader = DataLoader(train_samples, batch_size=1, shuffle=True)
            val_dataloader = DataLoader(val_samples, batch_size=1, shuffle=True)

            train_dataloaders.append(train_dataloader)
            val_dataloaders.append(val_dataloader)

        # do training with Cross-validation
        for fold in range(k_folds):
            train_dataloader = train_dataloaders[fold]
            val_dataloader = val_dataloaders[fold]

            total_loss_train = 0
            total_accuracy_train_topic = 0
            total_accuracy_train_sentiments = 0
            train_pred = []
            train_true = []
            
            for train_text, train_image, train_label, train_sentiments, train_label_num, train_sentiment_num, imagename in tqdm(train_dataloader):
                combined_output = None
                with torch.no_grad():
                    train_label = torch.stack(train_label).squeeze(1).to(device)
                    image = train_image.to(device)
                    mask = train_text['attention_mask'].to(device)
                    input_id = train_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size

                    __, __, output1 = modelImage(image, device)
                    __, __, output2 = modelText(input_id, mask)
                    __, __, output3 = modelSemantic(image, device)

                    #combine all feature to 1 tensor
                    combined_output = torch.cat((output1, output2, output3), dim=1)
                    combined_output = combined_output.to(device)

                pooled_output = model(combined_output)

                batch_loss1 = criterion(pooled_output.squeeze(), train_label)
                batch_loss2 = criterion(pooled_output.squeeze(), train_sentiments)
                total_loss_train += batch_loss1.item() + batch_loss2.item()

                pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(train_sentiments))

                accuracy1 = (pred1.item() == train_label_num).sum().item()
                accuracy2 = functions.compute_overlap_rate(pred2, train_sentiment_num)
                total_accuracy_train_topic += accuracy1
                total_accuracy_train_sentiments += accuracy2

                train_pred.append(pred1.item())
                train_true.append(train_label_num.item())

                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_train_topic.csv",imagename, pred1.item(), train_label_num.item(), pooled_output.squeeze().tolist())
                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_train_sentiment.csv",imagename, pred2.tolist(), train_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

                model.zero_grad()
                batch_loss1.backward(retain_graph=True)
                batch_loss2.backward()
                optimizer.step()

            # Validating the model
            total_loss_val = 0
            total_accuracy_val_topic = 0
            total_accuracy_val_sentiments = 0
            val_pred = []
            val_true = []

            with torch.no_grad():
                for val_text, val_image, val_label, val_sentiments, val_label_num, val_sentiment_num, imagename in tqdm(val_dataloader):
                    combined_output = None
                    val_label = torch.stack(val_label).squeeze(1).to(device)
                    image = val_image.to(device)
                    mask = val_text['attention_mask'].to(device)
                    input_id = val_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size

                    __, __, output1 = modelImage(image, device)
                    __, __, output2 = modelText(input_id, mask)
                    __, __, output3 = modelSemantic(image, device)
                    
                    combined_output = torch.cat((output1, output2, output3), dim=1)
                    combined_output = combined_output.to(device)

                    pooled_output = model(combined_output)

                    batch_loss1 = criterion(pooled_output.squeeze(), val_label)
                    batch_loss2 = criterion(pooled_output.squeeze(), val_sentiments)
                    total_loss_val += batch_loss1.item() + batch_loss2.item()
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(val_sentiments))
                    
                    accuracy1 = (pred1.item() == val_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, val_sentiment_num)
                    total_accuracy_val_topic += accuracy1
                    total_accuracy_val_sentiments += accuracy2

                    val_pred.append(pred1.item())
                    val_true.append(val_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_val_topic.csv",imagename, pred1.item(), val_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_val_sentiment.csv",imagename, pred2.tolist(), val_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

            # testing the model
            total_accuracy_test_topic = 0
            total_accuracy_test_sentiments = 0
            test_pred = []
            test_true = []
            
            with torch.no_grad():
                for test_text, test_image, test_label, test_sentiments, test_label_num, test_sentiment_num, imagename in tqdm(test_dataloader):
                    combined_output = None
                    test_label = torch.stack(test_label).squeeze(1).to(device)
                    image = test_image.to(device)
                    mask = test_text['attention_mask'].to(device)
                    input_id = test_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size

                    __, __, output1 = modelImage(image, device)
                    __, __, output2 = modelText(input_id, mask)
                    __, __, output3 = modelSemantic(image, device)
                    
                    combined_output = torch.cat((output1, output2, output3), dim=1)
                    combined_output = combined_output.to(device)

                    pooled_output = model(combined_output)

                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(test_sentiments))
                    
                    accuracy1 = (pred1.item() == test_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, test_sentiment_num)
                    total_accuracy_test_topic += accuracy1
                    total_accuracy_test_sentiments += accuracy2

                    test_pred.append(pred1.item())
                    test_true.append(test_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_test_topic.csv",imagename, pred1.item(), test_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_main_test_sentiment.csv",imagename, pred2.tolist(), test_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

            # if total_loss_val < best_val_loss, save model.
            if(total_loss_val < best_val_loss):
                torch.save(model.state_dict(), 'trained_main.pth')
                print("Main model saved! Now loss is:", end="")
                print()
                best_val_loss = total_loss_val

            train_f1 = classification_report(train_true, train_pred, digits=3)
            val_f1 = classification_report(val_true, val_pred, digits=3)
            test_f1 = classification_report(test_true, test_pred, digits=3)

            #Attention: The sentiment accuracy here is not standard way. Please calculate through csv files.
            
            main_train_log = f'''Folds: {fold + 1}
            Train Loss: {total_loss_train / len(train_dataloader): .3f} 
            Train Topic Accuracy: {total_accuracy_train_topic / len(train_dataloader): .3f} 
            Train Sentiments Accuracy: {total_accuracy_train_sentiments / len(train_dataloader): .3f}
            Val Loss: {total_loss_val / len(val_dataloader): .3f} 
            Val Topic Accuracy: {total_accuracy_val_topic / len(val_dataloader): .3f}
            Val Sentiments Accuracy: {total_accuracy_val_sentiments / len(val_dataloader): .3f}
            Test Topic Accuracy: {total_accuracy_test_topic / len(test_dataloader): .3f}
            Test Sentiments Accuracy: {total_accuracy_test_sentiments / len(test_dataloader): .3f}
            Current LR: {optimizer.param_groups[0]['lr']}
            '''
            print(main_train_log) 
            file_path = "main_train_log.txt"
            with open(file_path, "a") as file:
                file.write(main_train_log)
                file.write("\n")
                file.write(train_f1)
                file.write(val_f1)
                file.write(test_f1)
                file.write("\n")

train(model, modelImage, modelText, modelSemantic, df_trainAndVal, df_test, 1e-5, 2)