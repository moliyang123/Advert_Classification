from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torch
import functions
from tqdm import tqdm
from sklearn.metrics import classification_report


"""
sub module training
@Author Tianlin Yang
"""

"""
@parameter train:train_data
@parameter module:type of module
@parameter model:which model
@parameter criterion: criterion function
@parameter optimizer: model's optimizer
@parameter device: cuda/cpu
@parameter test_dataloader: test_dataloader
@parameter best_loss_dict: save dict for best loss
@parameter epoch_num: epoch num
"""
def subtrain(train, module, model, criterion, optimizer,  device, test_dataloader, best_loss_dict, epoch_num):
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    train_dataloaders = []
    val_dataloaders = []

    # split for Cross-validation
    for train_indices, val_indices in kf.split(train):
        train_samples = Subset(train, train_indices)
        val_samples = Subset(train, val_indices)

        train_dataloader = DataLoader(train_samples, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_samples, batch_size=1, shuffle=False)

        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)

    # do training with Cross-validation
    for fold in range(k_folds):
        train_dataloader = train_dataloaders[fold]
        val_dataloader = val_dataloaders[fold]

        total_loss_train = 0
        total_accuracy_train_topic = 0
        total_accuracy_train_sentiments = 0
        train_pred_topic = []
        train_true_topic = []

        total_loss_val = 0
        total_accuracy_val_topic = 0
        total_accuracy_val_sentiments = 0
        val_pred_topic = []
        val_true_topic = []

        total_accuracy_test_topic = 0
        total_accuracy_test_sentiments = 0
        test_pred_topic = []
        test_true_topic = []

        #if module is image
        if(module == "image"):
            for __, train_image, train_label, train_sentiments, train_label_num, train_sentiment_num, imagename in tqdm(train_dataloader):
                train_label = torch.stack(train_label).squeeze(1).to(device)
                image = train_image.to(device)

                output_topic, __, pooled_output = model(image, device)

                batch_loss1 = criterion(pooled_output.squeeze(), train_label)
                batch_loss2 = criterion(pooled_output.squeeze(), train_sentiments)
                total_loss_train += batch_loss1.item() + batch_loss2.item()

                pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(train_sentiments))
                
                accuracy1 = (pred1.item() == train_label_num).sum().item()
                accuracy2 = functions.compute_overlap_rate(pred2, train_sentiment_num)
                total_accuracy_train_topic += accuracy1
                total_accuracy_train_sentiments += accuracy2

                train_pred_topic.append(pred1.item())
                train_true_topic.append(train_label_num.item())

                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_topic.csv",imagename, pred1.item(), train_label_num.item(), pooled_output.squeeze().tolist())
                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_sentiment.csv",imagename, pred2.tolist(), train_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

                model.zero_grad()
                batch_loss1.backward(retain_graph=True)
                batch_loss2.backward()
                optimizer.step()

            # Validating the model
            with torch.no_grad():
                for __, val_image, val_label, val_sentiments, val_label_num, val_sentiment_num, imagename in tqdm(val_dataloader):
                    val_label = torch.stack(val_label).squeeze(1).to(device)
                    image = val_image.to(device)

                    output_topic, __, pooled_output = model(image, device)

                    batch_loss1 = criterion(pooled_output.squeeze(), val_label)
                    batch_loss2 = criterion(pooled_output.squeeze(), val_sentiments)
                    total_loss_val += batch_loss1.item() + batch_loss2.item()
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(val_sentiments))
                    
                    accuracy1 = (pred1.item() == val_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, val_sentiment_num)
                    total_accuracy_val_topic += accuracy1
                    total_accuracy_val_sentiments += accuracy2

                    val_pred_topic.append(pred1.item())
                    val_true_topic.append(val_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_topic.csv",imagename, pred1.item(), val_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_sentiment.csv",imagename, pred2.tolist(), val_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

            # testing the model
            with torch.no_grad():
                for __, test_image, test_label, test_sentiments, test_label_num, test_sentiment_num, imagename in tqdm(test_dataloader):
                    test_label = torch.stack(test_label).squeeze(1).to(device)
                    image = test_image.to(device)

                    output_topic, __, pooled_output = model(image, device)
                     
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(test_sentiments))
                    
                    accuracy1 = (pred1.item() == test_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, test_sentiment_num)
                    total_accuracy_test_topic += accuracy1
                    total_accuracy_test_sentiments += accuracy2

                    test_pred_topic.append(pred1.item())
                    test_true_topic.append(test_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_topic.csv",imagename, pred1.item(), test_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_sentiment.csv",imagename, pred2.tolist(), test_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

        #if module is semantic, it still use image input
        if(module == "semantic"):
            for __, train_image, train_label, train_sentiments, train_label_num, train_sentiment_num, imagename in tqdm(train_dataloader):
                train_label = torch.stack(train_label).squeeze(1).to(device)
                image = train_image.to(device)

                output_topic, __, pooled_output = model(image, device)

                batch_loss1 = criterion(pooled_output.squeeze(), train_label)
                batch_loss2 = criterion(pooled_output.squeeze(), train_sentiments)
                total_loss_train += batch_loss1.item() + batch_loss2.item()
                pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(train_sentiments))
                
                accuracy1 = (pred1.item() == train_label_num).sum().item()
                accuracy2 = functions.compute_overlap_rate(pred2, train_sentiment_num)
                total_accuracy_train_topic += accuracy1
                total_accuracy_train_sentiments += accuracy2

                train_pred_topic.append(pred1.item())
                train_true_topic.append(train_label_num.item())

                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_topic.csv",imagename, pred1.item(), train_label_num.item(), pooled_output.squeeze().tolist())
                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_sentiment.csv",imagename, pred2.tolist(), train_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

                model.zero_grad()
                batch_loss1.backward(retain_graph=True)
                batch_loss2.backward()
                # batch_loss1.backward()
                optimizer.step()

            # Validating the model
            with torch.no_grad():
                for __, val_image, val_label, val_sentiments, val_label_num, val_sentiment_num, imagename in tqdm(val_dataloader):
                    val_label = torch.stack(val_label).squeeze(1).to(device)
                    image = val_image.to(device)

                    output_topic, __, pooled_output = model(image, device)
                     
                    batch_loss1 = criterion(pooled_output.squeeze(), val_label)
                    batch_loss2 = criterion(pooled_output.squeeze(), val_sentiments)
                    total_loss_val += batch_loss1.item() + batch_loss2.item()
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(val_sentiments))
                    
                    accuracy1 = (pred1.item() == val_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, val_sentiment_num)
                    total_accuracy_val_topic += accuracy1
                    total_accuracy_val_sentiments += accuracy2

                    val_pred_topic.append(pred1.item())
                    val_true_topic.append(val_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_topic.csv",imagename, pred1.item(), val_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_sentiment.csv",imagename, pred2.tolist(), val_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())


            # testing the model
            with torch.no_grad():
                for __, test_image, test_label, test_sentiments, test_label_num, test_sentiment_num, imagename in tqdm(test_dataloader):
                    test_label = torch.stack(test_label).squeeze(1).to(device)
                    image = test_image.to(device)

                    output_topic, __, pooled_output = model(image, device)
                    
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(test_sentiments))
                    
                    accuracy1 = (pred1.item() == test_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, test_sentiment_num)
                    total_accuracy_test_topic += accuracy1
                    total_accuracy_test_sentiments += accuracy2

                    test_pred_topic.append(pred1.item())
                    test_true_topic.append(test_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_topic.csv",imagename, pred1.item(), test_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_sentiment.csv",imagename, pred2.tolist(), test_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

        # if module is text
        if(module == "text"):
            for train_text, __, train_label, train_sentiments, train_label_num, train_sentiment_num, imagename in tqdm(train_dataloader):
                train_label = torch.stack(train_label).squeeze(1).to(device)
                mask = train_text['attention_mask'].to(device) 
                input_id = train_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size.

                output_topic, __, pooled_output = model(input_id, mask)

                batch_loss1 = criterion(pooled_output.squeeze(), train_label)
                batch_loss2 = criterion(pooled_output.squeeze(), train_sentiments)
                total_loss_train += batch_loss1.item() + batch_loss2.item()
                pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(train_sentiments))
                
                accuracy1 = (pred1.item() == train_label_num).sum().item()
                accuracy2 = functions.compute_overlap_rate(pred2, train_sentiment_num)
                total_accuracy_train_topic += accuracy1
                total_accuracy_train_sentiments += accuracy2

                train_pred_topic.append(pred1.item())
                train_true_topic.append(train_label_num.item())

                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_topic.csv",imagename, pred1.item(), train_label_num.item(), pooled_output.squeeze().tolist())
                functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_train_sentiment.csv",imagename, pred2.tolist(), train_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

                model.zero_grad()
                batch_loss1.backward(retain_graph=True)
                batch_loss2.backward()
                optimizer.step()

            # Validating the model
            with torch.no_grad():
                for val_text, __, val_label, val_sentiments, val_label_num, val_sentiment_num, imagename in tqdm(val_dataloader):
                    val_label = torch.stack(val_label).squeeze(1).to(device)
                    mask = val_text['attention_mask'].to(device)
                    input_id = val_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size.

                    output_topic, __, pooled_output = model(input_id, mask)

                    batch_loss1 = criterion(pooled_output.squeeze(), val_label)
                    batch_loss2 = criterion(pooled_output.squeeze(), val_sentiments)
                    total_loss_val += batch_loss1.item() + batch_loss2.item()
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(val_sentiments))
                    
                    accuracy1 = (pred1.item() == val_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, val_sentiment_num)
                    total_accuracy_val_topic += accuracy1
                    total_accuracy_val_sentiments += accuracy2

                    val_pred_topic.append(pred1.item())
                    val_true_topic.append(val_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_topic.csv",imagename, pred1.item(), val_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_val_sentiment.csv",imagename, pred2.tolist(), val_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())
                    
            # testing the model
            with torch.no_grad():
                for test_text, __, test_label, test_sentiments, test_label_num, test_sentiment_num, imagename in tqdm(test_dataloader):
                    test_label = torch.stack(test_label).squeeze(1).to(device)
                    mask = test_text['attention_mask'].to(device)
                    input_id = test_text['input_ids'].squeeze(1).to(device) #squeeze(1) for correspond input, 0 is batch size.

                    output_topic, __, pooled_output = model(input_id, mask)
                    
                    pred1 = functions.getSimiliarLabel(pooled_output.squeeze(), criterion, device)
                    pred2 = functions.getSimiliarSentiment(pooled_output.squeeze(), criterion, device, len(test_sentiments))
                    
                    accuracy1 = (pred1.item() == test_label_num).sum().item()
                    accuracy2 = functions.compute_overlap_rate(pred2, test_sentiment_num)
                    total_accuracy_test_topic += accuracy1
                    total_accuracy_test_sentiments += accuracy2

                    test_pred_topic.append(pred1.item())
                    test_true_topic.append(test_label_num.item())

                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_topic.csv",imagename, pred1.item(), test_label_num.item(), pooled_output.squeeze().tolist())
                    functions.append_to_csv("Epoch_"+str(epoch_num)+"_Fold_"+str(fold)+"_"+module+"_test_sentiment.csv",imagename, pred2.tolist(), test_sentiment_num.tolist()[0], pooled_output.squeeze().tolist())

        if(total_loss_val < best_loss_dict[f"best_val_loss_{module}"]):
            torch.save(model.state_dict(), f'trained_{module}.pth')
            print(f"{module} model saved! Now Loss is:", end="")
            print(total_loss_val)
            best_loss_dict[f"best_val_loss_{module}"] = total_loss_val

        train_f1 = classification_report(train_true_topic, train_pred_topic, digits=3)
        val_f1 = classification_report(val_true_topic, val_pred_topic, digits=3)
        test_f1 = classification_report(test_true_topic, test_pred_topic, digits=3)
        
        #Attention: The sentiment accuracy here is not standard way. Please calculate through csv files.

        train_log = f'''Folds: {fold + 1}, module: {module}
        Train Topic Loss: {total_loss_train / len(train_dataloader): .3f} 
        Train Topic Accuracy: {total_accuracy_train_topic / len(train_dataloader): .3f} 
        Train Sentiments Accuracy: {total_accuracy_train_sentiments / len(train_dataloader): .3f}
        Val Topic Loss: {total_loss_val / len(val_dataloader): .3f} 
        Val Topic Accuracy: {total_accuracy_val_topic / len(val_dataloader): .3f}
        Val Sentiments Accuracy: {total_accuracy_val_sentiments / len(val_dataloader): .3f}
        Test Topic Accuracy: {total_accuracy_test_topic / len(test_dataloader): .3f}
        Test Sentiments Accuracy: {total_accuracy_test_sentiments / len(test_dataloader): .3f}
        '''

        print(train_log) 
        file_path = "train_log_"+module+".txt"  # save result as txt
        with open(file_path, "a") as file:
            file.write(train_log)
            file.write("\n")
            file.write(train_f1)
            file.write(val_f1)
            file.write(test_f1)
            file.write("\n")