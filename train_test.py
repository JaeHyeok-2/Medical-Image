import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torchvision.models as models 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader 
import torchvision 
import wandb 
from tqdm import tqdm 
import time 
import os 

def train(model, dataloader, criterion, optimizer, epochs, save_path, device, weights ):
    wandb.watch(model, criterion,log="all", log_freq=10)
    BEST_LOSS = 1e11
    model.to(device)
    model.train()
    print(device)
    criterion.to(device)

    save_path = os.path.join(save_path, f'{weights[0]}_{weights[1]}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for epoch in tqdm(range(epochs)): 
        start_time = time.time() 

        running_loss = 0.0
        for i, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad() 
            outputs = model(inputs)
            loss= criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels]) 
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            print(f"{i}th Iteration Loss : {loss.item()} ")
        running_loss /= float(i)

        if BEST_LOSS > running_loss:
            BEST_LOSS = running_loss 
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_path,f'epoch_{epoch}_all.tar'))
        
        wandb.log({"Training LOSS":running_loss})
        end_time = time.time() 
        print(f" EPOCH : {epoch + 1} / {epochs} | Loss per Epoch : {running_loss} | Training Time per epoch : {end_time - start_time}s ",end='\n')


def eval(model, dataloader, criterion, device='cpu', image_save=False):
    model.to(device)
    model.eval() 

    min_loss = 1e9
    min_loss_image = None
    test_loss = 0
    correct = 0
    batch_count = 0
    total = 0
    
    save_path = 'evaluation_image/1/'
    if image_save :
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        import torchvision 
        import numpy as np 
        from PIL import Image 


        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device) 
            
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            total += labels.size()[0]
            test_loss += loss.item() 
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()

            print(f" {i}th Iteration Loss : {loss.item()}")       
            



    print("Test Loss: {:.3f} | Test Acc : {:.3f}".format(test_loss / i, 100. * correct/total), f"| Total Image, Correct : {total}, {correct}")


