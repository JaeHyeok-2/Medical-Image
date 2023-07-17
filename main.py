import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torchvision.models as models 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader 
import torchvision 
import argparse
import os
# Making Datasets in different Folder 
from MakeDataset import * 

from tqdm import tqdm 
import time  
import wandb 

from classification_model import make_ResNet
from image_make_utils import make_HeatMap
from train_test import train,eval

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural Network")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning  rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for trainging')
    parser.add_argument('--labels', type=int, default=2,
                        help="ResNet18 number of Class")
    parser.add_argument("--batch_size", type=int, default=32,
                        help='Model Batch_size')
    parser.add_argument('--model_save_path', type=str, default='Model_save/2023_07_13_1,1',
                        help="Where to Save model ")
    parser.add_argument('--pretrained', type=bool,  default=False,
                        help="Using Pretrained or not")
    parser.add_argument('--loss_image_save', type=bool, default =False, 
                        help="Sampling Image getting High Loss ")
    args = parser.parse_args()

    return args




if __name__ == '__main__':


    args = get_args()

    wandb.init(
    project="2023_06_25_Train_Except LUAC Dataset_",

    config= {
        "learning_rate": args.learning_rate,
        "Architecture" : "Custom ResNet-18",
        "dataset" : "Tumor",
        "epochs" : args.epochs,
        }
    )

    # Define model, criterion, optimizer, batch_size, epoch 
 

    gpu_count = torch.cuda.device_count() 

    # dist.init_process_group(backend='nccl')


    model = make_ResNet(args)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("**************** Model Architecture **********************************\n")
    print(model)
    print("********************************************************************\n\n")

    model.to(device)

    train_Dataloader = DataLoader(concat_Dataset, batch_size = args.batch_size, shuffle =shuffle, 
                            pin_memory = pin_memory)


    
    criterion_weight = [[1, 1]]

    for weights in criterion_weight: 
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path, exist_ok=True)    
        pos, neg = weights 
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model = nn.DataParallel(model)

        train(model, train_Dataloader, criterion, optimizer, args.epochs, args.model_save_path, device,weights) 

        print("Complete !!")


    if args.pretrained : 
        checkpoint = torch.load('/home/lab/halyn/Model_save/2023_06_25/epoch_18_all.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    make_HeatMap(model, device)
        
