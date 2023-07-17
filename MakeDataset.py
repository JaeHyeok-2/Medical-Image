from dataset import * 
from PIL import Image
import os 
from torch.utils.data import ConcatDataset
import torch.distributed as dist 

os.chdir('..')


positive_train_file_path = "ImageTxtFile/positive_train.txt"
negative_train_file_path = "ImageTxtFile/negative_train.txt"

pos_train_data_path = []
neg_train_data_path = []


positive_test_file_path = "ImageTxtFile/positive_test.txt"
negative_test_file_path = "ImageTxtFile/negative_test.txt"

pos_test_data_path = []
neg_test_data_path = []



with open(positive_train_file_path, 'r') as f :
    for line in f : 
        line = line.strip()
        pos_train_data_path.append(line)


with open(negative_train_file_path, 'r') as f : 
    for line in f:
        line = line.strip()
        neg_train_data_path.append(line) 


with open(positive_test_file_path, 'r') as f :
    for line in f : 
        line = line.strip()
        pos_test_data_path.append(line)


with open(negative_test_file_path, 'r') as f : 
    for line in f:
        line = line.strip()
        neg_test_data_path.append(line) 

transform = ImageTransform() 



PosTrainDataset = TumorDataset(pos_train_data_path, 1,transform) 
NegTrainDataset = TumorDataset(neg_train_data_path, 0,transform)

concat_Dataset = ConcatDataset([PosTrainDataset, NegTrainDataset])

PosTestDataset = TumorDataset(pos_test_data_path, 1,transform) 
NegTestDataset = TumorDataset(neg_test_data_path, 0,transform)

concat_Test_Dataset = ConcatDataset([PosTestDataset, NegTestDataset])

shuffle = False 
pin_memory = True # Memory에 Data를 바로 올릴수있도록 하는 옵션 
