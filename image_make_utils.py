from PIL import Image 
import matplotlib.pyplot as plt 
import os 
import torch 
import torchvision 
import torch.nn as nn
from PIL import Image, ImageDraw
import re 
import torchvision.transforms as transforms 
import json 
from tqdm import tqdm 



def find_CLAM_JPEG(folder_name, tumor_name) :
    matching_name_length = len(tumor_name)
    
    CLAM_PATH = os.path.join("CLAM_DATASET", folder_name) 
    
    for image_name in os.listdir(CLAM_PATH): 
        if tumor_name == image_name[:matching_name_length]:
            return Image.open(os.path.join(CLAM_PATH, image_name)) 
     


def make_HeatMap(model, device):
    model.eval()
    transform = transforms.ToTensor()
    sigmoid = nn.Sigmoid() 

    TEST_PATH = []
    with open('/home/lab/halyn/tumorAnnotation.json', 'r') as f:
        json_data = json.load(f) 
    

    for k,v in json_data.items():
        if k == 'POSITIVE_TEST' or k == 'NEGATIVE_TEST':
            TEST_PATH.extend(v)
    
    for path in tqdm(TEST_PATH):
        folder_name, tumor_name = path.split('/')[1], path.split('/')[2]
        
        CLAM_RESULT_JPEG = find_CLAM_JPEG(folder_name, tumor_name) 
        

        white_Image = Image.new('RGB', (CLAM_RESULT_JPEG.width, CLAM_RESULT_JPEG.height), (255, 255, 255))
        draw = ImageDraw.Draw(white_Image) 

        with torch.no_grad():
            for patch in os.listdir(path): 
                pattern = r"\#(\d+)_(\d+)\.png"
                matches = re.findall(pattern, patch) 
                
                width, height = matches[0]
                width, height = int(width), int(height) 

                patch = Image.open(os.path.join(path, patch))
                image = transform(patch).to(device).unsqueeze(dim = 0)
                
                output = model(image) 
                probs = sigmoid(output)[0][0].item() 

                print(f"Width : {width} | Height : {height}      | Prob : {probs}" )
                red, blue = int(probs * 255), int((1-probs) * 255) 

                draw.rectangle((width, height, width+2, height+2), fill =(red, 0 ,blue))
        
        white_Image.save(f"SigmoidAttention/weight_1_1/{tumor_name}.jpg")
                
            

