""" Loading Library"""
import warnings 

import os 
import sys 
from PIL import Image 
import cv2 
import numpy as np
from tqdm import tqdm 
import time 
import openslide 

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
os.chdir('..')
# LUAC : 14 | SSSF : 6 | TCGA : 10 | YS : 11
# 해당 내용 SSSF의 경우 미리 뽑아놨기 때문에.
TUMOR_DATASET_TYPE = ['LUAC', 'TCGA', 'YS', 'SSSF']

def make_folders(): 
    if not os.path.exists('PATCH'):
        os.makedirs('PATCH',exist_ok=True) 


def get_Dataset_Folder_List(): 

    SVS_PATH = [os.path.join('DATASET',name) for name in os.listdir('DATASET') if name in TUMOR_DATASET_TYPE]
    JPEG_PATH = [os.path.join('CLAM_DATASET',name) for name in os.listdir('CLAM_DATASET') if name in TUMOR_DATASET_TYPE]
    

    return SVS_PATH, JPEG_PATH


def folder_Matching(svs_Folder, jpeg_List):
    svs_Folder_Name = svs_Folder.split('/')[1] # LUAC, SSSF, TCGA, YS
    
    for jpeg_Folder_Name in jpeg_List:
        tmp = jpeg_Folder_Name.split('/')[1] 
        if svs_Folder_Name == tmp: 
            return jpeg_Folder_Name
    
    raise ValueError('Matching 되는 폴더이름이 없습니다.')



def is_verify(patch, patch_size):
    sum_r, sum_g, sum_b = 0, 0, 0
    for pixel in patch.getdata():
        sum_r += pixel[0]
        sum_g += pixel[1] 
        sum_b += pixel[2] 

    p = patch.load() 

    total = sum_r + sum_g + sum_b

    # Empty Image  
    if total == 0 : 
        return False

    '''
        sum_r / (patch_size ** 2) >= Value 
        Value : Threshold 값 : CLAM의 결과 Heatmap에서 해당 patch_Size내에 빨간색 비율이 value값 보다 크다면 샘플링할 데이터
    '''
    if sum_r / (patch_size ** 2) >=200 and sum_g / (patch_size ** 2) <= 100 :
        return True 
    else :
        return False 
    

def is_matching(svs_name, JPEG_FOLDER):

    for jpeg_name in os.listdir(JPEG_FOLDER):

        if svs_name[:9] == jpeg_name[:9]:
            print(jpeg_name)
            return Image.open(os.path.join(JPEG_FOLDER, jpeg_name)).convert('RGB')


'''
    args : 
        svsList : DATASET/LUAC, DATASET/SSSF, DATASET/TCGA, DATASET/YS 
        jpegList:  CLAM_DATASET/LUAC, CLAM_DATASET/SSSF, CLAM_DATASET/TCGA, CLAM_DATASET/YS 

'''

def get_Current_dataInfo(svs_Image, jpeg_Image, svs_file):
    print(f'======== Current SVS JPEG Info ========= \n')
    print(f" File Name | {svs_file}")
    print(f' SVS Width | {svs_Image.level_dimensions[0][0]}    SVS Height | {svs_Image.level_dimensions[0][1]}')
    print(f' JPEG Width | {jpeg_Image.width}    JPEG Height | {jpeg_Image.height}')
    print(f' Ratio (Width, Height) | {svs_Image.level_dimensions[0][0] / jpeg_Image.width}   {svs_Image.level_dimensions[0][1] / jpeg_Image.height}')
    print('\n')
    # 비율이 16, 16으로 되는것만 잘 맵핑된거라 맵핑되지 않은 데이터셋은 우선 패치를 뽑지말자.
    
    width_ratio = round(svs_Image.level_dimensions[0][0] / jpeg_Image.width)
    height_ratio = round(svs_Image.level_dimensions[0][1] / jpeg_Image.height)
    return width_ratio, height_ratio 


def make_patch_folder(loc): 
    if not os.path.exists(loc):
        os.makedirs(loc, exist_ok=True)



def patching(slide, y, x):
    PATCH_SIZE = 32 
    level = 0
    image = slide.read_region((y * 16, x * 16), level ,(PATCH_SIZE, PATCH_SIZE))
    patch = np.array(image.convert('RGB'))

    return patch 


            
def make_patch(svsList, jpegList):
    
    #Sampling Patch Size 32 
    PATCH_SIZE = 32 
    JPEG_PATCH_SIZE = 2 


    for svs_Folder in tqdm(svsList) : 
        jpeg_Folder = folder_Matching(svs_Folder, jpegList) 
        for svs_file in tqdm(os.listdir(svs_Folder)):
            # DATASET/LUAC + LUAC_1_1.svs 
            current_svs_name = os.path.join(svs_Folder, svs_file)
            #각각 맵핑된 SVS, JPEG 파일 정보 출력
            current_svs_Image = openslide.open_slide(current_svs_name) 
            current_jpeg_Image = is_matching(svs_file, jpeg_Folder)
            print(current_jpeg_Image)

            w_r,h_r = get_Current_dataInfo(current_svs_Image, current_jpeg_Image, svs_file)
            print(w_r, h_r)
            if w_r != 16 and h_r != 16 : 
                continue 
            
            'PATCH + [LUAC, SSSF, TCGA, YS]  + [LUAC1,... SSF34, ...]'
            patch_save_folder_name = os.path.join(os.path.join('PATCH', svs_Folder.split('/')[1]), svs_file[:9])
            make_patch_folder(patch_save_folder_name)

            for h in tqdm(range(0, current_jpeg_Image.height, JPEG_PATCH_SIZE)):
                for w in range(0, current_jpeg_Image.width, JPEG_PATCH_SIZE):
                    patch = current_jpeg_Image.crop((h, w, h + JPEG_PATCH_SIZE, w + JPEG_PATCH_SIZE))

                    if is_verify(patch,JPEG_PATCH_SIZE):
                        patch = patching(current_svs_Image, h, w)

                        filename = f"{svs_file[:9]}#{h}_{w}.png"

                        Image.fromarray(patch).save(os.path.join(patch_save_folder_name, filename))
                    else : 
                        continue 

            
SVSPATH, JPEGPATH = get_Dataset_Folder_List()

make_patch(SVSPATH, JPEGPATH)