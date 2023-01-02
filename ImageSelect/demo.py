import slideio
import openslide 
import numpy as np 
import matplotlib.pyplot as plt

from preprocessing import * 
from tqdm import tqdm 
import os 
from PIL import Image
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator 

import argparse
IMAGE_PATH = "../../image_data/"
IMAGE_NAME = "SS19-14301#A1#NGS#_195021.svs" 

parser = argparse.ArgumentParser(description="Input Tile Size, Overlap") 
parser.add_argument("--tile_size",type=int, default=256, help="Tile Size per Image")
parser.add_argument("--overlap",type=int, default=0, help="How many overlap Tile") 
 

args = parser.parse_args()
print(args.tile_size)
print(args.overlap)

slide = open_slide(os.path.join(IMAGE_PATH, IMAGE_NAME))
tiles = DeepZoomGenerator(slide, tile_size=args.tile_size, overlap=args.overlap, limit_bounds=False) 

nth_tile_layer = 17
MAX_COL, MAX_ROW = tiles.level_tiles[nth_tile_layer]
print(f' Number of C,W : {MAX_COL, MAX_ROW}')

TILE_DIR = "17_256_tiled_" + IMAGE_NAME[5:10] + "/"

os.makedirs(TILE_DIR, exist_ok=True) 

for row in range(MAX_ROW):
  for col in range(MAX_COL): 
    tile_name = os.path.join(TILE_DIR, "%d_%d" %(col, row)) 
    print(f"Now Saving TIle with Title: {tile_name}")
    temp_tile = tiles.get_tile(nth_tile_layer, (col,row)) 
    temp_tile_RGB = temp_tile.convert('RGB')
    temp_tile_np = np.array(temp_tile_RGB)
    temp_tile_save = Image.fromarray(temp_tile_np)
    temp_tile_save.save(tile_name + ".tif")


