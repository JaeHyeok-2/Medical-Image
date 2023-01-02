import os
import re
import cv2
import tifffile as tiff
from PIL import Image

# from practice import *

# 특정 col, row를 가지는 Tile의 Image를 반환하는 함수
# from demo import MAX_COL, MAX_ROW

MAX_COL, MAX_ROW = 1e5, 1e5


def select_tile(search_col, search_row, directory_root):
    global MAX_COL, MAX_ROW

    # assert 0<= col <= MAX_COL, "Exceed the maximum length of the Tile Column"
    # assert 0<= row <= MAX_ROW, "Excced the maximum length of the Tile Row"
    file_list = os.listdir(directory_root)
    num_of_tile = len(file_list)
    print("Number of Tile: ", num_of_tile)

    try:
        for file_name in file_list:
            col, row = file_name[:-4].split("_")  # col_row.svs
            print(col, row)
            if search_col == col and search_row == row:
                img = cv2.imread(file_name, cv2.IMREAD_COLOR)
                # Print image
                cv2.imshow(f"{col}_{row} Image", img)
                return img

    except:
        return NotImplemented


def make_box_area_tile(upperleft, rightdown, directory_root):
    global MAX_COL, MAX_ROW
    u_col, u_row = upperleft
    d_col, d_row = rightdown
    # assert u_col <= d_col , "Invalid range Setting "
    # assert u_row <= d_row , "Invalid range Setting "

    file_list = os.listdir(directory_root)
    num_of_tile = len(file_list)

    range_image_list = []

    STORE_PATH = f"{u_col}_{u_row}_{d_col}_{d_row}/"
    print(STORE_PATH)
    if not os.path.exists(STORE_PATH):
        os.makedirs(STORE_PATH)

    for file_name in file_list:
        col, row = file_name[:-4].split("_")
        if u_col <= int(col) <= d_col and u_row <= int(row) <= d_row:
            print(f"Saving {col}_{row}.tif ....")
            img = tiff.imread(os.path.join(directory_root, file_name))
            print(img.shape)
            # img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            range_image_list.append((col, row))  # 어떤 이미지들이 저장되어 있나 확인 하는 방식으로
            img = Image.fromarray(img)
            img.save(STORE_PATH + "%d_%d" % (int(col), int(row)) + ".png")
            # cv2.imwrite(STORE_PATH + "%d_%d" % (int(col), int(row)) + ".tif")

    print("Complete Saving Image!")
    return range_image_list


# directory = "17_256_tiled_14031/"
path = "/home/lab/Projects/Cancer_detection_Yonsei/histo/deephistopath/17_256_tiled_14301/"
# print('HI')
# print(os.getcwd())
# print(os.listdir(path))
# selected_img = select_tile(3, 3, directory)
print(" ........................................")
image_box_list = make_box_area_tile((103, 32), (201, 44), path)
print("........................................")
print("Complete!")
print(len(image_box_list))
