import csv
import os
import random

import cv2
from PIL import Image

random.seed(1)

output_dir = 'lights_train'

if os.path.isdir(output_dir):
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(output_dir)
os.mkdir(output_dir)

with open('labels/lights.txt', 'rU') as inputfile:  # whatever he called it
    reader = csv.reader(inputfile)
    dp = list(reader)

directory = 'train'  ##whatever place the png are in


def is_blank(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for channel in range(3):
                if img[x][y][channel] != 255:
                    return False
    return True


def in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def wheelFinder(square):
    imgs = [cv2.imread(os.path.join(directory, filename)) for filename in os.listdir(directory)]
    generated_i = 0
    for img_i in range(len(imgs)):
        if len(dp[img_i]) == 1:
            continue
        ##iterate over every pixel
        img = imgs[img_i]
        SQUARE = square
        ##grab the corresponding data points of that file ( back and front wheel  fx1, fy1, fx2, fy2, bx1, bx1, bx2, by2) - list dp
        for x in range(0, img.shape[0] - SQUARE, 20):
            for y in range(1, img.shape[1] - SQUARE, 20):
                # all values dp[a][:5] have to be inside of box, or all values of dp[a][5:]

                # dp: x1, y1, x2, y2, ... (same for wheel 2)

                box = (x, y, x + SQUARE, y + SQUARE)
                object_p1 = (int(dp[img_i][0]), int(dp[img_i][1]))
                object_p2 = (int(dp[img_i][2]), int(dp[img_i][3]))

                check = in_box(object_p1, box) and in_box(object_p2, box)

                cropped = img[y: y + SQUARE, x: x + SQUARE]

                generated_i += 1

                # don't save every False image
                #blank = is_blank(img)
                #print(blank)
                if check or random.randint(0, 300) == 0:
                    cv2.imwrite(os.path.join(output_dir, 'file{}_{}.png'.format(generated_i, check)), cropped)


for z in range(30, 100, 30):
    wheelFinder(z)
