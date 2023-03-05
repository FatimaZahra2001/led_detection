##################################################################################################
#! author: fatima zahra al hajji
#! date : 04.03.2023
#! python script : led_classification.py
#! use : python script to detect, identify and classify LED STATE (of dyson zone) from camera & pi
##################################################################################################
import cv2
from PIL import Image
from PIL import ImageChops
import math, operator
import numpy as np
import os
import glob
import pandas as pd
import csv
import collections

#! capture() function uses the camera to take snapshots of LED light.
#! NOTE: needs to be automated.
def capture():
    # change cam value from 1 to 0 or vice versa if you get the cam not found error~
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("image")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("image", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()

#! crop_image() crops the captured frames to focus in on the LED.
#! NOTE: zeroing in on where the LED is on each frame needs to be automated and it's currently fixed.
def crop_image():
    for i in range(41,61):
        img = cv2.imread("/home/pi/picloud/N571/led_detection/snapshots/frame_" + str(i) + ".png")
        print(type(img))
        # Shape of the image
        print("Shape of the image", img.shape)
        # [rows, columns]
        crop = img[200:240, 300:330]
        print("original image....")
        print("cropped image......")
        cv2.imwrite('cropped_' + str(i) + '.png', crop)
        cv2.imshow('cropped', crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#! cal_rgb() calculates the rgb value of each cropped frame and writes the values to a csv file.
def cal_rgb():
    # dest folder:
    dest_img = "/home/pi/picloud/N571/led_detection/cropped"
    # listing files in images folder:
    img_list = os.listdir(dest_img)
    # iterating over dest_img to get the images as arrays:
    with open('data.csv', 'w') as csvfile:
        fields = ['filename', 'r', 'g', 'b']
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for image in sorted(img_list):
            # split the filename from it's extension:
            [file_name, ext] = os.path.splitext(image)
            # creating arrays for all the images:
            arr = np.array(Image.open(os.path.join(dest_img,image)))
            arr_mean = np.mean(arr, axis=(0,1))
            if len(arr_mean) == 3: #RGB case
                print(str(file_name) + ", " + str(arr_mean[0]) + ", " + str(arr_mean[1]) + ", " + str(arr_mean[2]) )
                csvfile.write((str(file_name) + ", " + str(arr_mean[0]) + ", " + str(arr_mean[1]) + ", " + str(arr_mean[2])) +"\n")

#! led_classification() classifies each detected LED light into the zones states e.g. bluetooth mode.
def led_classification():
    with open('data.csv') as file_obj:
        reader_obj = csv.reader(file_obj)

        #for row in reader_obj:
        #print(row)

    # for blue:
    br_lower = 179
    br_upper = 188
    bg_lower = 194
    bg_upper = 203
    bb_lower = 199
    bb_upper = 208
    # for green:
    gr_lower = 200
    gr_upper = 211
    gg_lower = 206
    gg_upper = 217
    gb_lower = 200
    gb_upper = 210
    # for black:
    bl_r_lower = 163
    bl_r_upper = 167
    bl_g_lower = 164
    bl_g_upper = 168
    bl_b_lower = 169
    bl_b_upper = 174
    # other colours need to be added

    data = pd.read_csv("/home/pi/picloud/N571/led_detection/data.csv")

    no_led = data.loc[(((data['r'] >= 163) & (data['r'] <= 167)) & ((data['g'] >= 164) & (data['g'] <= 168)) & ((data['b'] >= 169) & (data['b'] <= 174)))]
    blue_led = data.loc[(((data['r'] >= 179) & (data['r'] <= 188)) & ((data['g'] >= 194) & (data['g'] <= 203)) & ((data['b'] >= 199) & (data['b'] <= 208)))]
    green_led = data.loc[(((data['r'] >= 200) & (data['r'] <= 211)) & ((data['g'] >= 206) & (data['g'] <= 217)) & ((data['b'] >= 200) & (data['b'] <= 210)))]

    """
    if no_led:
        print("LED state is OFF...")
    elif blue_led:
        print("LED state is BLUETOOTH...")
    elif green_led:
        print("LED state is ON.....")
    """
    print("classification from compiled csv file....")
    print("LED OFF STATE:")
    print(no_led)
    print("LED BLUETOOTH STATE:")
    print(blue_led)
    print("LED ON STATE:")
    print(green_led)
    print("classification complete...")

#print("cropping frames...")
#crop_image()
#print("calculating rgb values....")
#cal_rgb()
print("rgb comparison....")
led_identification()

