from PIL import Image
from PIL import ImageChops
import math, operator
import numpy as np
import os
import cv2

# im = Image.open('/home/pi/picloud/N571/led_detection/pictures/frame_54.png')
im = Image.open('/home/pi/picloud/N571/led_detection/frame_1.png')
pix = im.load()
w, h = im.size
print(im.size)  # get the width and hight of the image for iterating over
x = im.size[0]
y = im.size[1]
print(x)
pix_val = list(im.getdata())


# print(pix_val)


def rms_diff(img1, img2):
    print("calculating the root-mean-square difference between two images")

    #  h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    # return math.sqrt(reduce(operator.add,
    #    map(lambda h, i: h*(i**2), h, range(256)))
    #   / (float(im1.size[0]) * im1.size[1]))

    diff = ImageChops.difference(img1, img2)
    print(diff)


def image_to_array(image):
    # load jpg / png into 3D numpy array of shape (width, height, channels)
    with Image.open(image) as image:
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3)) / 255
    return im_arr


def crop_image():
    img = cv2.imread("/home/pi/picloud/N571/led_detection/snapshots/frame_50.png")
    print(type(img))

    # Shape of the image
    print("Shape of the image", img.shape)

    # [rows, columns]
    crop = img[190:250, 320:350]

    print("original image....")

    cv2.imshow('original', img)

    print("cropped image......")
    cv2.imshow('cropped', crop)
    cv2.imwrite('cropped_50.png', crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_rgb():
    # dest folder:
    dest_img = "/home/pi/picloud/N571/led_detection/pictures"
    # listing files in images folder:
    img_list = os.listdir(dest_img)
    # iterating over dest_img to get the images as arrays:
    for image in sorted(img_list):
        # split the filename from it's extension:
        [file_name, ext] = os.path.splitext(image)
        # creating arrays for all the images:
        arr = np.array(Image.open(os.path.join(dest_img, image)))
        arr_mean = np.mean(arr, axis=(0, 1))
        if len(arr_mean) == 3:  # RGB case
            print(f'[{file_name}, R={arr_mean[0]:.1f},  G={arr_mean[1]:.1f}, B={arr_mean[2]:.1f} ]')


# else: #ALPHA CASE, transparency measure at the end
# print(f'[{file_name}, R={arr_mean[0]:.1f}, G={arr_mean[1]:.1f}, B={arr_mean[2]:.1f}, ALPHA={arr_mean[3]:.1f}]')

# cropping the image
print("cropping the image......")
crop_image()

# method 2:
# imag1 = image_to_array("/home/pi/picloud/N571/led_detection/pictures/frame_56.png")
# imag2 = image_to_array("/home/pi/picloud/N571/led_detection/pictures/frame_0.png")
# diff = np.absolute(imag1 - imag2)
# print("sum of differences between R G B values: ")
# print(np.sum(diff, axis=(0,1)))
# method 1:
# img1 = Image.open("/home/pi/picloud/N571/led_detection/pictures/frame_56.png")
# img2 = Image.open("/home/pi/picloud/N571/led_detection/pictures/frame_54.png")
# img2 = Image.open("/home/pi/picloud/N571/led_detection/pictures/frame_22.png")
# img2 = Image.open("/home/pi/picloud/N571/led_detection/frame_0.png")
# print(rms_diff(img1, img2))

# finding RGB values of images:
# print("finding RGB values of images...")
# cal_rgb()
