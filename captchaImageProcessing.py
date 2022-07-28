
# The Captcha image consists of four characters that are either a string of letters or a combination of letters with numbers. In some of the combinations, the captcha characters contain all letters but they are not aligned cleanly. Some of the letters are joined together or diagonally aligned, therefore correctly separating them in order to
# make them easily recognised as letters is one of the tasks at hand. In the instances of Captcha characters that contain both numbers and letters, the position of the number is not guaranteed and the letters are slanted.
# The expected outcome is to correctly identify each of the Captcha characters from the image. We are doing this inorder to measure the efficiency of the neural networks to correctly identify the Captcha characters. The label variable is the letter or number from
# the image of the separated Captcha characters.

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
import os


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

!pip install opencv
conda install -c conda-forge/label/gcc7 opencv
conda install -c menpo opencv
import zipfile
with zipfile.ZipFile("archive.zip","r") as zip_ref:
    zip_ref.extractall("targetdir")


import os

os.getcwd()

from PIL import Image
import os, os.path
imgs = []
path = "/home/LC/chakth01/Neural networks"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))


#image_file_Path ="/home/LC/chakth01/Neural networks/targetdir/captcha_images/"
image_file_Path ="/home/LC/mbuako01/DS 420 Project 2/imageFolder/captcha_images/"

image_file_Path

def read_image(image_file_path):
    """Read in an image file."""
    bgr_img = cv2.imread(image_file_path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    
    return rgb_img

import cv2
import imutils
import numpy as np
import os
from imutils import paths
import pandas as pd

images = []
labels = []

for image_file_path in imutils.paths.list_images(image_file_Path):
    image_file = read_image(image_file_path)
    label = image_file_path.split('/')[7]
    images.append(image_file)
    labels.append(label)

#images

#labels

newLabels = []

for label in labels:
    labelHere = label.split('.')[0]
    newLabels.append(labelHere)

#newLabels

images = np.array(images)
#images4Plot = np.array(images, dtype="float") / 255.0
labels = np.array(newLabels)

print(labels)

#print(images)

#A dataset of 9,955 of unique CAPTCHA images each with its label as the filename was used for this research. However, machine learning classification requires a one-to-many relationship between a label and in this context the CAPTCHA images. Therefore, uniqueness of the CAPTCHA images is problematic for a machine learning process.


images.shape

some_digit = images[300]
#Some_digit_image = some_digit.reshape(24, 72, 3)
plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()

labels[300]

import os
import os.path
import cv2
import glob
import imutils

def pureBlackWhiteConversionThreshold(image):
    # Add some extra padding around the image
    imagePadded = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(imagePadded, cv2.COLOR_RGB2GRAY)
    # threshold the image (convert it to pure black and white)
    imagethresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    
    return imagethresholded

def pureBlackWhiteConversionOGImage(image):
    # Add some extra padding around the image
    imagePadded = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(imagePadded, cv2.COLOR_RGB2GRAY)
       
    return gray

padded_ThreshImage300 = pureBlackWhiteConversionThreshold(images[300])

some_digit = padded_ThreshImage300

plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()

def regionsOfLetters(image):
    
     # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []
    
    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))
    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    #if len(letter_image_regions) != 4:
       # continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    return letter_image_regions

letter_image_regions = regionsOfLetters(padded_ThreshImage300)
letter_image_regions

#To deal with the uniqueness problem of the dataset, the solution was to separate  the CAPTCHA images into the individual 4 characters that make up the CHAPTCHA image. This was to make each character into its own image. The resulting dataset has 39,754 images with one character per image. The new dataset satisfies the one-to-many relationship between the images and the following 32 characters labels {'2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}


def extractLetters(letter_image_regions, image):
    # Save out each letter as a single image
    letter_images =[]
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        #image_file1 = read_image(letter_image)
        letter_images.append(letter_image)
    return letter_images

grayScaleImage = pureBlackWhiteConversionOGImage(images[300])
letter_image_List = extractLetters(letter_image_regions,grayScaleImage)

checkImage = letter_image_List[3]

some_digit = checkImage
#Some_digit_image = some_digit.reshape(24, 72, 3)
plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()

len(labels)

len(images)

images.shape

def expand2square(image):
    desired_size = 28
    im = image
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

letterImages = []
letterImageLabels = []


# loop over the unseparated image list
for label, image in zip(labels, images):
    
    padded_ThreshImage300 = pureBlackWhiteConversionThreshold(image)
    letter_image_regions = regionsOfLetters(padded_ThreshImage300)
    grayScaleImage = pureBlackWhiteConversionOGImage(image)
    letter_image_List = extractLetters(letter_image_regions,grayScaleImage)
     #image_reshape = letter_bounding_box.reshape(L0, L1)
    for letter_bounding_box, letter_text in zip(letter_image_List, label): 
        L0=letter_bounding_box.shape[0]
        L1=letter_bounding_box.shape[1]
       
        imageResize = expand2square(letter_bounding_box)
        letterImages.append(imageResize)
        letterImageLabels.append(letter_text)

len(letterImages)

len(letterImageLabels)

displayDigit = letterImages[609]

plt.imshow(displayDigit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()

letterImages[609].shape

images = np.array(letterImages)
labels = np.array(letterImageLabels)

images.shape

def plot_digit(image):
    some_digit = image
    plt.imshow(some_digit, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


    plt.show()

# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

index, = np.where(labels == 'F')
index

plt.figure(figsize=(15, 15))
example_images = np.r_[images[[14,39,51,39702,39752]], 
                       images[[7,56,61,39703,39714]],
                       images[[45,   198,   352,39705, 39719]], 
                       images[[2,    12,    52, 39698, 39712]], 
                       images[[3,    26,    87, 39612, 39619]]]

example_images
plot_digits(example_images, images_per_row=5)
#save_fig("more_digits_plot")
#plt.show()
